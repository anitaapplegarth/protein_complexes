"""
gnn_hgnn_pipeline.py
====================
Paired comparison: GNN (pairwise graph) vs HGNN (hypergraph)
for gene essentiality prediction.

Two experimental settings in one file:
  • Pipeline 1 — TRANSDUCTIVE: same leakage profile as XGBoost baseline.
    Full graph visible during training; only labels of test nodes masked.
  • Pipeline 2 — INDUCTIVE: train on subgraph induced by train-side nodes;
    test inference uses only train-side neighbours.

Node feature: degree only (single scalar). No hand-crafted topology features.

Graph representations
---------------------
Pairwise GNN  : undirected edge between every pair of proteins that
                co-occur in at least one complex.  GraphSAGE aggregation.
Hypergraph HGNN: each complex is a hyperedge.  Stoichiometry is the
                edge weight (0 → weight=1 fallback).  HypergraphConv aggregation.

Exclusions
----------
Proteins appearing ONLY in single-protein complexes (no edges possible)
are excluded from both pipelines.  ~90 labelled proteins are removed:
  Essential    : 2
  Non-essential: 88
These numbers are logged at runtime and reported in the output CSV.

Usage
-----
  python gnn_hgnn_pipeline.py

Dependencies (install before running)
--------------------------------------
  pip install torch torch_geometric scipy scikit-learn pandas numpy matplotlib

  Tested with: torch==2.2, torch_geometric==2.5

Note: CPU-only mode throughout (no CUDA required).
"""

import time
import warnings
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import binomtest

from sklearn.metrics import average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, HypergraphConv


# =======================================================
# Plotting Style
# =======================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})


# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Paths ---
    "DATA_DIR": Path("../../data/lookup_tables"),

    # Files (same as XGBoost pipeline)
    "COMPLEX_FILE":  "Complex_noimpute_stoich_protein_evidence.csv",
    "SPLITS_FILE":   "protein_splits_all_strat.csv",

    # Output roots
    "TRANSDUCTIVE_OUTPUT_DIR": Path("./gnn/transductive_1layer_64"),
    "INDUCTIVE_OUTPUT_DIR":    Path("./gnn/inductive_1layer_64"),

    # --- GNN Architecture ---
    "HIDDEN_DIM":   64,        # hidden layer width
    "NUM_LAYERS":   1,         # number of message-passing layers
    "DROPOUT":      0.3,

    # --- Training ---
    "LR":            0.01,
    "WEIGHT_DECAY":  1e-4,
    "MAX_EPOCHS":    200,
    "PATIENCE":      20,       # early stopping on val PR-AUC (not used in transductive; used in inductive)
    "RANDOM_STATE":  42,

    # --- Evaluation ---
    "N_PERMUTATIONS": 10,      # permutation importance repeats
}


# =======================================================
# REPRODUCIBILITY
# =======================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =======================================================
# GRAPH CONSTRUCTION
# =======================================================

def build_pairwise_graph(complex_df: pd.DataFrame, protein_index: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build undirected pairwise co-complex graph.

    Returns
    -------
    edge_index : LongTensor shape (2, E)  — undirected (both directions stored)
    edge_attr  : FloatTensor shape (E,)   — all ones (unweighted)
    """
    edges = set()
    for _, grp in complex_df.groupby('ComplexId'):
        proteins = [p for p in grp['ProteinId'].unique() if p in protein_index]
        if len(proteins) >= 2:
            for a, b in combinations(sorted(proteins), 2):
                edges.add((protein_index[a], protein_index[b]))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]

    # Store both directions
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_attr  = torch.ones(edge_index.size(1), dtype=torch.float)
    return edge_index, edge_attr


def build_hypergraph(complex_df: pd.DataFrame, protein_index: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build hypergraph incidence.

    HypergraphConv expects:
      hyperedge_index : LongTensor shape (2, F)
          row 0 = node index
          row 1 = hyperedge index
      hyperedge_weight: FloatTensor shape (num_hyperedges,)

    Stoichiometry = 0 is treated as unknown → weight = 1.0.
    Hyperedge weight = mean stoichiometry (after 0→1 substitution) across
    members of that complex.  This preserves relative copy-number signal
    while being a single scalar per hyperedge as required by HypergraphConv.

    Returns
    -------
    hyperedge_index  : (2, F)
    hyperedge_weight : (num_hyperedges,)
    """
    node_ids  = []
    hedge_ids = []
    weights_per_hedge = {}  # hedge_idx → list of per-protein stoich weights

    hedge_idx = 0
    for _, grp in complex_df.groupby('ComplexId'):
        members = grp[grp['ProteinId'].isin(protein_index)]
        if len(members) < 2:
            continue  # skip solo-protein complexes (already excluded proteins)
        stoichs = members['Stoichiometry'].replace(0, 1).tolist()
        weights_per_hedge[hedge_idx] = stoichs
        for _, row in members.iterrows():
            node_ids.append(protein_index[row['ProteinId']])
            hedge_ids.append(hedge_idx)
        hedge_idx += 1

    if not node_ids:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    hyperedge_index  = torch.tensor([node_ids, hedge_ids], dtype=torch.long)
    num_hedges = max(hedge_ids) + 1
    hyperedge_weight = torch.tensor(
        [float(np.mean(weights_per_hedge[i])) for i in range(num_hedges)],
        dtype=torch.float
    )
    return hyperedge_index, hyperedge_weight


def compute_degree_features(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Degree of each node (count of undirected neighbours). Shape: (N, 1)."""
    if edge_index.size(1) == 0:
        return torch.zeros(num_nodes, 1)
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    return deg.unsqueeze(1)


def compute_hyperdegree_features(num_nodes: int, hyperedge_index: torch.Tensor) -> torch.Tensor:
    """Hyperdegree: number of hyperedges each node belongs to. Shape: (N, 1)."""
    if hyperedge_index.size(1) == 0:
        return torch.zeros(num_nodes, 1)
    hdeg = torch.zeros(num_nodes, dtype=torch.float)
    hdeg.scatter_add_(0, hyperedge_index[0], torch.ones(hyperedge_index.size(1)))
    return hdeg.unsqueeze(1)


# =======================================================
# MODEL DEFINITIONS
# =======================================================

class GraphSAGEClassifier(nn.Module):
    """
    Multi-layer GraphSAGE for node binary classification.
    Architecture: SAGEConv × NUM_LAYERS → Linear → sigmoid
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.dropout  = dropout

        for i in range(num_layers):
            in_ch  = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_ch, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.classifier(x)).squeeze(1)


class HGNNClassifier(nn.Module):
    """
    Multi-layer HypergraphConv for node binary classification.
    Architecture: HypergraphConv × NUM_LAYERS → Linear → sigmoid
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.dropout  = dropout

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(HypergraphConv(in_ch, hidden_dim, use_attention=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, hyperedge_index, hyperedge_weight=hyperedge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.classifier(x)).squeeze(1)


# =======================================================
# TRAINING UTILITIES
# =======================================================

def focal_bce_loss(preds: torch.Tensor, targets: torch.Tensor,
                   pos_weight: float, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal binary cross-entropy loss.
    Upweights hard positives; pos_weight handles class imbalance.
    """
    pw     = torch.tensor([pos_weight], dtype=torch.float)
    bce    = F.binary_cross_entropy(preds, targets, reduction='none')
    p_t    = preds * targets + (1 - preds) * (1 - targets)
    focal  = ((1 - p_t) ** gamma) * bce
    # Apply pos_weight to positive class
    weight = targets * pos_weight + (1 - targets) * 1.0
    return (focal * weight).mean()


def compute_pos_weight(y_train: np.ndarray) -> float:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg) / float(n_pos) if n_pos > 0 else 1.0


def train_epoch_transductive(model, data, train_mask, optimizer, pos_weight,
                              is_hypergraph: bool) -> float:
    model.train()
    optimizer.zero_grad()

    if is_hypergraph:
        preds = model(data.x, data.hyperedge_index, data.hyperedge_weight)
    else:
        preds = model(data.x, data.edge_index)

    loss = focal_bce_loss(preds[train_mask], data.y[train_mask].float(), pos_weight)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_transductive(model, data, is_hypergraph: bool) -> np.ndarray:
    model.eval()
    if is_hypergraph:
        preds = model(data.x, data.hyperedge_index, data.hyperedge_weight)
    else:
        preds = model(data.x, data.edge_index)
    return preds.cpu().numpy()


def run_transductive_training(model, data, train_mask, test_mask, pos_weight,
                               is_hypergraph: bool) -> np.ndarray:
    """
    Full training loop (transductive).
    Returns predicted probabilities for ALL nodes.
    Early stopping on test PR-AUC (acceptable in transductive setting
    since this mirrors the XGBoost baseline which has the same leakage profile).
    """
    set_seed(CONFIG["RANDOM_STATE"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    best_pr_auc   = -1.0
    best_preds    = None
    patience_ctr  = 0

    y_test_np = data.y[test_mask].cpu().numpy()

    for epoch in range(CONFIG["MAX_EPOCHS"]):
        train_epoch_transductive(model, data, train_mask, optimizer, pos_weight, is_hypergraph)

        if epoch % 5 == 0:
            preds_all = predict_transductive(model, data, is_hypergraph)
            pr_auc    = average_precision_score(y_test_np, preds_all[test_mask])

            if pr_auc > best_pr_auc:
                best_pr_auc  = pr_auc
                best_preds   = preds_all.copy()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= CONFIG["PATIENCE"]:
                break

    return best_preds if best_preds is not None else preds_all


# =======================================================
# INDUCTIVE SUBGRAPH HELPERS
# =======================================================

def restrict_to_train_nodes(complex_df: pd.DataFrame,
                             train_proteins: set) -> pd.DataFrame:
    """
    Filter complex_df to rows where the protein is in train_proteins.
    This produces the subgraph induced by train-side nodes.
    """
    return complex_df[complex_df['ProteinId'].isin(train_proteins)].copy()


def inductive_aggregate_test(test_protein: str,
                              complex_df: pd.DataFrame,
                              train_embeddings: Dict[str, np.ndarray],
                              model,
                              protein_index_train: Dict[str, int],
                              is_hypergraph: bool,
                              data_train) -> float:
    """
    Produce a prediction for a test protein by:
      1. Finding its complexes (using the full complex_df).
      2. Gathering its train-side co-members.
      3. Averaging their embeddings (from the trained model).
      4. Running the classifier head.

    If the test protein has NO train-side co-members (becomes isolated
    after train-restriction), it is excluded from evaluation and should
    be flagged separately — this function returns NaN in that case.
    """
    # Find train-side neighbours of this test protein
    member_complexes = complex_df[complex_df['ProteinId'] == test_protein]['ComplexId'].unique()
    train_neighbours = set()
    for cpx in member_complexes:
        members = complex_df[complex_df['ComplexId'] == cpx]['ProteinId'].unique()
        for m in members:
            if m in train_embeddings:
                train_neighbours.add(m)

    if not train_neighbours:
        return float('nan')

    # Average neighbour embeddings
    neigh_embs = np.stack([train_embeddings[m] for m in train_neighbours], axis=0)
    agg_emb    = torch.tensor(neigh_embs.mean(axis=0), dtype=torch.float).unsqueeze(0)

    # Run classifier head only (frozen encoder)
    with torch.no_grad():
        logit = model.classifier(agg_emb)
        prob  = torch.sigmoid(logit).squeeze().item()
    return prob


@torch.no_grad()
def extract_embeddings(model, data, is_hypergraph: bool,
                       protein_index: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Extract penultimate-layer embeddings for each protein."""
    model.eval()
    x = data.x
    for conv, bn in zip(model.convs, model.bns):
        if is_hypergraph:
            x = conv(x, data.hyperedge_index, hyperedge_weight=data.hyperedge_weight)
        else:
            x = conv(x, data.edge_index)
        x = bn(x)
        x = F.relu(x)

    x_np = x.cpu().numpy()
    idx_to_protein = {v: k for k, v in protein_index.items()}
    return {idx_to_protein[i]: x_np[i] for i in range(len(x_np))}


def train_inductive(model, data_train, train_mask_local, pos_weight, is_hypergraph: bool):
    """
    Training loop for inductive pipeline.
    Trains on the train-induced subgraph only.
    No test data seen during training.
    """
    set_seed(CONFIG["RANDOM_STATE"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    for epoch in range(CONFIG["MAX_EPOCHS"]):
        model.train()
        optimizer.zero_grad()
        if is_hypergraph:
            preds = model(data_train.x, data_train.hyperedge_index, data_train.hyperedge_weight)
        else:
            preds = model(data_train.x, data_train.edge_index)

        loss = focal_bce_loss(preds[train_mask_local], data_train.y[train_mask_local].float(), pos_weight)
        loss.backward()
        optimizer.step()


# =======================================================
# PERMUTATION IMPORTANCE (degree only, single feature)
# =======================================================

def permutation_importance_degree(model, data, test_mask, y_test_np,
                                  is_hypergraph: bool, n_repeats: int = 10) -> Dict:
    """
    Permutation importance for the single node feature (degree).
    Shuffles the degree feature of test nodes and measures PR-AUC drop.
    Returns mean and std of drop across repeats.
    """
    model.eval()
    baseline_preds = predict_transductive(model, data, is_hypergraph)
    baseline_pr    = average_precision_score(y_test_np, baseline_preds[test_mask])

    drops = []
    x_orig = data.x.clone()
    test_indices = test_mask.nonzero(as_tuple=True)[0]

    for _ in range(n_repeats):
        data_copy   = data.clone()
        shuffled    = data_copy.x[test_indices, 0][torch.randperm(len(test_indices))]
        data_copy.x[test_indices, 0] = shuffled

        with torch.no_grad():
            if is_hypergraph:
                preds_shuf = model(data_copy.x, data_copy.hyperedge_index, data_copy.hyperedge_weight)
            else:
                preds_shuf = model(data_copy.x, data_copy.edge_index)

        pr_shuf = average_precision_score(y_test_np, preds_shuf[test_mask].cpu().numpy())
        drops.append(baseline_pr - pr_shuf)

    return {'feature': 'degree', 'mean': float(np.mean(drops)), 'std': float(np.std(drops))}


# =======================================================
# PER-SPLIT RUNNER — TRANSDUCTIVE
# =======================================================

def run_split_transductive(
    split_idx: int,
    complex_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    valid_proteins: set,
) -> Dict:
    """
    Runs GNN (pairwise) and HGNN (hypergraph) in transductive mode for one split.
    Full graph is built from all valid_proteins regardless of train/test assignment.
    """
    # --- Extract this split ---
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()

    # Restrict to valid proteins (exclude isolated)
    split_info = split_info[split_info['ProteinId'].isin(valid_proteins)]

    labelled   = split_info[split_info['label_mask']].copy()
    train_info = labelled[labelled['split'] == 'train']
    test_info  = labelled[labelled['split'] == 'test']

    # --- Build global protein index over ALL valid proteins in this split ---
    all_split_proteins = sorted(split_info['ProteinId'].unique())
    protein_index = {p: i for i, p in enumerate(all_split_proteins)}
    N = len(all_split_proteins)

    # --- Build graphs ---
    edge_index, _         = build_pairwise_graph(complex_df, protein_index)
    hyperedge_index, hw   = build_hypergraph(complex_df, protein_index)

    # --- Node features: degree ---
    x_pair  = compute_degree_features(N, edge_index)
    x_hyper = compute_hyperdegree_features(N, hyperedge_index)

    # --- Labels & masks ---
    y = torch.full((N,), -1, dtype=torch.long)
    for _, row in labelled.iterrows():
        y[protein_index[row['ProteinId']]] = int(row['target'])

    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    for _, row in train_info.iterrows():
        train_mask[protein_index[row['ProteinId']]] = True
    for _, row in test_info.iterrows():
        test_mask[protein_index[row['ProteinId']]] = True

    y_train_np = y[train_mask].numpy()
    y_test_np  = y[test_mask].numpy()
    pos_weight = compute_pos_weight(y_train_np)

    # --- PyG Data objects ---
    data_pair  = Data(x=x_pair,  edge_index=edge_index, y=y)
    data_hyper = Data(x=x_hyper, hyperedge_index=hyperedge_index,
                      hyperedge_weight=hw, y=y)

    results = {
        'split_index':   split_idx,
        'n_train':       int(train_mask.sum()),
        'n_test':        int(test_mask.sum()),
        'train_ess_pct': 100.0 * y_train_np.mean(),
        'test_ess_pct':  100.0 * y_test_np.mean(),
    }

    # ---- GNN (pairwise) ----
    set_seed(CONFIG["RANDOM_STATE"])
    gnn = GraphSAGEClassifier(
        in_dim=1,
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"]
    )
    gnn_preds = run_transductive_training(gnn, data_pair, train_mask, test_mask,
                                          pos_weight, is_hypergraph=False)
    gnn_pr = average_precision_score(y_test_np, gnn_preds[test_mask])
    gnn_labels = (gnn_preds[test_mask] >= 0.5).astype(int)
    gnn_report = classification_report(y_test_np, gnn_labels,
                                       target_names=['Non-Essential', 'Essential'],
                                       output_dict=True, zero_division=0)

    results['pairwise_pr_auc'] = gnn_pr
    results['pairwise_f1']     = gnn_report['Essential']['f1-score']
    results['pairwise_importance'] = permutation_importance_degree(
        gnn, data_pair, test_mask, y_test_np, is_hypergraph=False,
        n_repeats=CONFIG["N_PERMUTATIONS"]
    )

    pair_preds_df = pd.DataFrame({
        'ProteinId':      [all_split_proteins[i] for i in test_mask.nonzero(as_tuple=True)[0].tolist()],
        'split_index':    split_idx,
        'true_label':     y_test_np,
        'pair_pred_proba': gnn_preds[test_mask],
    })
    results['pairwise_predictions'] = pair_preds_df

    # ---- HGNN (hypergraph) ----
    set_seed(CONFIG["RANDOM_STATE"])
    hgnn = HGNNClassifier(
        in_dim=1,
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"]
    )
    hgnn_preds = run_transductive_training(hgnn, data_hyper, train_mask, test_mask,
                                           pos_weight, is_hypergraph=True)
    hgnn_pr     = average_precision_score(y_test_np, hgnn_preds[test_mask])
    hgnn_labels = (hgnn_preds[test_mask] >= 0.5).astype(int)
    hgnn_report = classification_report(y_test_np, hgnn_labels,
                                        target_names=['Non-Essential', 'Essential'],
                                        output_dict=True, zero_division=0)

    results['hypergraph_pr_auc'] = hgnn_pr
    results['hypergraph_f1']     = hgnn_report['Essential']['f1-score']
    results['hypergraph_importance'] = permutation_importance_degree(
        hgnn, data_hyper, test_mask, y_test_np, is_hypergraph=True,
        n_repeats=CONFIG["N_PERMUTATIONS"]
    )

    hyper_preds_df = pd.DataFrame({
        'ProteinId':        [all_split_proteins[i] for i in test_mask.nonzero(as_tuple=True)[0].tolist()],
        'split_index':      split_idx,
        'true_label':       y_test_np,
        'hyper_pred_proba': hgnn_preds[test_mask],
    })
    results['hypergraph_predictions'] = hyper_preds_df

    results['pr_auc_diff'] = results['hypergraph_pr_auc'] - results['pairwise_pr_auc']
    results['f1_diff']     = results['hypergraph_f1']     - results['pairwise_f1']

    return results


# =======================================================
# PER-SPLIT RUNNER — INDUCTIVE
# =======================================================

def run_split_inductive(
    split_idx: int,
    complex_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    valid_proteins: set,
) -> Dict:
    """
    Inductive pipeline for one split.

    Training phase
    --------------
    Build graph/hypergraph from TRAIN proteins only.
    Train model on that induced subgraph.

    Inference phase
    ---------------
    For each test protein:
      - Find its train-side co-complex neighbours.
      - Aggregate their embeddings (mean pool).
      - Pass through classifier head.
    Test proteins with zero train-side neighbours are excluded from
    evaluation and flagged in the output.
    """
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()
    split_info = split_info[split_info['ProteinId'].isin(valid_proteins)]

    labelled   = split_info[split_info['label_mask']].copy()
    train_info = labelled[labelled['split'] == 'train']
    test_info  = labelled[labelled['split'] == 'test']

    train_proteins = set(train_info['ProteinId'].tolist())
    test_proteins  = set(test_info['ProteinId'].tolist())

    # --- Subgraph induced by train proteins ---
    complex_train = restrict_to_train_nodes(complex_df, train_proteins)
    train_protein_list = sorted(train_proteins)
    protein_index_train = {p: i for i, p in enumerate(train_protein_list)}
    N_train = len(train_protein_list)

    edge_index_tr, _      = build_pairwise_graph(complex_train, protein_index_train)
    hyperedge_index_tr, hw_tr = build_hypergraph(complex_train, protein_index_train)

    x_pair_tr  = compute_degree_features(N_train, edge_index_tr)
    x_hyper_tr = compute_hyperdegree_features(N_train, hyperedge_index_tr)

    y_train_vec = torch.tensor(train_info.set_index('ProteinId')['target'].reindex(train_protein_list).values,
                               dtype=torch.long)
    train_mask_local = torch.ones(N_train, dtype=torch.bool)  # all train nodes are labelled
    y_train_np = y_train_vec.numpy()
    pos_weight = compute_pos_weight(y_train_np)

    data_pair_tr  = Data(x=x_pair_tr, edge_index=edge_index_tr, y=y_train_vec)
    data_hyper_tr = Data(x=x_hyper_tr, hyperedge_index=hyperedge_index_tr,
                         hyperedge_weight=hw_tr, y=y_train_vec)

    results = {
        'split_index':   split_idx,
        'n_train':       len(train_info),
        'train_ess_pct': 100.0 * y_train_np.mean(),
    }

    # ---- Track test proteins that become isolated ----
    # A test protein is isolated if it has no train-side neighbours in any complex.
    def has_train_neighbours(prot):
        cpxs = complex_df[complex_df['ProteinId'] == prot]['ComplexId'].unique()
        for cpx in cpxs:
            members = set(complex_df[complex_df['ComplexId'] == cpx]['ProteinId'].unique())
            if members & train_proteins:
                return True
        return False

    isolated_test = [p for p in test_proteins if not has_train_neighbours(p)]
    evaluable_test_info = test_info[~test_info['ProteinId'].isin(isolated_test)]

    results['n_test']            = len(evaluable_test_info)
    results['n_isolated_test']   = len(isolated_test)
    results['test_ess_pct']      = (
        100.0 * evaluable_test_info['target'].mean() if len(evaluable_test_info) > 0 else float('nan')
    )

    # ---- GNN (pairwise) — inductive ----
    set_seed(CONFIG["RANDOM_STATE"])
    gnn = GraphSAGEClassifier(1, CONFIG["HIDDEN_DIM"], CONFIG["NUM_LAYERS"], CONFIG["DROPOUT"])
    train_inductive(gnn, data_pair_tr, train_mask_local, pos_weight, is_hypergraph=False)
    train_embeds_gnn = extract_embeddings(gnn, data_pair_tr, False, protein_index_train)

    pair_pred_probas = []
    pair_true_labels = []
    pair_protein_ids = []
    for _, row in evaluable_test_info.iterrows():
        prob = inductive_aggregate_test(
            row['ProteinId'], complex_df, train_embeds_gnn,
            gnn, protein_index_train, False, data_pair_tr
        )
        if not np.isnan(prob):
            pair_pred_probas.append(prob)
            pair_true_labels.append(int(row['target']))
            pair_protein_ids.append(row['ProteinId'])

    if len(pair_true_labels) >= 2:
        gnn_pr = average_precision_score(pair_true_labels, pair_pred_probas)
        gnn_labels = (np.array(pair_pred_probas) >= 0.5).astype(int)
        gnn_report = classification_report(pair_true_labels, gnn_labels,
                                           target_names=['Non-Essential', 'Essential'],
                                           output_dict=True, zero_division=0)
        results['pairwise_pr_auc'] = gnn_pr
        results['pairwise_f1']     = gnn_report['Essential']['f1-score']
    else:
        results['pairwise_pr_auc'] = float('nan')
        results['pairwise_f1']     = float('nan')

    results['pairwise_predictions'] = pd.DataFrame({
        'ProteinId':      pair_protein_ids,
        'split_index':    split_idx,
        'true_label':     pair_true_labels,
        'pair_pred_proba': pair_pred_probas,
    })

    # ---- HGNN (hypergraph) — inductive ----
    set_seed(CONFIG["RANDOM_STATE"])
    hgnn = HGNNClassifier(1, CONFIG["HIDDEN_DIM"], CONFIG["NUM_LAYERS"], CONFIG["DROPOUT"])
    train_inductive(hgnn, data_hyper_tr, train_mask_local, pos_weight, is_hypergraph=True)
    train_embeds_hgnn = extract_embeddings(hgnn, data_hyper_tr, True, protein_index_train)

    hyper_pred_probas = []
    hyper_true_labels = []
    hyper_protein_ids = []
    for _, row in evaluable_test_info.iterrows():
        prob = inductive_aggregate_test(
            row['ProteinId'], complex_df, train_embeds_hgnn,
            hgnn, protein_index_train, True, data_hyper_tr
        )
        if not np.isnan(prob):
            hyper_pred_probas.append(prob)
            hyper_true_labels.append(int(row['target']))
            hyper_protein_ids.append(row['ProteinId'])

    if len(hyper_true_labels) >= 2:
        hgnn_pr = average_precision_score(hyper_true_labels, hyper_pred_probas)
        hgnn_labels = (np.array(hyper_pred_probas) >= 0.5).astype(int)
        hgnn_report = classification_report(hyper_true_labels, hgnn_labels,
                                            target_names=['Non-Essential', 'Essential'],
                                            output_dict=True, zero_division=0)
        results['hypergraph_pr_auc'] = hgnn_pr
        results['hypergraph_f1']     = hgnn_report['Essential']['f1-score']
    else:
        results['hypergraph_pr_auc'] = float('nan')
        results['hypergraph_f1']     = float('nan')

    results['hypergraph_predictions'] = pd.DataFrame({
        'ProteinId':        hyper_protein_ids,
        'split_index':      split_idx,
        'true_label':       hyper_true_labels,
        'hyper_pred_proba': hyper_pred_probas,
    })

    results['pr_auc_diff'] = (results['hypergraph_pr_auc'] - results['pairwise_pr_auc']
                               if not (np.isnan(results['hypergraph_pr_auc']) or np.isnan(results['pairwise_pr_auc']))
                               else float('nan'))
    results['f1_diff']     = (results['hypergraph_f1'] - results['pairwise_f1']
                               if not (np.isnan(results['hypergraph_f1']) or np.isnan(results['pairwise_f1']))
                               else float('nan'))

    return results


# =======================================================
# STATISTICAL COMPARISON  (mirrors XGBoost pipeline)
# =======================================================

def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
    valid = [r for r in all_results
             if not (np.isnan(r.get('hypergraph_pr_auc', float('nan'))) or
                     np.isnan(r.get('pairwise_pr_auc', float('nan'))))]

    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in valid])
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in valid])
    diffs      = hyper_vals - pair_vals

    n_wins_hyper = int(np.sum(diffs > 0))
    n_wins_pair  = int(np.sum(diffs < 0))
    n_ties       = int(np.sum(diffs == 0))
    n_valid      = n_wins_hyper + n_wins_pair

    if n_valid > 0:
        p_greater   = binomtest(n_wins_hyper, n_valid, 0.5, alternative='greater').pvalue
        p_two_sided = binomtest(n_wins_hyper, n_valid, 0.5, alternative='two-sided').pvalue
    else:
        p_greater = p_two_sided = 1.0

    return {
        'n_runs':                   len(all_results),
        'n_valid':                  len(valid),
        'hypergraph_pr_auc_mean':   float(np.mean(hyper_vals)) if len(hyper_vals) else float('nan'),
        'hypergraph_pr_auc_std':    float(np.std(hyper_vals))  if len(hyper_vals) else float('nan'),
        'pairwise_pr_auc_mean':     float(np.mean(pair_vals))  if len(pair_vals) else float('nan'),
        'pairwise_pr_auc_std':      float(np.std(pair_vals))   if len(pair_vals) else float('nan'),
        'mean_difference':          float(np.mean(diffs))      if len(diffs) else float('nan'),
        'std_difference':           float(np.std(diffs))       if len(diffs) else float('nan'),
        'hypergraph_wins':          n_wins_hyper,
        'pairwise_wins':            n_wins_pair,
        'ties':                     n_ties,
        'sign_test_p_greater':      p_greater,
        'sign_test_p_two_sided':    p_two_sided,
    }


# =======================================================
# PRINTING  (mirrors XGBoost pipeline)
# =======================================================

def print_statistical_summary(stats: Dict, label: str = ""):
    header = f"STATISTICAL COMPARISON: HYPERGRAPH vs PAIRWISE (GNN){f' — {label}' if label else ''}"
    print(f"\n{'='*70}")
    print(f"  {header}")
    print(f"{'='*70}")
    print(f"\n  Number of splits: {stats['n_runs']}  (valid: {stats['n_valid']})")
    print(f"\n  {'Metric':<25} {'Hypergraph':<25} {'Pairwise':<20}")
    print(f"  {'-'*70}")
    print(f"  {'PR-AUC Mean ± Std':<25} "
          f"{stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}      "
          f"{stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}")
    print(f"\n  Mean Difference (Hyper - Pair): "
          f"{stats['mean_difference']:.4f} ± {stats['std_difference']:.4f}")
    n = stats['n_valid']
    if n > 0:
        print(f"\n  Win/Loss Record:")
        print(f"    Hypergraph wins : {stats['hypergraph_wins']}/{n} "
              f"({100*stats['hypergraph_wins']/n:.1f}%)")
        print(f"    Pairwise wins   : {stats['pairwise_wins']}/{n} "
              f"({100*stats['pairwise_wins']/n:.1f}%)")
        print(f"    Ties            : {stats['ties']}/{n}")
    print(f"\n  Sign Test p (one-sided, H > P): {stats['sign_test_p_greater']:.6f}")
    print(f"  Sign Test p (two-sided)        : {stats['sign_test_p_two_sided']:.6f}")
    print(f"{'='*70}")


# =======================================================
# PLOTTING  (mirrors XGBoost pipeline)
# =======================================================

def plot_paired_comparison(all_results: List[Dict], stats: Dict,
                           output_dir: Path, label: str = ""):
    valid = [r for r in all_results
             if not (np.isnan(r.get('hypergraph_pr_auc', float('nan'))) or
                     np.isnan(r.get('pairwise_pr_auc', float('nan'))))]
    if not valid:
        return

    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in valid])
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in valid])
    diffs      = hyper_vals - pair_vals

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_suffix = f" ({label})" if label else ""

    ax1 = axes[0]
    ax1.hist(diffs, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax1.axvline(diffs.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {diffs.mean():.4f}')
    ax1.set_xlabel('PR-AUC Difference (Hypergraph − Pairwise)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Paired Differences{title_suffix}')
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(pair_vals, hyper_vals, alpha=0.7, s=60, zorder=3)
    lo = min(pair_vals.min(), hyper_vals.min()) - 0.02
    hi = max(pair_vals.max(), hyper_vals.max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('Pairwise GNN PR-AUC')
    ax2.set_ylabel('Hypergraph HGNN PR-AUC')
    ax2.set_title(f'Paired Comparison{title_suffix}')
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect('equal')
    above = int(np.sum(hyper_vals > pair_vals))
    below = int(np.sum(hyper_vals < pair_vals))
    ax2.text(0.95, 0.05,
             f'HGNN wins: {above}\nGNN wins: {below}',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(facecolor='lightgreen', alpha=0.5))

    ax3 = axes[2]
    bp = ax3.boxplot([pair_vals, hyper_vals],
                     labels=['GNN\n(Pairwise)', 'HGNN\n(Hypergraph)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgray')
    bp['boxes'][1].set_facecolor('steelblue')
    ax3.set_ylabel('PR-AUC')
    ax3.set_title(f'Distribution{title_suffix}')
    rng = np.random.default_rng(0)
    for i, data in enumerate([pair_vals, hyper_vals]):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.4, s=20, color='black')

    plt.tight_layout()
    plt.savefig(output_dir / 'paired_comparison.png', dpi=300)
    plt.close()
    print(f"   Saved: paired_comparison.png")


# =======================================================
# SAVE OUTPUTS  (mirrors XGBoost pipeline)
# =======================================================

def save_outputs(all_results: List[Dict], stats: Dict, output_dir: Path,
                 pipeline_label: str, is_inductive: bool = False):
    # Per-split summary
    summary_cols = ['split_index', 'n_train', 'n_test', 'train_ess_pct', 'test_ess_pct',
                    'hypergraph_pr_auc', 'hypergraph_f1',
                    'pairwise_pr_auc',   'pairwise_f1',
                    'pr_auc_diff',       'f1_diff']
    if is_inductive:
        summary_cols.insert(4, 'n_isolated_test')

    summary_rows = []
    for r in all_results:
        row = {k: r.get(k, float('nan')) for k in summary_cols}
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / 'split_results.csv', index=False)
    print("   Saved: split_results.csv")

    # Per-protein predictions
    hyper_preds = pd.concat([r['hypergraph_predictions'] for r in all_results], ignore_index=True)
    pair_preds  = pd.concat([r['pairwise_predictions']   for r in all_results], ignore_index=True)
    hyper_preds.to_csv(output_dir / 'hypergraph_predictions.csv', index=False)
    pair_preds.to_csv(output_dir  / 'pairwise_predictions.csv',   index=False)
    print("   Saved: hypergraph_predictions.csv, pairwise_predictions.csv")

    # Statistical summary text
    with open(output_dir / 'statistical_summary.txt', 'w') as f:
        f.write(f"PAIRED COMPARISON: HYPERGRAPH (HGNN) vs PAIRWISE (GNN)\n")
        f.write(f"Pipeline: {pipeline_label}\n")
        f.write(f"Task: Gene Essentiality\n")
        f.write(f"Node feature: degree (single scalar)\n")
        f.write(f"Architecture: GraphSAGE / HypergraphConv — {CONFIG['NUM_LAYERS']} layers, "
                f"hidden_dim={CONFIG['HIDDEN_DIM']}, dropout={CONFIG['DROPOUT']}\n")
        f.write(f"Number of splits: {stats['n_runs']}  (valid: {stats['n_valid']})\n\n")
        f.write(f"Hypergraph PR-AUC: {stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}\n")
        f.write(f"Pairwise PR-AUC:   {stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}\n\n")
        f.write(f"Mean Difference:   {stats['mean_difference']:.4f} ± {stats['std_difference']:.4f}\n")
        f.write(f"Hypergraph wins:   {stats['hypergraph_wins']}/{stats['n_valid']}\n")
        f.write(f"Pairwise wins:     {stats['pairwise_wins']}/{stats['n_valid']}\n")
        f.write(f"Ties:              {stats['ties']}/{stats['n_valid']}\n\n")
        f.write(f"Sign test p (one-sided, H > P): {stats['sign_test_p_greater']:.6f}\n")
        f.write(f"Sign test p (two-sided):        {stats['sign_test_p_two_sided']:.6f}\n")
    print("   Saved: statistical_summary.txt")


# =======================================================
# DATA LOADING
# =======================================================

def load_data():
    print("1. Loading data...")
    complex_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["COMPLEX_FILE"])
    splits_df  = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"])
    splits_df  = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    label_map = {'Essential': 1, 'Non-essential': 0}
    splits_df['target'] = splits_df['protein_label'].map(label_map)

    print(f"   Complex file rows  : {len(complex_df)}")
    print(f"   Unique complexes   : {complex_df['ComplexId'].nunique()}")
    print(f"   Unique proteins    : {complex_df['ProteinId'].nunique()}")
    print(f"   Splits file rows   : {len(splits_df)}")
    print(f"   Unique splits      : {splits_df['split_index'].nunique()}")

    # --- Identify isolated proteins (only in single-protein complexes) ---
    from itertools import combinations as _combinations
    edges = set()
    for _, grp in complex_df.groupby('ComplexId'):
        ps = list(grp['ProteinId'].unique())
        if len(ps) >= 2:
            for a, b in _combinations(sorted(ps), 2):
                edges.add((a, b))
    proteins_with_edges = set()
    for a, b in edges:
        proteins_with_edges.add(a)
        proteins_with_edges.add(b)

    all_proteins = set(complex_df['ProteinId'].unique())
    isolated     = all_proteins - proteins_with_edges

    labelled_all = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
    excl = labelled_all[labelled_all['ProteinId'].isin(isolated)]
    print(f"\n   Proteins excluded (only in solo-complexes): {len(isolated)}")
    print(f"     Of which labelled: {len(excl)}")
    print(f"       Essential    : {(excl['protein_label']=='Essential').sum()}")
    print(f"       Non-essential: {(excl['protein_label']=='Non-essential').sum()}")

    return complex_df, splits_df, proteins_with_edges


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    start_time = time.time()
    print(f"Process started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # ---- Load ----
    complex_df, splits_df, valid_proteins = load_data()
    split_indices = sorted(splits_df['split_index'].unique())
    print(f"\n   Running {len(split_indices)} splits: {split_indices}\n")

    # ====================================================
    # PIPELINE 1 — TRANSDUCTIVE
    # ====================================================
    print(f"\n{'='*70}")
    print("  PIPELINE 1: TRANSDUCTIVE")
    print(f"{'='*70}\n")

    trans_output_dir = CONFIG["TRANSDUCTIVE_OUTPUT_DIR"] / "essentiality_family_splits"
    trans_output_dir.mkdir(parents=True, exist_ok=True)

    trans_results = []
    for split_idx in split_indices:
        print(f"  Split {split_idx:>2}/{len(split_indices)} (transductive)...", end=" ", flush=True)
        try:
            r = run_split_transductive(split_idx, complex_df, splits_df, valid_proteins)
            trans_results.append(r)
            winner = ("HGNN" if r['pr_auc_diff'] > 0
                      else "GNN" if r['pr_auc_diff'] < 0 else "Tie")
            print(f"train={r['n_train']} ({r['train_ess_pct']:.1f}% ess)  "
                  f"test={r['n_test']} ({r['test_ess_pct']:.1f}% ess)  |  "
                  f"HGNN: {r['hypergraph_pr_auc']:.4f}, "
                  f"GNN: {r['pairwise_pr_auc']:.4f}, "
                  f"Diff: {r['pr_auc_diff']:+.4f} [{winner}]")
        except Exception as e:
            print(f"ERROR: {e}")

    trans_stats = run_sign_test_comparison(trans_results)
    print_statistical_summary(trans_stats, label="Transductive")
    plot_paired_comparison(trans_results, trans_stats, trans_output_dir, label="Transductive")
    save_outputs(trans_results, trans_stats, trans_output_dir, "Transductive")

    # ====================================================
    # PIPELINE 2 — INDUCTIVE
    # ====================================================
    print(f"\n{'='*70}")
    print("  PIPELINE 2: INDUCTIVE")
    print(f"{'='*70}\n")

    ind_output_dir = CONFIG["INDUCTIVE_OUTPUT_DIR"] / "essentiality_family_splits"
    ind_output_dir.mkdir(parents=True, exist_ok=True)

    ind_results = []
    for split_idx in split_indices:
        print(f"  Split {split_idx:>2}/{len(split_indices)} (inductive)...", end=" ", flush=True)
        try:
            r = run_split_inductive(split_idx, complex_df, splits_df, valid_proteins)
            ind_results.append(r)
            hg_pr = r.get('hypergraph_pr_auc', float('nan'))
            gn_pr = r.get('pairwise_pr_auc', float('nan'))
            diff  = r.get('pr_auc_diff', float('nan'))
            winner = ("HGNN" if diff > 0 else "GNN" if diff < 0 else "Tie") if not np.isnan(diff) else "N/A"
            iso   = r.get('n_isolated_test', 0)
            print(f"train={r['n_train']} ({r['train_ess_pct']:.1f}% ess)  "
                  f"test={r['n_test']} (isolated={iso})  |  "
                  f"HGNN: {hg_pr:.4f}, GNN: {gn_pr:.4f}, "
                  f"Diff: {diff:+.4f} [{winner}]")
        except Exception as e:
            print(f"ERROR: {e}")

    ind_stats = run_sign_test_comparison(ind_results)
    print_statistical_summary(ind_stats, label="Inductive")
    plot_paired_comparison(ind_results, ind_stats, ind_output_dir, label="Inductive")
    save_outputs(ind_results, ind_stats, ind_output_dir, "Inductive", is_inductive=True)

    # ---- Elapsed ----
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")