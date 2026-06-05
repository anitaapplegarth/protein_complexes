"""
Run from your ml_pipeline/ directory:
    python3 compute_analysis_tables.py

Produces three CSVs in ml_pipeline/:
    1. compact_wins_table.csv        — Hypergraph vs Pairwise wins + median % improvement
    2. compact_scope_gains.csv       — Scope expansion gains for both representations
    3. compact_stoich_ablation.csv   — Stoichiometry ablation (CP only)
"""

import pandas as pd
import numpy as np
import os
import glob

# ============================================================
# CONFIG — maps (task, database, scope) to directory patterns
# ============================================================
# Directory naming:
#   {model}/{db}_{task}_{scope}_features/{splits_folder}/split_results.csv
#
# For CP with stoich: cp_{task}_{scope}_stoich_features
# For CP without stoich (essentiality): cp_{scope}_stoich_features  (essentiality has no task prefix)
# For CORUM: corum_{task}_{scope}_features (no stoich)
# Essentiality splits folder: essentiality_family_splits
# Drug target splits folder: drug_target_family_splits

MODELS = ["lightgbm", "randomforest", "xgboost"]
MODEL_DISPLAY = {"lightgbm": "LightGBM", "randomforest": "RF", "xgboost": "XGBoost"}

TASKS = {
    "Essentiality": {
        "CP": {
            "one-hop": "cp_one_hop_stoich_features/essentiality_family_splits",
            "two-hop": "cp_two_hop_stoich_features/essentiality_family_splits",
            "all":     "cp_all_stoich_features/essentiality_family_splits",
        },
        "CORUM": {
            "one-hop": "corum_one_hop_features/essentiality_family_splits",
            "two-hop": "corum_two_hop_features/essentiality_family_splits",
            "all":     "corum_all_features/essentiality_family_splits",
        },
    },
    "HPA Drug Target": {
        "CP": {
            "one-hop": "cp_hpa_one_hop_stoich_features/drug_target_family_splits",
            "two-hop": "cp_hpa_two_hop_stoich_features/drug_target_family_splits",
            "all":     "cp_hpa_all_stoich_features/drug_target_family_splits",
        },
        "CORUM": {
            "one-hop": "corum_hpa_one_hop_features/drug_target_family_splits",
            "two-hop": "corum_hpa_two_hop_features/drug_target_family_splits",
            "all":     "corum_hpa_all_features/drug_target_family_splits",
        },
    },
    "ChEMBL Drug Target": {
        "CP": {
            "one-hop": "cp_chembl_one_hop_stoich_features/drug_target_family_splits",
            "two-hop": "cp_chembl_two_hop_stoich_features/drug_target_family_splits",
            "all":     "cp_chembl_all_stoich_features/drug_target_family_splits",
        },
        "CORUM": {
            "one-hop": "corum_chembl_one_hop_features/drug_target_family_splits",
            "two-hop": "corum_chembl_two_hop_features/drug_target_family_splits",
            "all":     "corum_chembl_all_features/drug_target_family_splits",
        },
    },
}

# ============================================================
# LOAD DATA
# ============================================================
def load_split_results(model, subpath):
    """Load split_results.csv for a given model and subpath."""
    path = os.path.join(model, subpath, "split_results.csv")
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None
    return pd.read_csv(path)

# Store all loaded data
data = {}  # (task, db, scope, model) -> DataFrame

print("Loading data...")
for task, databases in TASKS.items():
    for db, scopes in databases.items():
        for scope, subpath in scopes.items():
            for model in MODELS:
                df = load_split_results(model, subpath)
                if df is not None:
                    data[(task, db, scope, MODEL_DISPLAY[model])] = df
                    
print(f"Loaded {len(data)} experiment files.\n")

# ============================================================
# TABLE 1: Hypergraph vs Pairwise (wins + median %)
# ============================================================
print("=" * 70)
print("TABLE 1: Hypergraph vs Pairwise")
print("=" * 70)

rows_wins = []
for task in TASKS:
    for db in ["CORUM", "CP"]:
        for model in ["LightGBM", "RF", "XGBoost"]:
            for scope in ["one-hop", "two-hop", "all"]:
                key = (task, db, scope, model)
                if key not in data:
                    continue
                df = data[key]
                hyper = df["hypergraph_pr_auc"].values
                pair = df["pairwise_pr_auc"].values
                n = len(df)
                
                pct = (hyper - pair) / pair * 100
                wins = int(np.sum(hyper > pair))
                
                rows_wins.append({
                    "Task": task,
                    "Database": db,
                    "Model": model,
                    "Scope": scope,
                    "Cell": f"{wins}/{n} ({np.median(pct):+.1f}%)",
                })

# Pivot to wide format
wins_df = pd.DataFrame(rows_wins)
wins_pivot = wins_df.pivot_table(
    index=["Task", "Database", "Model"],
    columns="Scope",
    values="Cell",
    aggfunc="first"
)[["one-hop", "two-hop", "all"]]  # column order
wins_pivot = wins_pivot.reset_index()
print(wins_pivot.to_string(index=False))
wins_pivot.to_csv("compact_wins_table.csv", index=False)
print("\n  -> Saved: compact_wins_table.csv")

# ============================================================
# TABLE 2: Scope gains (both representations)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 2: Scope Expansion Gains")
print("=" * 70)

rows_scope = []
for task in TASKS:
    for db in ["CORUM", "CP"]:
        for model in ["LightGBM", "RF", "XGBoost"]:
            for rep_name, col in [("Hypergraph", "hypergraph_pr_auc"), ("Pairwise", "pairwise_pr_auc")]:
                splits = {}
                for scope in ["one-hop", "two-hop", "all"]:
                    key = (task, db, scope, model)
                    if key in data:
                        splits[scope] = data[key][col].values
                
                if len(splits) != 3:
                    continue
                
                n = len(splits["one-hop"])
                
                pct_1to2 = (splits["two-hop"] - splits["one-hop"]) / splits["one-hop"] * 100
                wins_1to2 = int(np.sum(splits["two-hop"] > splits["one-hop"]))
                
                pct_2toA = (splits["all"] - splits["two-hop"]) / splits["two-hop"] * 100
                wins_2toA = int(np.sum(splits["all"] > splits["two-hop"]))
                
                pct_1toA = (splits["all"] - splits["one-hop"]) / splits["one-hop"] * 100
                wins_1toA = int(np.sum(splits["all"] > splits["one-hop"]))
                
                rows_scope.append({
                    "Task": task,
                    "Database": db,
                    "Representation": rep_name,
                    "Model": model,
                    "One→Two": f"{wins_1to2}/{n} ({np.median(pct_1to2):+.1f}%)",
                    "Two→All": f"{wins_2toA}/{n} ({np.median(pct_2toA):+.1f}%)",
                    "One→All": f"{wins_1toA}/{n} ({np.median(pct_1toA):+.1f}%)",
                })

scope_df = pd.DataFrame(rows_scope)
print(scope_df.to_string(index=False))
scope_df.to_csv("compact_scope_gains.csv", index=False)
print("\n  -> Saved: compact_scope_gains.csv")

# ============================================================
# TABLE 3: Stoichiometry ablation (CP only)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 3: Stoichiometry Ablation (CP only)")
print("=" * 70)

rows_stoich = []
for task in TASKS:
    for model in ["LightGBM", "RF", "XGBoost"]:
        row = {"Task": task, "Model": model}
        for scope in ["one-hop", "two-hop", "all"]:
            key = (task, "CP", scope, model)
            if key not in data:
                continue
            df = data[key]
            if "hyper_nostoich_pr_auc" not in df.columns:
                continue
            
            hyper = df["hypergraph_pr_auc"].values
            nostoich = df["hyper_nostoich_pr_auc"].values
            pair = df["pairwise_pr_auc"].values
            n = len(df)
            
            sv_pct = (hyper - nostoich) / nostoich * 100
            sv_wins = int(np.sum(hyper > nostoich))
            
            nv_pct = (nostoich - pair) / pair * 100
            nv_wins = int(np.sum(nostoich > pair))
            
            row[f"{scope} Stoich>NoStoich"] = f"{sv_wins}/{n} ({np.median(sv_pct):+.1f}%)"
            row[f"{scope} NoStoich>Pair"] = f"{nv_wins}/{n} ({np.median(nv_pct):+.1f}%)"
        
        rows_stoich.append(row)

stoich_df = pd.DataFrame(rows_stoich)
print(stoich_df.to_string(index=False))
stoich_df.to_csv("compact_stoich_ablation.csv", index=False)
print("\n  -> Saved: compact_stoich_ablation.csv")

print("\nDone!")
