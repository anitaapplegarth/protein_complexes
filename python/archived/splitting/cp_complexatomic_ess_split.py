"""
Complex-ATOMIC repeated random split generator for the essentiality task.

WHY THIS EXISTS
---------------
The existing splits (ess_protein_merged_splits.csv) are atomic at the *structural
group* level, but ~70% of complexes span >1 group, so ~89% of essential test
proteins share a complex with a training protein. This script makes the atomic
unit the COMPLEX-CONNECTED-COMPONENT: proteins are merged whenever they co-occur
in any complex, and each whole component is assigned to train or test. That
guarantees no complex — and no shared-protein chain — spans the split, so the
leakage measured earlier drops to ~0. It is the companion "complex-novel"
robustness split to sit alongside the structural (Foldseek) split.

WHAT IT DOES / DOESN'T CHANGE
-----------------------------
- Labels are INHERITED verbatim from the existing splits file (protein_label,
  label_mask), so the ONLY thing that differs from your current runs is the
  train/test assignment. No label can drift.
- Output schema is identical to ess_protein_merged_splits.csv, so it drops into
  cp_ess_second_testA_fifty.py via the splits mechanism with no code changes.
- It NEVER overwrites: if the target file exists, a timestamp is appended.

HOW TO RUN
----------
Just press Run in VS Code. All configuration is in CONFIG below; there are no
command-line arguments. Then point the modelling pipeline at the output file
(see the note printed at the end).
"""

import os
import time
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION  — the only things you might edit
# ============================================================================
CONFIG = {
    # Existing splits file: source of the protein population AND the labels.
    "existing_splits":   "/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/ess_protein_merged_splits.csv",
    # Complex membership (ComplexId, ProteinId, Stoichiometry).
    "complex_membership":"/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/stoich_protein.csv",
    # Output directory + base filename (timestamp appended if it already exists).
    "output_dir":        "/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp",
    "output_basename":   "ess_protein_complexatomic_splits.csv",

    # Column names in the existing splits file.
    "col_protein":       "UniProt_AC",
    "col_label":         "protein_label",
    "col_mask":          "label_mask",

    # Column names in the complex-membership file.
    "cm_complex":        "ComplexId",
    "cm_protein":        "ProteinId",

    # Split settings (mirrors the structural pipeline).
    "n_splits":          50,
    "test_frac":         0.20,
    "stratify_by_bucket": True,   # draw ~20% of components per size bucket
    "size_tol_pp":       5.0,     # accept if |test protein frac - 20%| <= this
    "balance_tol_pp":    4.5,     # accept if |test ess% - global ess%| <= this
    "max_attempts":      10,      # redraws per split before accepting best
    "random_state":      42,

    # Diagnostics only — never affects the split. Prints per-split test coverage
    # by complex-size bucket (the finding-3 check) and by component-size bucket
    # (the stratification check), and saves it next to the output file.
    "report_size_profile": True,
}


# ============================================================================
# UNION-FIND (complex-connected components)
# ============================================================================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def size_bucket(n):
    if n == 1:      return "singleton"
    elif n <= 5:    return "small"
    elif n <= 20:   return "medium"
    else:           return "large"


# ============================================================================
# CORE
# ============================================================================
def build_components(cm, population, cm_complex, cm_protein):
    """Union every pair of proteins that co-occur in a complex; return
    protein -> component_id (restricted to `population`)."""
    uf = UnionFind()
    # Ensure every population protein exists as its own node (singletons included)
    for p in population:
        uf.find(p)
    for _, members in cm.groupby(cm_complex)[cm_protein]:
        members = [m for m in members.unique() if m in population]
        if len(members) >= 2:
            first = members[0]
            for other in members[1:]:
                uf.union(first, other)
    comp_of = {p: uf.find(p) for p in population}
    # Relabel roots to compact grp_XXXX ids, largest component first
    from collections import Counter
    sizes = Counter(comp_of.values())
    ordered = sorted(sizes, key=lambda r: (-sizes[r], str(r)))
    root_to_gid = {r: f"grp_{i:04d}" for i, r in enumerate(ordered)}
    comp_of = {p: root_to_gid[r] for p, r in comp_of.items()}
    comp_size = {root_to_gid[r]: sizes[r] for r in sizes}
    return comp_of, comp_size


def make_one_split(comp_ids, comp_size, comp_ess, comp_n_labelled,
                   global_ess, cfg, rng):
    """Assign whole components to train/test. Returns dict comp_id -> 'train'|'test'
    plus realised (size_frac, ess_frac, attempts)."""
    total_prot = sum(comp_size.values())
    total_lab  = sum(comp_n_labelled.values())

    def draw():
        assign = {}
        if cfg["stratify_by_bucket"]:
            buckets = {}
            for c in comp_ids:
                buckets.setdefault(size_bucket(comp_size[c]), []).append(c)
            for b, comps in buckets.items():
                comps = list(comps)
                rng.shuffle(comps)
                n_test = round(len(comps) * cfg["test_frac"])
                for i, c in enumerate(comps):
                    assign[c] = "test" if i < n_test else "train"
        else:
            comps = list(comp_ids)
            rng.shuffle(comps)
            n_test = round(len(comps) * cfg["test_frac"])
            for i, c in enumerate(comps):
                assign[c] = "test" if i < n_test else "train"
        return assign

    def deviations(assign):
        test_prot = sum(comp_size[c] for c in comp_ids if assign[c] == "test")
        test_ess  = sum(comp_ess[c]  for c in comp_ids if assign[c] == "test")
        test_lab  = sum(comp_n_labelled[c] for c in comp_ids if assign[c] == "test")
        size_frac = test_prot / total_prot if total_prot else 0.0
        ess_frac  = (test_ess / test_lab) if test_lab else 0.0
        return size_frac, ess_frac

    best, best_dev, best_stats = None, np.inf, None
    for attempt in range(1, cfg["max_attempts"] + 1):
        assign = draw()
        size_frac, ess_frac = deviations(assign)
        size_dev = abs(size_frac - cfg["test_frac"]) * 100
        ess_dev  = abs(ess_frac - global_ess) * 100
        combined = size_dev + ess_dev
        if combined < best_dev:
            best, best_dev, best_stats = assign, combined, (size_frac, ess_frac, attempt)
        if size_dev <= cfg["size_tol_pp"] and ess_dev <= cfg["balance_tol_pp"]:
            return assign, (size_frac, ess_frac, attempt, True)
    # No attempt met tolerance — return best effort, flagged.
    sf, ef, at = best_stats
    return best, (sf, ef, at, False)


def resolve_output_path(cfg):
    path = os.path.join(cfg["output_dir"], cfg["output_basename"])
    if os.path.exists(path):
        stem, ext = os.path.splitext(cfg["output_basename"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(cfg["output_dir"], f"{stem}_{stamp}{ext}")
        print(f"  NOTE: target existed; writing to timestamped file instead:\n"
              f"        {path}")
    return path


def report_size_profiles(out, cm, population, cfg, splits_path):
    """Diagnostics: does the complex-atomic split still cover the size range in
    test? Prints coverage by complex-size bucket (finding-3 check) and by
    component-size bucket (stratification check). Pure reporting — no effect on
    the split or the saved file."""
    from collections import Counter

    # --- Per-protein complex-size bucket = bucket of the LARGEST complex it is in ---
    cm_pop = cm[cm[cfg["cm_protein"]].isin(population)]
    complex_nodes = cm_pop.groupby(cfg["cm_complex"])[cfg["cm_protein"]].nunique()
    p2max = {}
    for cid, grp in cm_pop.groupby(cfg["cm_complex"]):
        sz = int(complex_nodes[cid])
        for p in grp[cfg["cm_protein"]].unique():
            if sz > p2max.get(p, 0):
                p2max[p] = sz
    cplx_bucket = {p: (size_bucket(p2max[p]) if p in p2max else "none")
                   for p in population}

    # --- Component-size bucket = the atomic split unit (already in `out`) ---
    comp_bucket = (out.drop_duplicates("UniProt_AC")
                   .set_index("UniProt_AC")["group_bucket"].to_dict())

    # --- Per-protein labels (constant across splits) ---
    lab = out.drop_duplicates("UniProt_AC").set_index("UniProt_AC")
    is_lab = {p: bool(lab.loc[p, "label_mask"]) for p in population}
    is_ess = {p: (is_lab[p] and lab.loc[p, "protein_label"] == "Essential")
              for p in population}

    lines = []
    def emit(s):
        print(s)
        lines.append(s)

    def profile(bucket_of, title):
        order = [b for b in ["large", "medium", "small", "singleton", "none"]
                 if b in set(bucket_of.values())]
        tot = Counter(bucket_of.values())
        members_by_bucket = {b: [p for p in population if bucket_of[p] == b]
                             for b in order}

        emit("\n" + "=" * 88)
        emit(title)
        emit("=" * 88)
        emit("  (N = test proteins from that bucket; Ess% = essential rate among "
             "labelled test proteins)")
        emit(" Split | " + " | ".join(f"{b:>9} N  Ess%" for b in order))
        emit("-" * 88)

        frac_hist = {b: [] for b in order}
        ess_hist  = {b: [] for b in order}
        for s, sub in out.groupby("split_index"):
            test = set(sub[sub["split"] == "test"]["UniProt_AC"])
            cells = []
            for b in order:
                tin  = [p for p in members_by_bucket[b] if p in test]
                labn = [p for p in tin if is_lab[p]]
                essn = [p for p in labn if is_ess[p]]
                er   = (len(essn) / len(labn)) if labn else float("nan")
                frac_hist[b].append(len(tin) / tot[b] if tot[b] else 0.0)
                if labn:
                    ess_hist[b].append(er)
                    cells.append(f"{len(tin):>10}  {100*er:4.0f}%")
                else:
                    cells.append(f"{len(tin):>10}    -- ")
            emit(f" {s:>5} | " + " | ".join(cells))

        emit("-" * 88)
        emit(f"  {'bucket':<10} {'proteins':>9} {'mean test-frac':>16} "
             f"{'mean Ess%':>11}   (min-max test-frac)")
        for b in order:
            fr = np.array(frac_hist[b])
            es = np.array(ess_hist[b], dtype=float)
            ess_mean = float(np.nanmean(es)) if es.size else float("nan")
            emit(f"  {b:<10} {tot[b]:>9} {100*fr.mean():>15.1f}% "
                 f"{100*ess_mean:>10.1f}%   "
                 f"({100*fr.min():.1f}-{100*fr.max():.1f}%)")

        if "large" in order:
            fr = np.array(frac_hist["large"])
            nzero = int((fr == 0).sum())
            emit(f"\n  LARGE-bucket coverage: on average {100*fr.mean():.1f}% of "
                 f"large-bucket proteins land in test per split "
                 f"(min {100*fr.min():.1f}%, max {100*fr.max():.1f}%). "
                 f"Splits with ZERO large-bucket test proteins: {nzero}/{len(fr)}.")
            if nzero > len(fr) // 2:
                emit("  ** WARNING: the large bucket is absent from test in most "
                     "splits — the finding-3 size signal is under-tested here. **")

    profile(cplx_bucket,
            "TEST COVERAGE BY COMPLEX-SIZE BUCKET  "
            "(largest complex a protein belongs to — the finding-3 check)")
    profile(comp_bucket,
            "TEST COVERAGE BY COMPONENT-SIZE BUCKET  "
            "(the atomic split unit — the stratification check)")

    # --- Save alongside the splits file, never overwriting ---
    base, _ext = os.path.splitext(splits_path)
    ppath = base + "_sizeprofile.txt"
    if os.path.exists(ppath):
        ppath = base + f"_sizeprofile_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(ppath, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved size profile: {ppath}")


def generate(cfg):
    print("=" * 70)
    print("Complex-ATOMIC repeated random splitting (essentiality)")
    print("=" * 70)

    # --- Load population + inherited labels from existing splits ---
    ex = pd.read_csv(cfg["existing_splits"])
    for c in (cfg["col_protein"], cfg["col_label"], cfg["col_mask"]):
        if c not in ex.columns:
            raise KeyError(f"'{c}' not in {cfg['existing_splits']} "
                           f"(has: {list(ex.columns)})")
    labels = (ex[[cfg["col_protein"], cfg["col_label"], cfg["col_mask"]]]
              .drop_duplicates(cfg["col_protein"])
              .rename(columns={cfg["col_protein"]: "UniProt_AC",
                               cfg["col_label"]:   "protein_label",
                               cfg["col_mask"]:    "label_mask"}))
    population = set(labels["UniProt_AC"])
    label_of  = dict(zip(labels["UniProt_AC"], labels["protein_label"]))
    mask_of   = dict(zip(labels["UniProt_AC"], labels["label_mask"]))
    print(f"  Population (from existing splits): {len(population)} proteins")

    # --- Build complex-connected components ---
    cm = pd.read_csv(cfg["complex_membership"])
    comp_of, comp_size = build_components(
        cm, population, cfg["cm_complex"], cfg["cm_protein"])
    n_comp = len(set(comp_of.values()))
    print(f"  Complexes: {cm[cfg['cm_complex']].nunique()} -> "
          f"{n_comp} atomic components (union-find over co-membership)")
    biggest = max(comp_size.values())
    print(f"  Largest component: {biggest} proteins "
          f"({100*biggest/len(population):.1f}% of population)")

    # --- Per-component essential counts (labelled only) ---
    is_ess = {p: (mask_of[p] and label_of[p] == "Essential") for p in population}
    is_lab = {p: bool(mask_of[p]) for p in population}
    comp_ess, comp_lab = {}, {}
    for p in population:
        g = comp_of[p]
        comp_ess[g] = comp_ess.get(g, 0) + (1 if is_ess[p] else 0)
        comp_lab[g] = comp_lab.get(g, 0) + (1 if is_lab[p] else 0)
    n_ess_glob = sum(is_ess.values())
    n_lab_glob = sum(is_lab.values())
    global_ess = n_ess_glob / n_lab_glob if n_lab_glob else 0.0
    print(f"  Global essential rate (labelled): {100*global_ess:.1f}% "
          f"({n_ess_glob}/{n_lab_glob})")

    comp_ids = sorted(set(comp_of.values()))

    # --- Membership for the leak check: protein -> set(ComplexId) ---
    cm_pop = cm[cm[cfg["cm_protein"]].isin(population)]
    prot_to_complexes = (cm_pop.groupby(cfg["cm_protein"])[cfg["cm_complex"]]
                         .apply(set).to_dict())

    # --- Generate splits ---
    print(f"\n  Generating {cfg['n_splits']} splits "
          f"(stratified={cfg['stratify_by_bucket']})...")
    rows = []
    n_redraw = 0
    n_flagged = 0
    for s in range(1, cfg["n_splits"] + 1):
        rng = np.random.default_rng(cfg["random_state"] + s)
        assign, (sf, ef, att, ok) = make_one_split(
            comp_ids, comp_size, comp_ess, comp_lab, global_ess, cfg, rng)
        if att > 1:
            n_redraw += 1
        if not ok:
            n_flagged += 1

        # --- Leak verification: no complex may span train and test ---
        train_prot = {p for p in population if assign[comp_of[p]] == "train"}
        train_cplx = set()
        for p in train_prot:
            train_cplx |= prot_to_complexes.get(p, set())
        contam = 0
        n_test_complexbelonging = 0
        for p in population:
            if assign[comp_of[p]] == "test":
                pc = prot_to_complexes.get(p, set())
                if pc:
                    n_test_complexbelonging += 1
                    if pc & train_cplx:
                        contam += 1
        contam_rate = (contam / n_test_complexbelonging) if n_test_complexbelonging else 0.0

        flag = "" if ok else " [tolerance not met — best effort]"
        print(f"    Split {s:>3} | test size {100*sf:4.1f}% | "
              f"test ess {100*ef:4.1f}% | attempt {att} | "
              f"complex-contam {100*contam_rate:4.1f}%{flag}")
        if contam_rate > 1e-9:
            raise RuntimeError(
                f"LEAK: split {s} has {contam} contaminated test proteins — "
                f"components are not atomic. Aborting (this should never happen).")

        for p in population:
            g = assign[comp_of[p]]
            rows.append({
                "split_index":  s,
                "seed_accepted": cfg["random_state"] + s,
                "UniProt_AC":   p,
                "group_id":     comp_of[p],
                "group_size":   comp_size[comp_of[p]],
                "group_bucket": size_bucket(comp_size[comp_of[p]]),
                "group_status": "component",
                "split":        g,
                "protein_label": label_of[p],
                "label_mask":   mask_of[p],
            })

    out = pd.DataFrame(rows, columns=[
        "split_index", "seed_accepted", "UniProt_AC", "group_id", "group_size",
        "group_bucket", "group_status", "split", "protein_label", "label_mask"])

    # --- Write (never overwrite) ---
    print("\n  --- Saving ---")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    path = resolve_output_path(cfg)
    out.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(out):,} rows)")
    print(f"  Splits needing a redraw: {n_redraw}/{cfg['n_splits']}")
    print(f"  Splits not meeting tolerance (best effort): {n_flagged}/{cfg['n_splits']}")
    print(f"  Leakage: 0/{cfg['n_splits']} (guaranteed by atomic components)")

    if cfg.get("report_size_profile", True):
        report_size_profiles(out, cm, population, cfg, path)

    print("\n  --- To consume this in cp_ess_second_testA_fifty.py (VS Code) ---")
    print("  In that file's CONFIG, add ONE entry to SPLIT_SETS and select it:")
    print('      "SPLIT_SETS": {')
    print('          "strat":   "ess_protein_merged_splits.csv",')
    print('          "unstrat": "ess_protein_merged_splits_unstrat.csv",')
    print(f'          "cplxatomic": "{os.path.basename(path)}",   # <-- add')
    print("      },")
    print('      "SPLIT_SET": "cplxatomic",   # <-- set this')
    print("  Output then lands in  ./randomforest/cp_ess_second_testA_cplxatomic")
    print("  (parallel dir — your 'strat' results are NOT overwritten).")
    print("=" * 70)
    return out


if __name__ == "__main__":
    generate(CONFIG)