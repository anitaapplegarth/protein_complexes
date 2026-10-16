"""
Grid-boundary check: how often does the search pick a value at the edge of
its own range? Reads existing sweep CSVs; nothing is refitted.
"""

from pathlib import Path
from ast import literal_eval
import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    "CSVS": {
        ("ess", "group"):     Path("cp_ess/sweep_results.csv"),
        ("ess", "random"):    Path("cp_ess/randomprotein/sweep_results_randomprotein.csv"),
        ("hpa", "group"):     Path("cp_hpa/sweep_results.csv"),
        ("hpa", "random"):    Path("cp_hpa/randomprotein/sweep_results_randomprotein.csv"),
        ("chembl", "group"):  Path("cp_chembl/sweep_results.csv"),
        ("chembl", "random"): Path("cp_chembl/randomprotein/sweep_results_randomprotein.csv"),
    },
    # Only ordered numeric grids can have an "edge". Non-numeric values
    # ('scale', None, tuples) are ignored for the boundary test.
    "GRIDS": {
        "RBF-SVM":      {"C": [0.1, 1, 10, 100, 1000],
                         "gamma": [0.001, 0.01, 0.1, 1]},
        "RandomForest": {"n_estimators": [80, 100, 200],
                         "max_depth": [5, 10],
                         "min_samples_split": [2, 5, 10]},
        "XGBoost":      {"n_estimators": [80, 100, 200],
                         "learning_rate": [0.01, 0.05, 0.1],
                         "max_depth": [5, 10],
                         "subsample": [0.75, 0.8, 1.0]},
        "LightGBM":     {"n_estimators": [80, 100, 200],
                         "learning_rate": [0.01, 0.05, 0.1],
                         "max_depth": [5, 10],
                         "num_leaves": [30, 50, 100]},
    },
    "FLAG_ABOVE": 0.5,   # highlight cells where over half the splits sit at an edge
}


def main():
    C = CONFIG
    rows = []
    for (task, arm), path in C["CSVS"].items():
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        df = pd.read_csv(path)
        df = df[df["best_params"].notna() & (df["best_params"] != "{}")]
        for model, grid in C["GRIDS"].items():
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            parsed = sub["best_params"].apply(literal_eval)
            for tier in sub["tier"].unique():
                mask = (sub["tier"] == tier).to_numpy()
                chosen = parsed[mask]
                n = len(chosen)
                for param, values in grid.items():
                    key = f"clf__{param}"
                    vals = [d.get(key, d.get(param)) for d in chosen]
                    vals = [v for v in vals if isinstance(v, (int, float))]
                    if not vals:
                        continue
                    lo, hi = min(values), max(values)
                    rows.append({
                        "task": task, "arm": arm, "model": model, "tier": tier,
                        "param": param, "n": n,
                        "at_min": sum(v == lo for v in vals) / n,
                        "at_max": sum(v == hi for v in vals) / n,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No searchable parameters found.")
        return
    out["at_edge"] = out[["at_min", "at_max"]].max(axis=1)
    flagged = out[out["at_edge"] > C["FLAG_ABOVE"]].sort_values(
        "at_edge", ascending=False)
    print(f"\nCells with over {C['FLAG_ABOVE']:.0%} of splits at a grid edge:\n")
    print(flagged.to_string(index=False,
                            float_format=lambda v: f"{v:.2f}"))
    out.to_csv("grid_boundary_check.csv", index=False)
    print(f"\nFull results written to grid_boundary_check.csv "
          f"({len(out)} rows)")


main()