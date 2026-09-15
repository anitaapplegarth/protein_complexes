"""
RBF-SVM hyperparameter choice under the two splitting regimes.
Reads the existing sweep CSVs; nothing is refitted.
"""

from pathlib import Path
import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    "GROUP_CSV":  Path("cp_chembl/sweep_results.csv"),
    "RANDOM_CSV": Path("cp_chembl/randomprotein/sweep_results_randomprotein.csv"),
    "MODEL":      "RBF-SVM",
}


def main():
    C = CONFIG
    for arm, path in [("group", C["GROUP_CSV"]), ("random", C["RANDOM_CSV"])]:
        df = pd.read_csv(path)
        sub = df[df["model"] == C["MODEL"]]
        print(f"\n{'='*70}\n{arm.upper()}  ({path.name})\n{'='*70}")
        print(f"features per tier: "
              f"{sub.groupby('tier')['n_features'].first().to_dict()}")
        print(sub.groupby(["tier", "best_params"]).size()
                 .unstack(level=0, fill_value=0).to_string())


main()