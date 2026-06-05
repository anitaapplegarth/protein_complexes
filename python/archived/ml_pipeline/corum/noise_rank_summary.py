"""
noise_injection_summary.py
==========================
Produces ranked feature importance tables from the noise injection experiment,
showing where random features sit within each model's ranking.

A feature is considered uninformative if it ranks at or below the
highest-ranked noise feature — no numeric threshold is needed.

Outputs
-------
  - noise_ranked_tables.txt        — plain-text ranked tables (for supervisor)
  - noise_ranked_hypergraph.csv    — full ranked hypergraph table with noise
  - noise_ranked_pairwise.csv      — full ranked pairwise table with noise
  - noise_ranked_random.csv        — pure random model noise floor

Usage
-----
  python noise_injection_summary.py

  Expects these files in INPUT_DIR (output from corum_noise_injection.py):
    hypergraph_noise_importance.csv
    pairwise_noise_importance.csv
    random_noise_importance.csv
    noise_floor_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =======================================================
# CONFIGURATION — update INPUT_DIR to your output folder
# =======================================================
INPUT_DIR  = Path("./randomforest/ess_noise_injection/drug_target_family_splits")
OUTPUT_DIR = INPUT_DIR   # write summary files alongside the other outputs

MODEL_TYPE  = "RandomForest"
TASK        = "Drug Target Prediction"
N_SPLITS    = 15
N_SEEDS     = 3
N_NOISE     = 4


# =======================================================
# LOAD
# =======================================================

def load_tables():
    hyper = pd.read_csv(INPUT_DIR / 'hypergraph_noise_importance.csv')
    pair  = pd.read_csv(INPUT_DIR / 'pairwise_noise_importance.csv')
    rand  = pd.read_csv(INPUT_DIR / 'random_noise_importance.csv')
    floor = pd.read_csv(INPUT_DIR / 'noise_floor_summary.csv')
    return hyper, pair, rand, floor


# =======================================================
# BUILD RANKED TABLES
# =======================================================

def build_ranked_table(imp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge real features and individual noise features into one ranked table.
    The pooled noise row ([noise × N]) is excluded — individual rows are used
    so the noise features appear at their actual rank positions.
    """
    # Real features
    real = imp_df[~imp_df['is_noise']].copy()

    # Individual noise features (exclude the pooled summary row)
    noise = imp_df[
        imp_df['feature'].str.startswith('rand_')
    ].copy()

    combined = pd.concat([real, noise], ignore_index=True)
    combined = combined.sort_values('mean', ascending=False).reset_index(drop=True)
    combined['rank'] = range(1, len(combined) + 1)
    combined['type'] = combined['feature'].apply(
        lambda f: 'NOISE' if f.startswith('rand_') else 'real'
    )

    return combined[['rank', 'type', 'feature', 'mean', 'std']]


# =======================================================
# PRINTING & SAVING
# =======================================================

def format_table(df: pd.DataFrame) -> str:
    lines = []
    header = f"  {'Rank':<5} {'Type':<7} {'Feature':<38} {'Mean':>9}  {'Std':>9}"
    lines.append(header)
    lines.append("  " + "-" * 72)
    for _, row in df.iterrows():
        lines.append(
            f"  {int(row['rank']):<5} {row['type']:<7} {row['feature']:<38} "
            f"{row['mean']:>9.5f}  {row['std']:>9.5f}"
        )
    return "\n".join(lines)


def write_summary(
    rand_df, pair_ranked, hyper_ranked,
    output_path: Path,
):
    lines = []
    sep = "=" * 85

    # Top noise rank for each model
    pair_top_noise  = pair_ranked[pair_ranked['type'] == 'NOISE']['rank'].min()
    hyper_top_noise = hyper_ranked[hyper_ranked['type'] == 'NOISE']['rank'].min()

    pair_below  = pair_ranked[
        (pair_ranked['type'] == 'real') & (pair_ranked['rank'] >= pair_top_noise)
    ]
    hyper_below = hyper_ranked[
        (hyper_ranked['type'] == 'real') & (hyper_ranked['rank'] >= hyper_top_noise)
    ]

    lines += [
        sep,
        "  NOISE INJECTION — FEATURE IMPORTANCE RANKING SUMMARY",
        sep,
        f"  Task          : {TASK}",
        f"  Model         : {MODEL_TYPE}",
        f"  Splits × Seeds: {N_SPLITS} × {N_SEEDS} = {N_SPLITS * N_SEEDS} realisations per feature",
        f"  Noise features: {N_NOISE} (standard normal, i.i.d.)",
        "",
        "  Criterion: features are ranked by mean permutation importance across",
        "  all splits and seeds.  A real feature is considered uninformative if",
        "  it ranks at or below the highest-ranked noise feature.",
        "",
    ]

    # Pure random model (sanity check)
    lines += [
        sep,
        "  1. PURE RANDOM MODEL (sanity check)",
        sep,
        "  All importances should be near zero — confirms noise is uninformative.",
        "",
    ]
    noise_only = rand_df[rand_df['feature'].str.startswith('rand_')][
        ['feature', 'mean', 'std']
    ].copy()
    noise_only = noise_only.sort_values('mean', ascending=False).reset_index(drop=True)
    noise_only['rank'] = range(1, len(noise_only) + 1)
    hdr = f"  {'Rank':<5} {'Feature':<20} {'Mean':>9}  {'Std':>9}"
    lines.append(hdr)
    lines.append("  " + "-" * 48)
    for _, row in noise_only.iterrows():
        lines.append(f"  {int(row['rank']):<5} {row['feature']:<20} "
                     f"{row['mean']:>9.5f}  {row['std']:>9.5f}")
    lines.append("")

    # Pairwise
    lines += [
        sep,
        "  2. PAIRWISE + NOISE — Full ranked list",
        sep,
        "  Noise features (rand_*) are interleaved at their actual rank position.",
        f"  Highest-ranked noise feature: rank {pair_top_noise}",
    ]
    if len(pair_below) == 0:
        lines.append("  All real features rank above noise.")
    else:
        names = ", ".join(pair_below['feature'].tolist())
        lines.append(f"  Real features at or below noise: {names}")
    lines.append("")
    lines.append(format_table(pair_ranked))
    lines.append("")

    # Hypergraph
    lines += [
        sep,
        "  3. HYPERGRAPH + NOISE — Full ranked list",
        sep,
        "  Noise features (rand_*) are interleaved at their actual rank position.",
        f"  Highest-ranked noise feature: rank {hyper_top_noise}",
    ]
    if len(hyper_below) == 0:
        lines.append("  All real features rank above noise.")
    else:
        names = ", ".join(hyper_below['feature'].tolist())
        lines.append(f"  Real features at or below noise: {names}")
    lines.append("")
    lines.append(format_table(hyper_ranked))
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text)
    print(text)
    return text


# =======================================================
# MAIN
# =======================================================

if __name__ == '__main__':

    print(f"Reading from: {INPUT_DIR}\n")
    hyper_df, pair_df, rand_df, floor_df = load_tables()

    pair_ranked  = build_ranked_table(pair_df)
    hyper_ranked = build_ranked_table(hyper_df)

    # Write plain-text summary
    write_summary(
        rand_df, pair_ranked, hyper_ranked,
        OUTPUT_DIR / 'noise_ranked_tables.txt',
    )

    # Save ranked CSVs
    pair_ranked.to_csv(OUTPUT_DIR  / 'noise_ranked_pairwise.csv',    index=False)
    hyper_ranked.to_csv(OUTPUT_DIR / 'noise_ranked_hypergraph.csv',   index=False)
    rand_df[rand_df['feature'].str.startswith('rand_')].to_csv(
        OUTPUT_DIR / 'noise_ranked_random.csv', index=False
    )

    print(f"\nSaved to: {OUTPUT_DIR}")
    print("  noise_ranked_tables.txt")
    print("  noise_ranked_pairwise.csv")
    print("  noise_ranked_hypergraph.csv")
    print("  noise_ranked_random.csv")