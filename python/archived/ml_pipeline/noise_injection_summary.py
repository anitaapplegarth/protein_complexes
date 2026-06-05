"""
noise_injection_summary.py
==========================
Produces ranked feature importance tables from the noise injection experiment,
showing where random features sit within each model's ranking.

Outputs
-------
  - noise_ranked_tables.txt        — plain-text ranked tables (for supervisor)
  - noise_ranked_hypergraph.csv    — full ranked hypergraph table with noise
  - noise_ranked_pairwise.csv      — full ranked pairwise table with noise
  - noise_ranked_random.csv        — pure random model noise floor
  - features_to_drop.txt           — recommended features to remove + rationale

Usage
-----
  python noise_injection_summary.py

  Expects these files in INPUT_DIR (output from cp_noise_injection.py):
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
INPUT_DIR  = Path("./randomforest/chembl_noise_injection/drug_target_family_splits")
OUTPUT_DIR = INPUT_DIR   # write summary files alongside the other outputs

MODEL_TYPE  = "RandomForest"
TASK        = "Drug Target Prediction"
N_SPLITS    = 15
N_SEEDS     = 3
N_NOISE     = 4

# Threshold rule: noise mean + 1 SD of the per-noise-feature means.
# A real feature must exceed this to be considered above the noise floor.
# Using the PURE RANDOM model's noise features as the reference, since that
# model has no real features that could influence the noise importance estimates.
THRESHOLD_RULE = "pure_random mean + mean of per-feature SDs"


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
# THRESHOLD COMPUTATION
# =======================================================

def compute_threshold(rand_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Derive noise threshold from the pure-random model.

    Returns (noise_mean, noise_sd, threshold) where:
      noise_mean = mean of the per-noise-feature means
      noise_sd   = mean of the per-noise-feature SDs
      threshold  = noise_mean + noise_sd
    """
    noise_rows = rand_df[rand_df['feature'].str.startswith('rand_')]
    noise_mean = float(noise_rows['mean'].mean())
    noise_sd   = float(noise_rows['std'].mean())
    threshold  = noise_mean + noise_sd
    return noise_mean, noise_sd, threshold


# =======================================================
# BUILD RANKED TABLES
# =======================================================

def build_ranked_table(imp_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
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
    combined['above_noise'] = combined['mean'] > threshold
    combined['type'] = combined['feature'].apply(
        lambda f: 'NOISE' if f.startswith('rand_') else 'real'
    )

    return combined[['rank', 'type', 'feature', 'mean', 'std', 'above_noise']]


# =======================================================
# PRINTING & SAVING
# =======================================================

def format_table(df: pd.DataFrame) -> str:
    lines = []
    header = f"  {'Rank':<5} {'Type':<7} {'Feature':<38} {'Mean':>9}  {'Std':>9}  {'Signal'}"
    lines.append(header)
    lines.append("  " + "-" * 80)
    for _, row in df.iterrows():
        signal = "✓ above noise" if row['above_noise'] else "– within noise"
        noise_marker = "  <-- NOISE" if row['type'] == 'NOISE' else ""
        lines.append(
            f"  {int(row['rank']):<5} {row['type']:<7} {row['feature']:<38} "
            f"{row['mean']:>9.5f}  {row['std']:>9.5f}  {signal}{noise_marker}"
        )
    return "\n".join(lines)


def write_summary(
    rand_df, pair_ranked, hyper_ranked,
    noise_mean, noise_sd, threshold,
    drop_pair, drop_hyper,
    output_path: Path,
):
    lines = []
    sep = "=" * 85

    lines += [
        sep,
        "  NOISE INJECTION — FEATURE IMPORTANCE RANKING SUMMARY",
        sep,
        f"  Task          : {TASK}",
        f"  Model         : {MODEL_TYPE}",
        f"  Splits × Seeds: {N_SPLITS} × {N_SEEDS} = {N_SPLITS * N_SEEDS} realisations per feature",
        f"  Noise features: {N_NOISE} (standard normal, i.i.d.)",
        f"  Threshold rule: {THRESHOLD_RULE}",
        "",
        f"  Noise floor (pure random model):",
        f"    Mean of per-feature means : {noise_mean:+.5f}",
        f"    Mean of per-feature SDs   : {noise_sd:.5f}",
        f"    Threshold (mean + 1 SD)   : {threshold:+.5f}",
        "",
        "  Interpretation: a real feature must have mean importance > threshold",
        "  to be considered as carrying signal above noise.",
        "",
    ]

    # Pure random model
    lines += [
        sep,
        "  1. PURE RANDOM MODEL (noise features only)",
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
        "",
    ]
    lines.append(format_table(pair_ranked))
    lines.append("")

    # Hypergraph
    lines += [
        sep,
        "  3. HYPERGRAPH + NOISE — Full ranked list",
        sep,
        "  Noise features (rand_*) are interleaved at their actual rank position.",
        "",
    ]
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

    noise_mean, noise_sd, threshold = compute_threshold(rand_df)

    pair_ranked  = build_ranked_table(pair_df,  threshold)
    hyper_ranked = build_ranked_table(hyper_df, threshold)

    drop_pair  = pair_ranked[
        ~pair_ranked['feature'].str.startswith('rand_') & ~pair_ranked['above_noise']
    ]
    drop_hyper = hyper_ranked[
        ~hyper_ranked['feature'].str.startswith('rand_') & ~hyper_ranked['above_noise']
    ]

    # Write plain-text summary
    write_summary(
        rand_df, pair_ranked, hyper_ranked,
        noise_mean, noise_sd, threshold,
        drop_pair, drop_hyper,
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