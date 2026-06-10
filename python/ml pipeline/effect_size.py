import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================
# CONFIG
# =======================================================
INPUT_FILE = Path("/Users/anitaapplegarth/github/dphil/protein_complexes/python/ml pipeline/corum/randomforest/randomforest_corum_hpa_noise_injection/noise_injection_records.csv")
OUTPUT_DIR = Path("./effect_size")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature groups by model
MODELS = {
    "Pairwise": {
        "prefix": "pair__",
        "real": ["pair_Degree", "pair_LocalClustCoeff",
                 "pair_TriangleCount", "pair_AvgNeighborDegree"],
        "noise": ["rand_0", "rand_1", "rand_2", "rand_3"],
    },
    "Hypergraph": {
        "prefix": "hyper__",
        "real": ["base_Degree", "base_LocalClustCoeff", "base_TriangleCount",
                 "base_UniquePartners", "base_AvgNeighbourDegree",
                 "stoich_WeightedTriangles", "stoich_AvgNeighbourDegreeStoich",
                 "stoich_RangeComplexSize", "stoich_MedComplexSize",
                 "stoich_MedianRatio", "stoich_RangeRatio",
                 "protein_MedianUniqueRatio", "protein_RangeUniqueRatio",
                 "protein_MedComplexNodes", "protein_RangeComplexNodes"],
        "noise": ["rand_0", "rand_1", "rand_2", "rand_3"],
    },
}

# =======================================================
# Plotting style
# =======================================================
plt.rcParams.update({
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  11,
    'figure.titlesize': 16,
})

# =======================================================
# Load
# =======================================================
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} observations")

# =======================================================
# Cohen's d: real feature vs max-noise baseline
# =======================================================
# For each observation, the "max noise" is the highest importance
# across the 4 noise features. This creates the most conservative
# baseline — the real feature must beat the BEST noise feature,
# not just the average one.

def cohens_d(x, y):
    """Cohen's d for two samples (independent, pooled SD)."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return 0.0
    return (mx - my) / pooled_sd


results = []

for model_name, cfg in MODELS.items():
    prefix = cfg["prefix"]

    # Build max-noise baseline: per observation, max across 4 noise features
    noise_cols = [prefix + n for n in cfg["noise"]]
    max_noise = df[noise_cols].max(axis=1).values

    for feat in cfg["real"]:
        col = prefix + feat
        real_vals = df[col].values

        d = cohens_d(real_vals, max_noise)
        mean_real  = np.mean(real_vals)
        mean_noise = np.mean(max_noise)
        diff       = mean_real - mean_noise

        results.append({
            'model':          model_name,
            'feature':        feat,
            'mean_importance': round(mean_real, 6),
            'mean_max_noise':  round(mean_noise, 6),
            'mean_diff':       round(diff, 6),
            'cohens_d':        round(d, 3),
            'effect_label':    (
                'large'    if abs(d) >= 0.8 else
                'medium'   if abs(d) >= 0.5 else
                'small'    if abs(d) >= 0.2 else
                'negligible'
            ),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'effect_sizes.csv', index=False)

# ----- Print -----
print(f"\n{'='*95}")
print("  EFFECT SIZE: real features vs max-noise baseline (Cohen's d)")
print(f"{'='*95}")

for model_name in MODELS:
    subset = results_df[results_df['model'] == model_name]
    print(f"\n  {model_name}")
    print(f"  {'Feature':<38s} {'Mean imp.':<12s} {'Mean noise':<12s} {'Diff':<10s} {'Cohen d':<10s} {'Effect':<12s}")
    print(f"  {'-'*38} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
    for _, row in subset.iterrows():
        print(f"  {row['feature']:<38s} {row['mean_importance']:<12.6f} {row['mean_max_noise']:<12.6f} "
              f"{row['mean_diff']:<10.6f} {row['cohens_d']:<10.3f} {row['effect_label']:<12s}")

# =======================================================
# Plot: Cohen's d bar chart per model
# =======================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colours = {'large': '#2ca02c', 'medium': '#ff7f0e', 'small': '#d62728', 'negligible': '#999999'}

for ax, model_name in zip(axes, MODELS):
    subset = results_df[results_df['model'] == model_name].sort_values('cohens_d', ascending=True)
    bars = ax.barh(
        subset['feature'], subset['cohens_d'],
        color=[colours[e] for e in subset['effect_label']],
        edgecolor='white', linewidth=0.5,
    )
    ax.set_xlabel("Cohen's d (vs max noise)")
    ax.set_title(model_name)

    # Reference lines for conventional thresholds
    for val, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axvline(val, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(val + 0.02, ax.get_ylim()[1] * 0.98, label, fontsize=9,
                color='grey', va='top')

    ax.axvline(0, color='black', linewidth=0.5)

fig.suptitle("Effect size of real features vs max-noise baseline", fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'effect_sizes.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'effect_sizes.png'}")
print(f"Saved: {OUTPUT_DIR / 'effect_sizes.csv'}")
plt.close(fig)