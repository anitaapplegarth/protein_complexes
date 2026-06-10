import pandas as pd
from pathlib import Path

# =======================================================
# CONFIG
# =======================================================
DATABASES = {
    "Complex Portal": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "CORUM":          Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/"),
}

HYPERGRAPH_FEATURES = [
    'base_Degree', 'base_LocalClustCoeff', 'base_TriangleCount',
    'base_UniquePartners', 'base_AvgNeighbourDegree',
    'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
    'stoich_RangeComplexSize', 'stoich_MedComplexSize',
    'stoich_MedianRatio', 'stoich_RangeRatio',
    'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
    'protein_MedComplexNodes', 'protein_RangeComplexNodes',
]

PAIRWISE_FEATURES = [
    'pair_Degree', 'pair_LocalClustCoeff',
    'pair_TriangleCount', 'pair_AvgNeighborDegree',
]

PROTEIN_FEATURES_FILE  = "hypergraph_features.csv"
PAIRWISE_FEATURES_FILE = "pairwise_features.csv"

# =======================================================
# COVERAGE
# =======================================================
results = []

for db_name, data_dir in DATABASES.items():
    hg_df   = pd.read_csv(data_dir / PROTEIN_FEATURES_FILE)
    pair_df = pd.read_csv(data_dir / PAIRWISE_FEATURES_FILE)

    for group_name, df, feats in [("Hypergraph", hg_df, HYPERGRAPH_FEATURES),
                                   ("Pairwise",  pair_df, PAIRWISE_FEATURES)]:
        n = len(df)
        for f in feats:
            if f in df.columns:
                non_nan = int(df[f].notna().sum())
                results.append({
                    'database':    db_name,
                    'group':       group_name,
                    'feature':     f,
                    'non_nan':     non_nan,
                    'total':       n,
                    'coverage_pct': round(100 * non_nan / n, 1),
                })

coverage_df = pd.DataFrame(results)

# ----- Side-by-side pivot -----
pivot = coverage_df.pivot_table(
    index=['group', 'feature'],
    columns='database',
    values=['non_nan', 'total', 'coverage_pct'],
    aggfunc='first',
)

# Flatten column names and reorder for readability
flat = pd.DataFrame({
    'group':    coverage_df[coverage_df['database'] == list(DATABASES)[0]]
                .set_index('feature')['group'],
    'feature':  coverage_df[coverage_df['database'] == list(DATABASES)[0]]['feature'].values,
})

for db_name in DATABASES:
    db_rows = coverage_df[coverage_df['database'] == db_name].set_index('feature')
    flat[f'{db_name} coverage'] = flat['feature'].map(
        lambda f, db=db_rows: f"{int(db.loc[f, 'non_nan'])}/{int(db.loc[f, 'total'])} ({db.loc[f, 'coverage_pct']}%)"
        if f in db.index else "N/A"
    )

# ----- Print -----
print(f"\n{'='*90}")
print("  FEATURE COVERAGE: Complex Portal vs CORUM")
print(f"{'='*90}")

for group in ['Hypergraph', 'Pairwise']:
    subset = flat[flat['group'] == group]
    print(f"\n  {group}")
    print(f"  {'Feature':<42s} {'Complex Portal':<22s} {'CORUM':<22s}")
    print(f"  {'-'*42} {'-'*22} {'-'*22}")
    for _, row in subset.iterrows():
        print(f"  {row['feature']:<42s} {row['Complex Portal coverage']:<22s} {row['CORUM coverage']:<22s}")

# ----- Save CSV -----
output_path = Path("./feature_coverage_comparison.csv")
coverage_df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")