import pandas as pd
s = pd.read_csv('../../data/lookup_tables/protein_splits_all_strat.csv')
dt = pd.read_csv('../../data/lookup_tables/lu_drug_target_protein_chembl.csv')  # whatever your label file is

# merge and check positives per split in test
merged = s.merge(dt, left_on='UniProt_AC', right_on='ProteinId')
for idx in sorted(merged['split_index'].unique()):
    test = merged[(merged['split_index']==idx) & (merged['split']=='test')]
    n_pos = (test['target']==1).sum()
    print(f"Split {idx}: {len(test)} test proteins, {n_pos} positives ({100*n_pos/len(test):.1f}%)")