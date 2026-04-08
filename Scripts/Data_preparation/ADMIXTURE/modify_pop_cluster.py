import pandas as pd

# Load cluster assignments (ensure your CSV has columns: SampleID, Cluster)
clusters_df = pd.read_csv("/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Paper_revisions/ADMIXTURE/sample_clusters.csv")
print(clusters_df)

# Load your main phenotype dataset
phenotype_df = pd.read_csv("/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Phenotype/D4_OIL_DB_updated__.csv")
print(phenotype_df)
# Step 3: Replace the Pop values with correct clusters explicitly by matching 'Name'
cluster_dict = clusters_df.set_index('Name')['Assigned_Cluster'].to_dict()
phenotype_names = phenotype_df['Name'].unique()

# explicitly check matching samples
missing_samples = [sample for sample in phenotype_names if sample not in cluster_dict]
if missing_samples:
    print(f"Warning: Samples missing in clusters file: {missing_samples}")

# Map the new clusters explicitly into Pop column
phenotype_df['Pop'] = phenotype_df['Name'].map(cluster_dict)

# Handling missing matches explicitly
phenotype_df['Pop'] = phenotype_df['Pop'].fillna(-1).astype(int)

# Verify the mapping explicitly:
print(phenotype_df[['Name', 'Pop']].head(10))

# Finally save your updated phenotype file explicitly
phenotype_df.to_csv("/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Phenotype/D4_OIL_DB_updated.csv", index=False)