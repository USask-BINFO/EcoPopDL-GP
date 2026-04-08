import pandas as pd
from pandas_plink import read_plink

# Load ADMIXTURE output (Q-matrix)
# Replace 'admixture_output.Q' with your actual Q file
admixture_output = pd.read_csv('/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Paper_revisions/Dataset4_Brassica_NAM/imputed_data_final.6.Q', sep='\s+', header=None)

# Load corresponding sample names
(bim, fam, bed) = read_plink("/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean", verbose=False)
# sample_names = fam['iid'].str.split("_").str[-1].values  # explicitly cleaned IDs
sample_names = fam['iid'].str.replace("_", "", regex=False).values

# Identify clusters: assign each sample to the cluster with the highest proportion
assigned_clusters = admixture_output.idxmax(axis=1)

# Combine sample names with their assigned clusters
cluster_assignments = pd.DataFrame({
    'Sample': sample_names,
    'Assigned_Cluster': assigned_clusters
})

# Save to a CSV for later use in modeling
cluster_assignments.to_csv('sample_clusters.csv', index=False)

print(assigned_clusters.head())
