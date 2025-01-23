import pandas as pd

# Load the raw PLINK file
genotype_raw = pd.read_csv("genotype_matrix.raw", sep=" ")

# Drop metadata columns
genotype_clean = genotype_raw.iloc[:, 6:]

# Save as CSV
genotype_clean.to_csv("genotype_matrix.csv", index=False)