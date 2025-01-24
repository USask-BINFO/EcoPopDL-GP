#!/bin/bash

# STEP I

## Genotypic data visualization

python3 Genotypic_data_vis.py

## Phenotypic data visualization
python3 Phenotypic_data_vis.py

## Genotypic data pre-processing

# Filter put SNPS with maf rate < 0.05
./plink --allow-extra-chr --bfile chikpea --maf 0.01 --make-bed --out chikpea_maf

# LD Prunning based on Ancestry-Adjusted LD
./PCAone -B adj.residuals \
         --match-bim adj.mbim \
         --ld-r2 0.8 \
         --ld-bp 1000000 \
         -o adj

#Generate bed file from pruned data
./plink --bfile chikpea_maf --extract adj.ld.prune.in --make-bed --out pruned_data_admixture

#STEP II

# Run unsupervised admixture for K = 2 to 12

for K in range(2, 25):
    ./admixture --cv pruned_data_admixture.bed {K} | tee log{K}.out

grep -h "CV error" log*.out

# Run supervised admixture 

./admixture --supervised pruned_data_admixture.bed 17

# DAPC : R script

# ADMIXTURE plot: R script admixture_plot.R

# Generating genotype_matrix
./plink --bfile pruned_data_admixture --recode A --out genotype_matrix

# Converting genotype matrix to csv
python3 genotype_matrix_to_csv.py

# STEP III

## Generating ChromoMap
### i. For genotypic data as CSV input

## a). SNP density barcode per chromosome

python3 SNP_barcode_per_chromosome_csv.py
## b). ChromoMap per sample
python3 ChromoMap_csv.py


### ii. For genotypic data as PLINK input
## a). SNP density barcode per chromosome and ChromoMap
python3 SNP_barcode_per_chromosome_plink.py


## Generating Chromosome masks
### i. For genotypic data as CSV input
python3 Chromosome_masks_and_ChromoMap_csv.py

### ii. For genotypic data as PLINK input
python3 Chromosome_masks_and_ChromoMap_plink.py


## Generating CNN-based model: seprate python script


### Running LMM: R script LMM.R

### Baseline model: R script Baseline_models.R
