# EcoPopDL-DP Framework

## Overview
EcoPopDL-DP is a comprehensive framework for environmental-aware and population-informed genomic prediction using deep learning and ChromoMap. It is designed to address challenges in predicting complex traits influenced by genotype-by-environment and genotype-by-location interactions in resource-limited breeding populations. The framework integrates genomic, population structure, and environmental data to improve predictive accuracy for complex traits such as yield, flowering time, and seed weight.
![Pipeline](Figures/EcoPopGp_1_.png)

## Features
- **ChromoMap:** A visual-spatial representation of SNP-level genomic variation, chromosome structure, and positional relationships.
- **Deep Learning Integration:** Utilizes convolutional neural networks (CNNs) for feature extraction and trait prediction.
- **Linear Mixed Model (LMM):** Captures both fixed and random effects to refine predictions and improve interpretability.
- **Hybrid Framework:** Combines CNN-derived features with LMM for improved genomic prediction performance.
- **Data Augmentation and Transfer Learning:** Enhances model generalizability and robustness with advanced techniques.

## Workflow
1. **Preliminary Data Processing:**
    - **Input Data:** Genotypic data, phenotypic data, and environmental variables.
    - **Genotypic Data Processing:** Minor allele frequency (MAF) filtering and linkage disequilibrium (LD) pruning.
    - **Phenotypic Data Processing:** Normalization and outlier removal.
2. **Genetic Ancestry Analysis:**
    - Incorporates unsupervised and supervised admixture analysis to derive genetic ancestry profiles.
    - Integrates population clusters into predictive models.
3. **Prediction Model Design:**
    - **ChromoMap Generation:** Encodes SNPs into a color-coded image representing the genome.
    - **CNN Architecture:** Leverages EfficientNet-B0 for trait prediction.
    - **Feature Engineering:** Extracts and integrates genomic and metadata features.
    - **Linear Mixed Model:** Adds environmental covariates and population structure.
4. **Benchmarking:**
    - Compares against baseline models such as GBLUP, RRBLUP, Bayesian Ridge Regression, Lasso Regression, and SVM.

## Installation
To install and run the EcoPopDL-DP framework:

1. Clone the repository:
   ```bash
   git clone https://git.cs.usask.ca/qnm481/ecopopgp.git
   cd ecopopgp
   ```
2. Prepare input data:
   - Ensure genotypic, phenotypic, and environmental data are in the appropriate formats as outlined in the scripts.

## Citation
If you use EcoPopDL-DP in your research, please cite:
> T. Hewavithana et al., "EcoPopDL-DP: Environmental-Aware and Population-Informed Genomic Prediction using Deep Learning and ChromoMap," Bioinformatics, 2022.

## Authors
- Thulani Hewavithana
- Sophie Duchesne
- Bunyamin Tar’an
- Ian Stavness
- Steve Shirtliffe
- Kirstin Bett
- Isobel A. P. Parkin
- Lingling Jin

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Create a pull request.
