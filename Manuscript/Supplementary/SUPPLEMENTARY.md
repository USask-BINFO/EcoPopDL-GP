# Supplementary Figures and Tables for EcoPopDL-GP

This page hosts the supplementary material that will be linked from the manuscript.
Figure numbering and titles follow the manuscript order exactly for **Figures S2-S6** and **Tables S3-S13**.

## Supplementary Figures

### Figure S2. Temporal-window perturbation sensitivity across tasks.

<p align="center">
  <img src="Figures/main_geno_feature_importance.png" alt="Figure S2. Temporal-window perturbation sensitivity across tasks." width="1000">
</p>

*Caption.* Each panel shows the mean ΔR² after sample-wise permutation of one temporal window in the environmental sequence for a given dataset-trait task. Positive values indicate reduced predictive performance after perturbation. These profiles therefore quantify reliance on temporally localized environmental signal rather than identifying exact causal developmental windows. Panels are shown in a consistent task order to facilitate comparison across datasets and traits.

### Figure S3. Stage×feature perturbation sensitivity across tasks.

<p align="center">
  <img src="Figures/supp_stage_feature_heatmaps.png" alt="Figure S3. Stage×feature perturbation sensitivity across tasks." width="1000">
</p>

*Caption.* Each cell summarizes the mean ΔR² after sample-wise permutation of one environmental feature within one temporal stage. These panels therefore quantify post-hoc reliance on stage-specific environmental signal rather than causal effects of individual variables. Panels are shown in a consistent task order to facilitate comparison across datasets and traits. Missing panels indicate analyses that were still running when the figure was generated and can be updated later without changing the layout.

### Figure S4. Module and input perturbation sensitivity across tasks.

<p align="center">
  <img src="Figures/supp_module_input_sensitivity.png" alt="Figure S3. Module and input perturbation sensitivity across tasks." width="1000">
</p>

*Caption.* Each panel shows mean ΔR² after perturbing one input group or model component. Environmental and metadata inputs were perturbed by sample-wise permutation when appropriate, whereas BAE- and HABE-related genomic groups were evaluated by channel occlusion implemented by zeroing selected genomic channels. Panels are ordered consistently across tasks to support side-by-side comparison.

### Figure S5. Per-population held-out predictive performance across tasks.

<p align="center">
  <img src="Figures/supp_population_r2.png" alt="Figure S4. Per-population held-out predictive performance across tasks." width="1000">
</p>

*Caption.* Each panel shows mean held-out R² within inferred population groups for one dataset-trait task under within-genotype environment-holdout cross-validation. Error bars indicate variability across folds. This figure reports subgroup performance summaries rather than perturbation-based importance. Panels are ordered consistently across tasks to facilitate comparison of subgroup performance across datasets and traits. Missing panels indicate analyses that were still running when the figure was generated and can be updated later without changing the layout.

_Note:_ Population labels denote inferred population-cluster IDs and are ordered within each panel by mean held-out $R^2$, not by cluster number. In panels (b) and (c), at least one cluster had too few held-out observations within a fold ($n<2$), so fold-level $R^2$ was undefined. In plots generated without filtering undefined values, that cluster label may therefore appear on the y-axis without a visible bar.

### Figure S6. Homoeolog-related perturbation sensitivity in the polyploid D4 dataset.

<p align="center">
  <img src="Figures/supp_d4_homoeolog.png" alt="Figure S5. Homoeolog-related perturbation sensitivity in the polyploid D4 dataset." width="1000">
</p>

*Caption.* Panels show mean ΔR² after perturbing homoeolog-aware channels or temporarily disabling homoeolog-related modules for D4 Oil content and D4 days to flowering. These analyses were specific to the polyploid dataset and are shown separately from the shared perturbation panels.

## Supplementary Results Tables

### Table S3. Full ChromoMap channel inventory.

| Channel group | Description | Dataset-specific use |
|---|---|---|
| SNP genotype state | Normalized allele dosage and encoded dosage state used by BAE | All datasets |
| Positional channels | Chromosome identity, within-chromosome order, and padding mask | All datasets |
| Haplotype / LD context | Block ID, normalized block properties, LD-derived context summaries | All datasets |
| Gene annotation | Gene overlap / proximity features | All datasets |
| Promoter annotation | Promoter overlap indicators / proximity features | When promoter annotation available |
| TE annotation | TE overlap / proximity features | All datasets |
| Hotspot mask | Binary flag used by HABE to retain loci as individual tokens; set from SNP-level annotation (transposable element, gene body, promoter) and block-level summaries (high gene count, SNP density, or mean minor allele frequency) | Annotation-dependent |
| Population metadata | Population-cluster assignments inferred by ADMIXTURE | All datasets |
| Subgenome label | Subgenome identity for polyploid chromosomes | Polyploid only |
| Homoeolog presence | Indicator that a locus belongs to an annotated homoeolog group | Polyploid only |
| Homoeolog group size | Normalized homoeolog-group size | Polyploid only |
| Homoeolog group identity | Raw or hash-encoded group identifier | Polyploid only |
| Homoeolog anchor density | Local density of annotated homoeolog anchors | Polyploid only |

### Table S4. Complete predictive performance across all datasets under within-genotype environment-holdout cross-validation (env-CV).

Wall-clock time is reported in minutes. Higher R² and CCC indicate better performance; lower MAE and RMSE indicate better performance. Abbreviations: **EcoPop** = EcoPopDL-GP; **LMET** = LearnMET; **DGxE** = DeepG×E; **SW** = seed weight; **DTF** = days to flowering; **FT** = flowering time. Entries marked **—** indicate values not generated because of memory constraints.

#### D1 Yield

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 60 | **0.62** | **0.545** | **0.689** | **0.77** |
| GBLUP | 3 | 0.46 | 0.597 | 0.749 | 0.64 |
| BRR | 10 | 0.40 | 0.652 | 0.797 | 0.63 |
| RF | 20 | 0.47 | 0.595 | 0.742 | 0.64 |
| XGB | 20 | 0.47 | 0.594 | 0.746 | 0.65 |
| LMET | 30 | 0.44 | 0.604 | 0.747 | 0.61 |
| DGxE | 75 | 0.36 | 0.670 | 0.843 | 0.57 |

#### D1 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 45 | **0.59** | **0.470** | **0.637** | **0.74** |
| GBLUP | 1 | 0.53 | 0.537 | 0.705 | 0.68 |
| BRR | 7 | 0.39 | 0.637 | 0.805 | 0.68 |
| RF | 15 | 0.53 | 0.537 | 0.738 | 0.71 |
| XGB | 15 | 0.49 | 0.604 | 0.738 | 0.70 |
| LMET | 20 | 0.44 | 0.604 | 0.805 | 0.63 |
| DGxE | 50 | 0.54 | 0.637 | 0.906 | 0.68 |

#### D1 SW

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 50 | **0.59** | **0.517** | **0.676** | **0.73** |
| GBLUP | 3 | 0.49 | 0.573 | 0.757 | 0.66 |
| BRR | 9 | 0.48 | 0.577 | 0.761 | 0.67 |
| RF | 17 | 0.52 | 0.559 | 0.745 | 0.69 |
| XGB | 17 | 0.53 | 0.556 | 0.750 | 0.69 |
| LMET | 30 | 0.42 | 0.584 | 0.789 | 0.66 |
| DGxE | 60 | 0.48 | 0.520 | 0.745 | 0.67 |

#### D2 FT

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 62 | **0.50** | **0.424** | **0.767** | **0.65** |
| GBLUP | 3 | 0.37 | 0.537 | 0.915 | 0.52 |
| BRR | 12 | 0.36 | 0.529 | 0.903 | 0.57 |
| RF | 20 | 0.45 | 0.483 | 0.849 | 0.62 |
| XGB | 20 | 0.40 | 0.526 | 0.985 | 0.59 |
| LMET | 35 | 0.40 | 0.459 | 0.841 | 0.58 |
| DGxE | 65 | 0.47 | 0.514 | 0.794 | 0.63 |

#### D3 Yield

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3180 | **0.45** | **0.579** | **0.847** | **0.62** |
| GBLUP | 150 | 0.34 | 0.609 | 0.913 | 0.51 |
| BRR | 180 | 0.34 | 0.605 | 0.903 | 0.50 |
| RF | 600 | 0.34 | 0.603 | 0.905 | 0.51 |
| XGB | 825 | 0.34 | 0.603 | 0.905 | 0.52 |
| LMET | — | — | — | — | — |
| DGxE | 4320 | -0.06 | 0.910 | 1.186 | 0.10 |

#### D3 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3300 | **0.71** | **0.405** | **0.541** | **0.83** |
| GBLUP | 165 | 0.60 | 0.436 | 0.584 | 0.79 |
| BRR | 210 | 0.60 | 0.475 | 0.639 | 0.74 |
| RF | 720 | 0.65 | 0.434 | 0.568 | 0.79 |
| XGB | 910 | 0.66 | 0.439 | 0.584 | 0.79 |
| LMET | — | — | — | — | — |
| DGxE | 4566 | 0.30 | 0.667 | 0.826 | 0.53 |

#### D3 SW

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3720 | **0.62** | **0.411** | **0.616** | **0.75** |
| GBLUP | 210 | 0.43 | 0.548 | 0.760 | 0.63 |
| BRR | 255 | 0.42 | 0.538 | 0.766 | 0.57 |
| RF | 865 | 0.46 | 0.518 | 0.745 | 0.64 |
| XGB | 1080 | 0.47 | 0.505 | 0.734 | 0.64 |
| LMET | — | — | — | — | — |
| DGxE | 4900 | 0.33 | 0.590 | 0.880 | 0.59 |

#### D4 Oil content

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 45 | **0.56** | **0.509** | **0.638** | **0.71** |
| GBLUP | 7 | 0.48 | 0.542 | 0.703 | 0.64 |
| BRR | 5 | 0.46 | 0.561 | 0.719 | 0.63 |
| RF | 20 | 0.44 | 0.571 | 0.729 | 0.59 |
| XGB | 22 | 0.47 | 0.548 | 0.709 | 0.63 |
| LMET | 305 | 0.39 | 0.600 | 0.758 | 0.56 |
| DGxE | 110 | 0.33 | 0.645 | 0.829 | 0.55 |

#### D4 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 32 | **0.52** | **0.422** | **0.550** | **0.70** |
| GBLUP | 6 | 0.46 | 0.534 | 0.717 | 0.63 |
| BRR | 5 | 0.43 | 0.545 | 0.735 | 0.63 |
| RF | 18 | 0.46 | 0.543 | 0.714 | 0.63 |
| XGB | 19 | 0.45 | 0.520 | 0.723 | 0.65 |
| LMET | 280 | 0.42 | 0.573 | 0.760 | 0.59 |
| DGxE | 90 | 0.29 | 0.650 | 0.881 | 0.58 |


### Table S5. Complete predictive performance across all datasets under genotype-grouped cross-validation (geno-CV).

Wall-clock time is reported in minutes. Higher R² and CCC indicate better performance; lower MAE and RMSE indicate better performance. Abbreviations: **EcoPop** = EcoPopDL-GP; **LMET** = LearnMET; **DGxE** = DeepG×E; **SW** = seed weight; **DTF** = days to flowering; **FT** = flowering time. Entries marked **—** indicate values not generated because of memory constraints.

### D1 Yield

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 60 | **0.58** | **0.573** | **0.724** | **0.73** |
| GBLUP | 3 | 0.41 | 0.639 | 0.797 | 0.56 |
| BRR | 10 | 0.36 | 0.665 | 0.830 | 0.51 |
| RF | 20 | 0.43 | 0.628 | 0.784 | 0.58 |
| XGB | 20 | 0.42 | 0.633 | 0.791 | 0.57 |
| LMET | 30 | 0.39 | 0.649 | 0.811 | 0.54 |
| DGxE | 75 | 0.31 | 0.691 | 0.862 | 0.46 |

### D1 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 45 | **0.51** | **0.514** | **0.696** | **0.66** |
| GBLUP | 1 | 0.49 | 0.587 | 0.777 | 0.64 |
| BRR | 7 | 0.35 | 0.663 | 0.877 | 0.50 |
| RF | 15 | 0.49 | 0.587 | 0.777 | 0.64 |
| XGB | 15 | 0.45 | 0.610 | 0.806 | 0.60 |
| LMET | 20 | 0.40 | 0.637 | 0.842 | 0.55 |
| DGxE | 50 | 0.50 | 0.581 | 0.769 | 0.65 |

### D1 SW

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 50 | **0.52** | **0.559** | **0.731** | **0.66** |
| GBLUP | 3 | 0.47 | 0.574 | 0.771 | 0.61 |
| BRR | 9 | 0.46 | 0.579 | 0.778 | 0.60 |
| RF | 17 | 0.50 | 0.571 | 0.749 | 0.64 |
| XGB | 17 | 0.51 | 0.568 | 0.741 | 0.65 |
| LMET | 30 | 0.40 | 0.610 | 0.820 | 0.54 |
| DGxE | 60 | 0.45 | 0.584 | 0.785 | 0.59 |

### D2 FT

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 62 | **0.48** | **0.432** | **0.782** | **0.63** |
| GBLUP | 3 | 0.34 | 0.531 | 0.925 | 0.49 |
| BRR | 12 | 0.33 | 0.535 | 0.932 | 0.48 |
| RF | 20 | 0.42 | 0.498 | 0.867 | 0.57 |
| XGB | 20 | 0.37 | 0.519 | 0.904 | 0.52 |
| LMET | 35 | 0.36 | 0.523 | 0.911 | 0.51 |
| DGxE | 65 | 0.44 | 0.490 | 0.852 | 0.59 |


### D3 Yield

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3180 | **0.43** | **0.589** | **0.862** | **0.60** |
| GBLUP | 150 | 0.31 | 0.643 | 0.936 | 0.48 |
| BRR | 180 | 0.31 | 0.643 | 0.936 | 0.48 |
| RF | 600 | 0.32 | 0.638 | 0.929 | 0.49 |
| XGB | 825 | 0.33 | 0.633 | 0.922 | 0.50 |
| LMET | — | — | — | — | — |
| DGxE | 4320 | -0.07 | 0.800 | 1.165 | 0.10 |

### D3 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3300 | **0.68** | **0.425** | **0.568** | **0.80** |
| GBLUP | 165 | 0.58 | 0.484 | 0.636 | 0.70 |
| BRR | 210 | 0.57 | 0.489 | 0.644 | 0.69 |
| RF | 720 | 0.63 | 0.454 | 0.597 | 0.75 |
| XGB | 910 | 0.64 | 0.448 | 0.589 | 0.76 |
| LMET | — | — | — | — | — |
| DGxE | 4566 | 0.28 | 0.633 | 0.833 | 0.40 |

### D3 SW

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 3720 | **0.60** | **0.422** | **0.632** | **0.73** |
| GBLUP | 210 | 0.41 | 0.540 | 0.782 | 0.54 |
| BRR | 255 | 0.40 | 0.545 | 0.789 | 0.53 |
| RF | 865 | 0.44 | 0.526 | 0.762 | 0.57 |
| XGB | 1080 | 0.45 | 0.522 | 0.755 | 0.58 |
| LMET | — | — | — | — | — |
| DGxE | 4900 | 0.31 | 0.584 | 0.846 | 0.44 |

### D4 Oil content

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 45 | **0.56** | **0.509** | **0.638** | **0.71** |
| GBLUP | 7 | 0.46 | 0.562 | 0.719 | 0.61 |
| BRR | 5 | 0.44 | 0.573 | 0.732 | 0.59 |
| RF | 20 | 0.42 | 0.583 | 0.745 | 0.57 |
| XGB | 22 | 0.45 | 0.567 | 0.725 | 0.60 |
| LMET | 305 | 0.37 | 0.607 | 0.776 | 0.52 |
| DGxE | 110 | 0.31 | 0.636 | 0.813 | 0.46 |

### D4 DTF

| Model | Time | R² | MAE | RMSE | CCC |
|---|---:|---:|---:|---:|---:|
| **EcoPop** | 32 | **0.51** | **0.426** | **0.556** | **0.69** |
| GBLUP | 6 | 0.42 | 0.548 | 0.735 | 0.60 |
| BRR | 5 | 0.39 | 0.562 | 0.753 | 0.57 |
| RF | 18 | 0.43 | 0.544 | 0.728 | 0.61 |
| XGB | 19 | 0.41 | 0.553 | 0.741 | 0.59 |
| LMET | 280 | 0.38 | 0.567 | 0.760 | 0.56 |
| DGxE | 90 | 0.26 | 0.619 | 0.830 | 0.44 |

### Table S6. Qualitative summary of mixed post-hoc perturbation sensitivity analyses.

Entries summarize the dominant patterns observed after sample-wise permutation of environmental or metadata signals, channel occlusion of structured genomic inputs, and, for selected D4 components, temporary module disablement.

| Dataset | Trait | Temporal-window profile | Dominant stage/feature groups | Strongest module/input perturbations | Per-population pattern | Homoeolog effect |
|---|---|---|---|---|---|---|
| D1 | Yield | Broad profile with modest mid-to-late emphasis | Heat, Vapour pressure deficit, drought (later stage) | BAE strongest; other perturbations smaller | Highly heterogeneous, including one strongly negative group | -- |
| D1 | DTF | Distributed; no single dominant window | Daylength, growing degree days, photo-thermal temperature | Population identifiers, then BAE | Heterogeneous | -- |
| D1 | SW | Distributed across windows | Temperature and photo-thermal features | BAE strongest; HABE secondary | Heterogeneous | -- |
| D2 | FT | Distributed across windows | Daylength, growing degree days, photo-thermal temperature | BAE with additional environmental contribution | Heterogeneous | -- |
| D3 | Yield | Flatter profile | Weak stage localization; broad environmental effects | Population identifiers largest; environmental perturbation smaller | One lower-performing group, including a negative mean R² group | -- |
| D3 | DTF | Weak, diffuse profile with small mid-to-late peaks; no single dominant window | Mid-stage minimum temperature and growing degree days; smaller later-stage thermal effects | BAE strongest; environmental, population, and HABE perturbations small | All displayed groups positive; moderate heterogeneity across populations | -- |
| D3 | SW | Weak, diffuse profile with no single dominant window; modest mid-to-late peaks with broad uncertainty | Late-stage heat-related degree days strongest; smaller positive changes for late maximum temperature and drought/Vapour pressure deficit-related features; most other cells weak or near zero | BAE strongest by a wide margin; population and full-environment perturbations small; HABE and threshold perturbations near zero | All displayed groups positive; moderate heterogeneity, with one lower-performing group showing wide uncertainty | -- |
| D4 | Oil content | Later-window emphasis | Late heat, Vapour pressure deficit, drought | BAE and environment strongest; HABE secondary | All displayed groups positive | Small |
| D4 | DTF | Later-window emphasis | Daylength, growing degree days, photo-thermal temperature | Environment strongest; BAE and HABE smaller | All displayed groups positive | Small |

### Table S7. Hyperparameter search spaces for the tuned machine-learning baselines.

| Model | Hyperparameter | Search values |
|---|---|---|
| RF | number of trees | 200, 500, 1000 |
| RF | maximum depth | 5, 10, 20, none |
| RF | feature fraction | sqrt, 0.3, 1.0 |
| RF | minimum samples per leaf | 1, 3, 5 |
| XGB | number of trees | 200, 500, 1000 |
| XGB | maximum depth | 3, 5, 8 |
| XGB | learning rate | 0.01, 0.03, 0.1 |
| XGB | subsample | 0.7, 1.0 |
| XGB | column subsample | 0.7, 1.0 |
| Ridge (rrBLUP) | regularization strength (alpha) | 0.001, 0.01, 0.1, 1, 10, 100 |
| GBLUP | (none tuned) | genomic relationship kernel; standard settings |
| BRR | (self-determined) | priors estimated from the training data |
| LMET | (package defaults) | multi-environment machine-learning pipeline defaults |
| DGxE | (fixed) | same optimizer, early stopping, and epoch budget as EcoPopDL-GP |

_Search strategy:_ randomized search, 20 iterations, inner 3-fold cross-validation on the training split, scored by R². The outer split remains genotype-grouped, so no test observation contributes to hyperparameter selection.

### Table S8. Transfer to entirely unseen environments (environment-blocked cross-validation).

| Dataset | Trait | Held out | Ranking *r* | Absolute R² |
|---|---|---|---:|---:|
| D1 | Yield | location | 0.42 | −1.18 |
| D1 | Yield | year | 0.41 | −2.38 |
| D1 | Yield | location×year | 0.48 | −1.08 |
| D1 | DTF | location | 0.60 | −0.46 |
| D1 | DTF | year | 0.50 | −0.51 |
| D1 | DTF | location×year | 0.69 | −1.23 |
| D1 | SW | location | 0.72 | −1.50 |
| D1 | SW | year | 0.55 | 0.18 |
| D1 | SW | location×year | 0.72 | -1.10 |
| D2 | FT | location | 0.46 | −2.19 |
| D2 | FT | year | 0.41 | −0.21 |
| D2 | FT | location×year | 0.54 | −0.93 |
| D3 | Yield | location | {{...}} | {{...}} |
| D3 | Yield | year | 0.42 | 0.00 |
| D3 | Yield | location×year | {{...}} | {{...}} |
| D3 | DTF | location | {{...}} | {{...}} |
| D3 | DTF | year | {{...}} | {{...}} |
| D3 | DTF | location×year | {{...}} | {{...}} |
| D3 | SW | location | {{...}} | {{...}} |
| D3 | SW | year | {{...}} | {{...}} |
| D3 | SW | location×year | {{...}} | {{...}} |
| D4 | Oil content | location | 0.61 | −0.24 |
| D4 | Oil content | year | 0.65 | −0.32 |
| D4 | Oil content | location×year | 0.65| -0.21 |
| D4 | DTF | location | 0.41 | -0.57 |
| D4 | DTF | year | 0.51 | -5.64 |
| D4 | DTF | location×year | 0.52 | −5.79 |

### Table S9. Weather-branch and population-metadata ablations (relative to the full model).

| Dataset | Trait | Full R² | −Weather R² | −Weather ΔR² | −Population R² | −Population ΔR² |
|---|---|---:|---:|---:|---:|---:|
| D1 | yield | 0.62 | 0.57 | −0.05 | 0.61 | −0.01 | 
| D1 | DTF | 0.59 | 0.55 | −0.04 | 0.59 | 0.00 | 
| D1 | SW | 0.59 | 0.56	 | -0.03 | 0.59 | 0.00 | 
| D2 | FT | 0.50 | 0.46 | −0.04 | 0.50 | 0.00 | 
| D3 | yield | 0.45 | 0.40 | −0.05 | 0.44 | −0.01 | 
| D3 | DTF | 0.71 | 0.65 | −0.06 | 0.70 | −0.01 | 
| D3 | SW | 0.62 | 0.57 | −0.05 | 0.62 | 0.00 | 
| D4 | Oil content | 0.56 | 0.51 | −0.05 | 0.56 | 0.00 | 
| D4 | DTF | 0.52 | 0.49 | −0.03 | 0.52 | 0.00 | 

### Table S10. Seed stability of the model (reproducibility across random seeds).

| Dataset | # genotypes | Trait | # seeds | Mean R² | SD | Min | Max | Range |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| D1 | 195 | yield | 4 | 0.60 | 0.02 | 0.58 | 0.62 | 0.04 |
| D1 | 195 | DTF | 4 | 0.57 | 0.01 | 0.55 | 0.59 | 0.03 |
| D1 | 195 | SW | 4 | 0.56 | 0.04 | 0.53 | 0.59 | 0.08 |
| D2 | 378 | FT | 4 | 0.48 | 0.03 | 0.45 | 0.51 | 0.06 |
| D4 | 3{,}032 | Oil content | 4 | 0.56 | 0.01 | 0.56 | 0.56 | 0.01 |
| D4 | 3{,}032 | DTF | 4 | 0.52 | 0.01 | 0.52 | 0.52 | 0.02 |


### Table S11. Early in-season selection (partial-season prediction).

| Dataset | Trait | Full-season R² | 50% windows R² (% of full) | 75% windows R² (% of full) |
|---|---|---:|---:|---:|
| D1 | yield | 0.623 | 0.607 (97%) | 0.623 (100%) |
| D1 | DTF | 0.592 | 0.599 (101%) | 0.628 (106%) |
| D1 | SW | 0.598 | 0.554 (92%) | 0.580 (97%) |
| D2 | FT | 0.504 | 0.529 (105%) | 0.533 (106%) |
| D3 | yield | {{...}} | {{...}} | {{...}} |
| D3 | DTF | {{...}} | {{...}} | {{...}} |
| D3 | SW | {{...}} | {{...}} | {{...}} |
| D4 | Oil content | 0.566 | 0.562 | 0.558 |
| D4 | DTF | {{...}} | {{...}} | {{...}} |

## Validation of the model

### Table S12. Full-model predictive performance across four generalization regimes.

| Dataset | Trait | env-CV (*r* / R²) | geno-CV (*r* / R²) | env-blocked (*r* / R²) | pop-blocked (*r* / R²) |
|---|---|---|---|---|---|
| D1 | yield | 0.79 / 0.62 | 0.76 / 0.58 | 0.44 / −1.55 | 0.75 / 0.38 |
| D1 | DTF | 0.77 / 0.59 | 0.71 / 0.51 | 0.60 / −0.73 | 0.69 / −0.04 |
| D1 | SW | 0.77 / 0.59 | 0.72 / 0.52 | 0.66 / −0.81 | 0.68 / 0.09 |
| D2 | FT | 0.71 / 0.50 | 0.69 / 0.48 | 0.47 / −1.11 | 0.49 / −0.38 |
| D3 | yield | 0.67 / 0.45 | 0.66 / 0.43 | 0.42 / 0.00 | {{...}} |
| D3 | DTF | 0.84 / 0.71 | 0.82 / 0.68 | {{...}} | {{...}} |
| D3 | SW | 0.79 / 0.62 | 0.77 / 0.60 | {{...}} | {{...}} |
| D4 | Oil content | 0.75 / 0.56 | 0.75 / 0.56 | 0.63 / −0.26 | {{...}} |
| D4 | DTF | 0.72 / 0.52 | 0.71 / 0.51 | 0.48 / −4.00 | {{...}} |

### Table S13. Scenario-matched ablation of the weather and population branches.


| Dataset | Trait | Weather ΔR² · env-CV (redundant) | Weather Δr · env-blocked (informative) | Population ΔR² · env-CV (redundant) | Population Δr · pop-blocked (informative) |
|---|---|---:|---:|---:|---:|
| D1 | yield | −0.022 | +0.001 | −0.008 | +0.008 |
| D1 | DTF | −0.022 | +0.023 | +0.001 | −0.010 |
| D1 | SW | −0.011 | +0.008 | −0.034 | +0.024 |
| D2 | FT |  | {{...}} | | {{...}} |
| D3 | yield | {{...}} | {{...}} | {{...}} | {{...}} |
| D3 | DTF | {{...}} | {{...}} | {{...}} | {{...}} |
| D3 | SW | {{...}} | {{...}} | {{...}} | {{...}} |
| D4 | Oil content | {{...}} | {{...}} | {{...}} | {{...}} |
| D4 | DTF | {{...}} | {{...}} | {{...}} | {{...}} |

