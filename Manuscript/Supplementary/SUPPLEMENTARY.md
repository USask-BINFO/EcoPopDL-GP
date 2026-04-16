# Supplementary Figures and Tables for EcoPopDL-GP

This page hosts the supplementary material that will be linked from the manuscript.
Figure numbering and titles follow the manuscript order exactly for **Figures S2-S6** and **Tables S2-S5**.

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
