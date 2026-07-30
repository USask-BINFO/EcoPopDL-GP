# EcoPopDL-GP

EcoPopDL-GP predicts crop phenotypes from genotype, weather and trial metadata together. It is
built for multi-environment breeding trials, where the same lines are grown across locations and
years, and it works on diploid and polyploid crops.

One script runs everything: `run.sh`.

![EcoPopDL-GP pipeline overview](Images/EcopoDLGP-Pipeline.png)

---

## 1. Install

```bash
git clone https://github.com/USask-BINFO/EcoPopDL-GP.git
cd EcoPopDL-GP
pip install -r requirements.txt
```

You also need these on your PATH:

| Tool | Needed for |
|---|---|
| Python 3.10+ | everything |
| Java | Beagle imputation |
| `bcftools` | VCF handling |
| `admixture` | population clustering (skip with `--skip-admixture`) |

`plink`, `plink2`, `tabix` and Beagle are bundled in `Scripts/Data_preparation/Genotype/`. If the
bundled binaries do not run on your system, point at your own:

```bash
./run.sh --plink /usr/local/bin/plink --plink2 /usr/local/bin/plink2 --tabix /usr/bin/tabix
```

---

## 2. Run the bundled example

A small synthetic chickpea dataset ships in `Test/chickpea/` so you can check your install before
using your own data. This runs the whole pipeline end to end:

```bash
./run.sh \
  --mode diploid \
  --workdir out/demo \
  --stages all \
  --skip-admixture \
  --genotype-source Test/chickpea/synthetic_chickpea \
  --genotype-source-type plink \
  --trait-file Test/chickpea/synthetic_yield.csv \
  --trait-col Yield \
  --trial-file Test/chickpea/synthetic_trial.csv
```

It takes roughly 20 minutes, almost all of it in training. When it finishes you should have:

```text
out/demo/00_genotype/final_genotype.{bed,bim,fam}       genotype after QC
out/demo/02_phenotype/yield_mean.txt                    replicate-averaged phenotypes
out/demo/03_environment/env_matrix.csv                  weather matrix from NASA POWER
out/demo/04_tensors/tensors/<sample>/                   one ChromoTensor per line
out/demo/05_train/diploid/best_model_tmp.pt             trained model weights
out/demo/05_train/diploid/tier2_env_cv_fold_predictions.csv   per-fold predictions
```

The environment stage downloads weather from NASA POWER, so this step needs internet access.

**The synthetic data is 12 lines and 24 markers.** It exercises every stage but is far too small to
predict anything. Expect a negative validation R². That is normal and is not a sign of a broken
install. A polyploid equivalent is in `Test/bnapus/`; use `--mode polyploid`.

---

## 3. Run on your own data

You need three files. Only the first two are strictly required.

### Genotype

Point `--genotype-source` at what you have and let the launcher work out the format:

```bash
--genotype-source /path/to/your/data --genotype-source-type auto
```

Recognised: a PLINK `bed/bim/fam` prefix, a PLINK2 `pgen/pvar/psam` prefix, a `.vcf`/`.vcf.gz`, or a
raw Axiom export table. Anything else must be converted to one of those first. Add
`--run-beagle-imputation` if your genotypes have missing calls.

### Phenotype

A CSV with one row per line per trial. Column names are configurable; these are the defaults:

```csv
Name,Location,Year,SD,Pop,Yield
CHK_T01,Amber Plains,2021,1,1,97.8
CHK_T01,Amber Plains,2021,2,1,100.2
```

`Name`, `Location`, `Year` and your trait column are required. `Pop` is optional. Replicate columns
such as `SD` are detected and averaged over automatically, so you do not need `--group-cols` unless
you want to force a specific grouping.

Pass it with `--trait-file` and name the trait with `--trait-col`.

### Trial windows

Tells the pipeline which weather to fetch for each trial:

```csv
sample_id,lat,lon,start,end,place
CHK_T01,46.812,-108.441,2021-04-28,2021-09-12,Amber Plains
```

Pass it with `--trial-file`. If you do not have planting and harvest dates, omit `--trial-file` and
give a fixed season window instead; the launcher will build the trial file and geocode locations
from the `Location` column:

```bash
--env-season-start 05-01 --env-season-end 09-30
```

Exact dates give better results than a fixed window.

---

## 4. Optional annotation inputs

These are all optional. Without them the pipeline still runs; the corresponding ChromoTensor
channels are just filled with `none`.

| Flag | What it is |
|---|---|
| `--gene-gff FILE` | Gene annotation GFF3 for your reference |
| `--te-gff FILE` | Transposable-element annotation GFF3 |
| `--te-annotation FILE` | A TE annotation already in TSV form (see below) |
| `--homoeolog-pairs FILE` | Homoeologous gene pairs, polyploid only |

Use `--te-gff` **or** `--te-annotation`, not both. For polyploid tensor generation, `--gene-gff` and
`--homoeolog-pairs` are both required.

### What `te_gene_annotation.tsv` is

This is the file the older examples referred to. It is a **tab-separated** table of transposable
elements, one row per TE:

```text
chr	start	end	strand	ID	family	region	gene_id
cicar.CDCFrontier.gnm1.Ca8	3	63	+	TE277071	LINE/Penelope	intergenic	none
cicar.CDCFrontier.gnm1.Ca8	100	120	-	TE277070	DNA/TcMar-Fot1	intergenic	none
```

| Column | Required | Meaning |
|---|---|---|
| `chr` | yes | Chromosome name, must match your genotype's chromosome names |
| `start`, `end` | yes | 1-based TE coordinates |
| `ID` | yes | TE identifier, any unique string |
| `region` | yes | Where the TE sits, for example `intergenic`, `exon`, `intron` |
| `gene_id` | yes | Overlapping or nearest gene, or `none` |
| `strand`, `family` | no | Read but not used by the model |

**You do not have to build this by hand.** If you have a TE GFF3, pass `--te-gff` and the launcher
converts it into this schema for you. Supply `--te-annotation` only if you already have the TSV.

If the file is missing or a column is missing, tensor generation logs a warning, sets the TE
channels to `none`, and continues.

---

## 5. The model

There is one model. The configuration reported in the paper is the default, so a plain `run.sh` call
gives you the published setup. Specifically, and without you asking for any of it:

- the mixing weight between the additive and deep branches is **learned during training**
- regularization is **chosen by validation** from three levels (low, medium, high)
- in `--mode polyploid` the **homoeolog-context branch is enabled automatically**
- the model is **single-trait** unless you supply auxiliary traits

Three things you can change:

| Flag | Effect |
|---|---|
| `--aux-pheno FILE` + `--aux-targets LIST` | Switch to **multi-trait**. Extra traits supervise a shared representation. Needs both flags |
| `--genic-ids FILE` | Restrict the additive branch to a list of genic SNP IDs. Off by default |
| `--no-auto-reg` | Skip the regularization search and train once. About four times faster |

Multi-trait example:

```bash
./run.sh --mode diploid --workdir out/d1_yield \
  --genotype-source /data/D1/geno --genotype-source-type auto \
  --trait-file /data/D1/pheno.csv --trait-col Yield \
  --trial-file /data/D1/trials.csv \
  --aux-pheno /data/D1/pheno.csv --aux-targets DTF,SW
```

> **On runtime.** The regularization search trains four models instead of one: three short runs to
> pick the level, then the final model. On a large panel that is a lot of GPU time. Use
> `--no-auto-reg` while you are setting up and getting your file formats right, then drop it for the
> run you intend to report.

---

## 6. Stages

`--stages` takes any comma-separated subset. Upstream dependencies are added automatically, so
`--stages train` will build genotypes, phenotypes, environment and tensors first if they are absent.

```text
genotype  phenotype  admixture  env  tensors  train  all
```

Finished stages are skipped on re-run. Use `--force` to redo them.

Add `--skip-admixture` if you do not have the `admixture` binary or your panel is too small for
population clustering.

---

## 7. Output layout

```text
<workdir>/
  00_genotype/     QC'd genotypes, VCF, haplotype blocks
  01_admixture/    ancestry clusters, sample_clusters.csv
  02_phenotype/    metadata_<trait>.csv, <trait>_mean.txt
  03_environment/  env_matrix.csv, env_meta.json
  04_tensors/      one directory of ChromoTensors per line
  05_train/        trained model, metrics, embeddings, plots
  logs/  tmp/      run logs and the configured script copies
```

---

## 8. Reproducing the paper

`run_experiments.sh` drives the full evaluation protocol across all tasks: env-CV, geno-CV,
environment-blocked, population-blocked, the component ablations, multi-seed replicates and the
partial-season runs. The helpers in `Scripts/Model/` turn the run logs into the manuscript tables:

| Script | Produces |
|---|---|
| `collect_metrics.py` | per-task metric tables |
| `build_results_table.py` | the main results tables |
| `eval_env_blocked.py` | environment-blocked and population-blocked summaries |
| `aggregate_multiseed.py` | seed means and standard deviations |
| `emit_best_reg.py` | the regularization level chosen by validation |

Supplementary figures and tables: [Manuscript/Supplementary/SUPPLEMENTARY.md](Manuscript/Supplementary/SUPPLEMENTARY.md)

---

## 9. Reference: engine variables

Both training engines read `ECOPOP_*` environment variables, so anything can be set without editing
code. `run.sh` sets the common ones for you. These are what `run_experiments.sh` uses.

| Variable | Purpose |
|---|---|
| `ECOPOP_DOSAGE_WEIGHT` | `learn` for the learned gate, or a fixed number |
| `ECOPOP_AUX_PHENO`, `ECOPOP_AUX_TARGETS`, `ECOPOP_AUX_WEIGHT` | Multi-trait heads |
| `ECOPOP_ADDITIVE_GENIC_IDS` | Restrict the additive branch to a SNP-ID list |
| `ECOPOP_EVAL_MODE` | `env`, `geno_cv`, `env_blocked` |
| `ECOPOP_ENV_BLOCKED_MODES` | `location`, `year`, `loc_year`, `population` |
| `ECOPOP_ENV_WINDOW_FRAC` | Fraction of the season to use, for early selection |
| `ECOPOP_GXE_DROPOUT`, `ECOPOP_WEIGHT_DECAY` | Regularization, set by the search |
| `ECOPOP_USE_BAE`, `ECOPOP_USE_HABE`, `ECOPOP_USE_ADDITIVE`, `ECOPOP_USE_POP`, `ECOPOP_ABLATE_WEATHER`, `ECOPOP_USE_HOMOEOLOG` | Component ablations, `0` removes the component |
| `ECOPOP_SEED`, `ECOPOP_CV_FOLDS` | Reproducibility and fold count |
| `ECOPOP_3D_CHANNELS`, `ECOPOP_3D_PATH` | Optional 3D-genome channels, polyploid engine only |

---

## 10. Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'requests'` | `pip install -r requirements.txt` was not run |
| Environment stage hangs or fails | No internet access; NASA POWER cannot be reached |
| `0 haploblocks written` | Too few markers to form blocks. Expected on the synthetic test data |
| Negative validation R² on the test data | Expected; 12 lines and 24 markers carry no signal |
| ADMIXTURE errors on a small panel | Add `--skip-admixture` |
| Tensor stage reports missing TE annotation | Harmless. TE channels are set to `none` |

---

## Contact

Open an issue, or email [qnm481@usask.ca](mailto:qnm481@usask.ca).

## License

[MIT](LICENSE)
