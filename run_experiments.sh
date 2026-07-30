#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  revision experiment matrix, run DIRECTLY on the
# existing Chromomap tensors (no run.sh, no tensor rebuild).
#
# HOW IT WORKS
#  * Two patched "engine" scripts do the training:  diploid/train.py, polyploid/train.py
#    (env-var controllable copies of your final D3_SW + Bnapus scripts; originals untouched).
#  * For each (dataset,trait) TASK you point at THAT TASK'S OWN dev script. The runner
#    reads its METADATA_FILE / ENVIRONMENT_FILE / TENSOR_DIR / TARGET_COL / N_MONTHS and
#    feeds them to the engine as env-vars — so the exact paths you used for the paper are
#    reused, with no hand-copying.
#  * Each experiment = the engine run once with ONE toggle env-var changed, into its own folder.
#
# USAGE
#   conda activate <your torch env>            # the env with torch+pandas+cyvcf2+pandas_plink
#   bash run_experiments.sh --check            # DRY RUN: print the config extracted per task. VERIFY IT.
#   bash run_experiments.sh --go               # actually run everything
#   bash run_experiments.sh --go d1_yield      # run only one task (optional filter)
#
# Results: revision/runs/<task>_<experiment>/   (models, cv_fold_predictions.csv, plots)
# =============================================================================
set -uo pipefail
REV="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"          # .../EcoPopDLGP/revision
BASE="$(cd "$REV/.." && pwd)"                                   # .../EcoPopDLGP  (task scripts live under here)
RUNROOT="${ECOPOP_RUNROOT:-$REV/runs}"; mkdir -p "$RUNROOT"   # ECOPOP_RUNROOT lets run2 redirect output per (dataset,trait)

# ---------------------------------------------------------------- 1) TASKS ----
# Format:  name|mode|<path to THAT task's dev script, relative to EcoPopDLGP/>
# The runner extracts the correct paths from each script, so just make sure each
# path points at the script whose TOP CONFIG matches that dataset+trait.
# (Run `--check` first: it prints what got extracted so you can confirm before launching.)
# Optional 4th field = per-task OVERRIDES (space-separated ECOPOP_KEY=VALUE), applied
# AFTER extraction — used to fix a stale/missing value in a source script.
D="/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo"
G="$D/input_files/Genotype/Axiom_genotype"
P1="$G/D1/imp.qc.all.withdc.clean"; P2="$G/D2/Genotype/imp.qc.all.withdc.clean"
P3="$G/D3/Genotype_files/Genotype/imp.qc.all.clean"; P4="$G/D4/Genotype_files/imp.qc.all.withdc.clean.fixed"
T1SNP="$D/Chromomap/images_AF_combined_d1_allele_comb_xyz_te_new_snp_rep/tensors"
TASKS=(
  "d1_yield|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/integrated_training_chromomap.py|ECOPOP_TENSOR_DIR=$T1SNP ECOPOP_PLINK_PREFIX=$P1 ECOPOP_DOSAGE_WEIGHT=0.1 ECOPOP_GXE_DROPOUT=0.3 ECOPOP_WEIGHT_DECAY=0.01"
  "d1_dtf|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/no_inter_intra_block/simple_traits/DTF_D1/integrated_training_chromomap.py|ECOPOP_PLINK_PREFIX=$P1 ECOPOP_DOSAGE_WEIGHT=learn"
  "d1_sw|diploid|D1/SW/integrated_training_chromomap.py|ECOPOP_TENSOR_DIR=$T1SNP ECOPOP_PLINK_PREFIX=$P1 ECOPOP_ENVIRONMENT_FILE=$D/input_files/Environment/D1/d1_env_matrix_SW.csv"
  "d2_ft|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/no_inter_intra_block/integrated_training_chromomap.py|ECOPOP_PLINK_PREFIX=$P2 ECOPOP_USE_ENV_ZSCORE=1 ECOPOP_GXE_DROPOUT=0.1 ECOPOP_WEIGHT_DECAY=0.001 ECOPOP_DOSAGE_WEIGHT=0.9"
  "d3_yield|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/no_inter_intra_block/D3_Yield/integrated_training_chromomap.py|ECOPOP_PLINK_PREFIX=$P3"
  "d3_dtf|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/no_inter_intra_block/D3_SW/integrated_training_chromomap.py|ECOPOP_METADATA_FILE=$D/input_files/Genotype/Axiom_genotype/D3/Phenotype/Phenotype_files/flowering.txt ECOPOP_ENVIRONMENT_FILE=$D/input_files/Environment/D3/d3_env_matrix_dtf.csv ECOPOP_TARGET_COL=DTF ECOPOP_PLINK_PREFIX=$P3"
  "d3_sw|diploid|D1/Yield/CV_Genotype/No_Ensemble/faster_version/faster_version/with_important_snps/no_inter_intra_block/D3_SW/integrated_training_chromomap.py|ECOPOP_PLINK_PREFIX=$P3"
  "d4_oil|polyploid|Bnapus/integrated_training_chromomap.py|ECOPOP_TENSOR_DIR=$D/Chromomap/Bnapus/images_AF_combined_d4_allele_comb_xyz_te_new_snp_rep_/tensors ECOPOP_PLINK_PREFIX=$P4"
  "d4_dtf|polyploid|Bnapus/DTF/Feature_importance/integrated_training_chromomap_D4_homfix_homgroup_signal.py|ECOPOP_PLINK_PREFIX=$P4"
)

# --------------------------------------------------------- 2) EXPERIMENTS -----
# Format:  expname|ENV_VARS   (empty for the baseline)
EXPERIMENTS=(
  "full|"                                                          # E0 baseline (env-CV, seed 20)
  "seed07|ECOPOP_SEED=7"                                           # E2 multi-seed stability (spread seeds; full uses default seed 20)
  "seed42|ECOPOP_SEED=42"
  "seed88|ECOPOP_SEED=88"
  "seed123|ECOPOP_SEED=123"
  "genocv|ECOPOP_EVAL_MODE=geno_cv"                               # unseen genotypes
  "envblocked|ECOPOP_EVAL_MODE=env_blocked ECOPOP_USE_ENV_ZSCORE=0"  # E1 truly-unseen environments
  "abl_bae|ECOPOP_USE_BAE=0"                                       # E3 ablations
  "abl_habe|ECOPOP_USE_HABE=0"
  "abl_add|ECOPOP_USE_ADDITIVE=0"
  "abl_weather|ECOPOP_ABLATE_WEATHER=1"                           # -Weather (comment #2)
  "abl_pop|ECOPOP_USE_POP=0"                                       # -Population (comment #2/#5)
  "abl_homoeolog|ECOPOP_USE_HOMOEOLOG=0"                           # -Homoeolog (polyploid D4 tasks only)
  "season50|ECOPOP_ENV_WINDOW_FRAC=0.5"                           # E5 early-selection (comment #6)
  "season75|ECOPOP_ENV_WINDOW_FRAC=0.75"
  "pop_blocked|ECOPOP_EVAL_MODE=env_blocked ECOPOP_ENV_BLOCKED_MODES=population ECOPOP_USE_ENV_ZSCORE=0"  # #5 leave-one-population-out
  # ---- SCENARIO-MATCHED ABLATIONS: ablate each component under the CV scheme that reveals its value ----
  "abl_weather_eb|ECOPOP_ABLATE_WEATHER=1 ECOPOP_EVAL_MODE=env_blocked ECOPOP_USE_ENV_ZSCORE=0"                          # weather VALUE under UNSEEN environments (compare vs envblocked, ranking r)
  "abl_pop_pb|ECOPOP_USE_POP=0 ECOPOP_EVAL_MODE=env_blocked ECOPOP_ENV_BLOCKED_MODES=population ECOPOP_USE_ENV_ZSCORE=0"  # population VALUE under UNSEEN populations (compare vs pop_blocked, ranking r)
  "abl_bae_gc|ECOPOP_USE_BAE=0 ECOPOP_EVAL_MODE=geno_cv"                                                                  # genomic components under UNSEEN genotypes (compare vs genocv)
  "abl_habe_gc|ECOPOP_USE_HABE=0 ECOPOP_EVAL_MODE=geno_cv"
  "abl_add_gc|ECOPOP_USE_ADDITIVE=0 ECOPOP_EVAL_MODE=geno_cv"
  "reactionnorm|ECOPOP_USE_ENV_ZSCORE=1"                          # reaction-norm: per-env target centering (env-CV; D2 diagnosis test)
  "rn_lowreg|ECOPOP_USE_ENV_ZSCORE=1 ECOPOP_GXE_DROPOUT=0.1 ECOPOP_WEIGHT_DECAY=0.001 ECOPOP_DOSAGE_WEIGHT=0.9"  # D2: reaction-norm + reduced reg + additive-leaning fusion
  # ---- per-dataset regularization sweep (isolates reg; no reaction-norm). "full" = reg_orig (dropout 0.4, wd 0.02) ----
  "reg_mid|ECOPOP_GXE_DROPOUT=0.25 ECOPOP_WEIGHT_DECAY=0.005"      # regularization sweep: medium
  "reg_low|ECOPOP_GXE_DROPOUT=0.1 ECOPOP_WEIGHT_DECAY=0.001"       # regularization sweep: light (the D2 winner, minus reaction-norm)
  "reg_high|ECOPOP_GXE_DROPOUT=0.5 ECOPOP_WEIGHT_DECAY=0.05"       # regularization sweep: heavy (for overfitting traits, e.g. D4 oil / yield)
  "learn_blend|ECOPOP_DOSAGE_WEIGHT=learn"                         # self-adjusting additive-vs-DL blend (paper reg) — test 0.62 recovery
  "rn_learn|ECOPOP_USE_ENV_ZSCORE=1 ECOPOP_GXE_DROPOUT=0.1 ECOPOP_WEIGHT_DECAY=0.001 ECOPOP_DOSAGE_WEIGHT=learn"  # magic + self-adjusting blend
  "valblend|ECOPOP_DOSAGE_WEIGHT=learn"                            # retired val-blend probe -> now routes to the learnable gate (same as learn_blend)
  "auxmt|ECOPOP_DOSAGE_WEIGHT=learn ECOPOP_AUX_TARGETS=Flowering,SW ECOPOP_AUX_PHENO=/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/EcoPopDLGP/revision/aux_pheno/d1_aux_flowering_sw.csv ECOPOP_AUX_WEIGHT=0.2"  # B3 multi-trait smoke: d1_yield + Flowering/SW aux heads
)
# (Still separate: baseline hyperparameter tuning #4 -> see Demo/Baseline_models/revision/.)

# ----------------------------------------------------------- helpers ----------
extract() {   # $1=script  $2=KEY   -> value (unquotes strings, strips comments)
  grep -m1 -E "^$2[[:space:]]*=" "$1" 2>/dev/null \
    | sed -E 's/^[^=]*=[[:space:]]*"?([^"#]*)"?.*$/\1/' \
    | sed -E 's/[[:space:]]+$//'
}
task_env() {  # $1=script  -> prints the ECOPOP_ path/target env assignments
  printf 'ECOPOP_METADATA_FILE=%s\n'    "$(extract "$1" METADATA_FILE)"
  printf 'ECOPOP_ENVIRONMENT_FILE=%s\n' "$(extract "$1" ENVIRONMENT_FILE)"
  printf 'ECOPOP_TENSOR_DIR=%s\n'       "$(extract "$1" TENSOR_DIR)"
  printf 'ECOPOP_TARGET_COL=%s\n'       "$(extract "$1" TARGET_COL)"
  printf 'ECOPOP_N_MONTHS=%s\n'         "$(extract "$1" N_MONTHS)"
}

MODE_FLAG="${1:---check}"
ONLY="${2:-}"

for task in "${TASKS[@]}"; do
  IFS='|' read -r NAME TMODE REL OVR <<< "$task"
  [[ -n "$ONLY" && "$NAME" != "$ONLY" ]] && continue
  SCRIPT="$BASE/$REL"
  CODE="$REV/$TMODE"                          # diploid/ or polyploid/
  if [[ ! -f "$SCRIPT" ]]; then echo "!! [$NAME] task script NOT FOUND: $SCRIPT"; continue; fi
  mapfile -t CFG < <(task_env "$SCRIPT")
  # apply per-task overrides (replace or add)
  if [[ -n "${OVR// }" ]]; then
    for kv in $OVR; do
      key="${kv%%=*}"; NEW=()
      for c in "${CFG[@]}"; do [[ "${c%%=*}" == "$key" ]] || NEW+=("$c"); done
      NEW+=("$kv"); CFG=("${NEW[@]}")
    done
  fi

  echo "============================================================"
  echo "TASK $NAME  ($TMODE)   engine: $CODE/train.py"
  echo "  source config: $REL"
  printf '  %s\n' "${CFG[@]}"
  # sanity: do the referenced files/dirs exist?
  for kv in "${CFG[@]}"; do
    key="${kv%%=*}"; val="${kv#*=}"
    case "$key" in
      ECOPOP_METADATA_FILE|ECOPOP_ENVIRONMENT_FILE) [[ -f "$val" ]] || echo "    !! missing file: $val";;
      ECOPOP_TENSOR_DIR) [[ -d "$val" ]] || echo "    !! missing tensor dir: $val";;
    esac
  done
  [[ "$MODE_FLAG" != "--go" ]] && continue

  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXPNAME EXPVARS <<< "$exp"
    # optional filter: ECOPOP_EXPERIMENTS="full genocv envblocked ..." runs only those (e.g. reduced set for slow D3)
    if [[ -n "${ECOPOP_EXPERIMENTS:-}" && " ${ECOPOP_EXPERIMENTS} " != *" ${EXPNAME} "* ]]; then continue; fi
    if [[ "${ECOPOP_FLAT_OUT:-0}" == "1" ]]; then OUT="$RUNROOT/$EXPNAME"; else OUT="$RUNROOT/${NAME}_${EXPNAME}"; fi; mkdir -p "$OUT"   # FLAT_OUT=1 -> run2/<D>/<trait>/<exp>/
    # ECOPOP_SKIP_DONE=1 -> resume: skip an experiment whose predictions already exist.
    if [[ "${ECOPOP_SKIP_DONE:-0}" == "1" ]] && ls "$OUT"/*fold_predictions.csv >/dev/null 2>&1; then
      echo ">>> [$NAME] $EXPNAME  SKIP (already complete)"; continue
    fi
    echo ">>> [$NAME] $EXPNAME  ${EXPVARS:-<baseline>}"
    # ECOPOP_EXTRA (highest priority: applied AFTER EXPVARS) lets run2_automodel force
    # gate+aux+picked-reg over each task's baked config without editing the TASKS table.
    # Use the EXPLICIT ecopop_env python (not bare `python`) so a changed PATH / active conda
    # env can't break the run midway with ModuleNotFoundError. Override via ECOPOP_PYTHON.
    ( cd "$OUT" && env "${CFG[@]}" $EXPVARS ${ECOPOP_EXTRA:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$CODE" "${ECOPOP_PYTHON:-/binfo-nas4/projects/ecopop_env/bin/python}" "$CODE/train.py" ) \
        > "$RUNROOT/${NAME}_${EXPNAME}.log" 2>&1 \
        || echo "    !! [$NAME/$EXPNAME] FAILED — see $RUNROOT/${NAME}_${EXPNAME}.log"
  done
done

echo
[[ "$MODE_FLAG" == "--go" ]] && echo "Done. Results in $RUNROOT/<task>_<experiment>/" \
  || echo "Dry run. Verify the extracted config above, then re-run with:  bash run_experiments.sh --go"
