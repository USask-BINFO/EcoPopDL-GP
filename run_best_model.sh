#!/usr/bin/env bash
# =============================================================================
# run_best_model.sh -- EcoPopDL-GP, best (automated) configuration.
#
# This is a thin wrapper around ./run.sh. It enables the automated-model
# settings reported in the paper and then forwards every argument you pass
# straight through to run.sh, so anything run.sh accepts works here too.
#
# What "best model" turns on
#   * learnable dual-branch gate  (the additive/deep mixing weight w is learned
#     rather than fixed)                     -> ECOPOP_DOSAGE_WEIGHT=learn
#   * multi-trait auxiliary heads (optional) -> ECOPOP_AUX_*
#   * genic-restricted additive branch (opt) -> ECOPOP_ADDITIVE_GENIC_IDS
#
# The gate is on by default and needs no extra files. The auxiliary heads and
# the genic filter need extra inputs, so they switch on only when you supply
# them with the flags below.
#
# QUICK START
#   ./run_best_model.sh --mode diploid --workdir out/d1_yield \
#       --genotype-source /data/D1/geno --trait-file /data/D1/pheno.csv \
#       --trait-col Yield --trial-file /data/D1/trials.csv
#
# WITH THE OPTIONAL COMPONENTS
#   ./run_best_model.sh --mode diploid --workdir out/d1_yield \
#       --aux-pheno /data/D1/aux_traits.csv --aux-targets Flowering,SW \
#       --genic-ids /data/D1/genic_snps.txt \
#       ... (all the usual run.sh options)
#
# EXTRA FLAGS ADDED BY THIS WRAPPER
#   --aux-pheno FILE     phenotype table holding the auxiliary traits
#   --aux-targets LIST   comma-separated auxiliary trait columns
#   --aux-weight FLOAT   loss weight for the auxiliary heads (default 0.2)
#   --genic-ids FILE     newline-separated SNP IDs to keep in the additive branch
#   --no-gate            disable the learnable gate (reverts to the fixed weight)
#   --seed INT           random seed
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/run.sh"
[[ -x "$RUN_SH" ]] || { echo "ERROR: run.sh not found or not executable at $RUN_SH" >&2; exit 1; }

# ---- best-model defaults -----------------------------------------------------
: "${ECOPOP_DOSAGE_WEIGHT:=learn}"      # learnable gate: the headline change
: "${ECOPOP_AUX_WEIGHT:=0.2}"

# ---- pull out wrapper-only flags; forward the rest verbatim ------------------
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aux-pheno)   export ECOPOP_AUX_PHENO="$2";          shift 2 ;;
    --aux-targets) export ECOPOP_AUX_TARGETS="$2";        shift 2 ;;
    --aux-weight)  ECOPOP_AUX_WEIGHT="$2";                shift 2 ;;
    --genic-ids)   export ECOPOP_ADDITIVE_GENIC_IDS="$2"; shift 2 ;;
    --seed)        export ECOPOP_SEED="$2";               shift 2 ;;
    --no-gate)     ECOPOP_DOSAGE_WEIGHT="fixed";          shift ;;
    -h|--help)     sed -n '2,45p' "${BASH_SOURCE[0]}"; echo; "$RUN_SH" --help; exit 0 ;;
    *)             FORWARD+=("$1");                       shift ;;
  esac
done

export ECOPOP_DOSAGE_WEIGHT ECOPOP_AUX_WEIGHT

# auxiliary heads need BOTH the table and the target list
if [[ -n "${ECOPOP_AUX_PHENO:-}" && -z "${ECOPOP_AUX_TARGETS:-}" ]] \
|| [[ -z "${ECOPOP_AUX_PHENO:-}" && -n "${ECOPOP_AUX_TARGETS:-}" ]]; then
  echo "ERROR: --aux-pheno and --aux-targets must be given together." >&2
  exit 1
fi
[[ -n "${ECOPOP_AUX_PHENO:-}" && ! -f "${ECOPOP_AUX_PHENO}" ]] && \
  { echo "ERROR: --aux-pheno file not found: $ECOPOP_AUX_PHENO" >&2; exit 1; }
[[ -n "${ECOPOP_ADDITIVE_GENIC_IDS:-}" && ! -f "${ECOPOP_ADDITIVE_GENIC_IDS}" ]] && \
  { echo "ERROR: --genic-ids file not found: $ECOPOP_ADDITIVE_GENIC_IDS" >&2; exit 1; }

# ---- report the active configuration ----------------------------------------
echo "=============================================================="
echo " EcoPopDL-GP - best (automated) configuration"
echo "   learnable gate      : ${ECOPOP_DOSAGE_WEIGHT}"
echo "   auxiliary heads     : ${ECOPOP_AUX_TARGETS:-off}${ECOPOP_AUX_TARGETS:+  (weight ${ECOPOP_AUX_WEIGHT})}"
echo "   genic additive filter: ${ECOPOP_ADDITIVE_GENIC_IDS:-off}"
echo "   seed                : ${ECOPOP_SEED:-default}"
echo "=============================================================="

exec "$RUN_SH" "${FORWARD[@]}"
