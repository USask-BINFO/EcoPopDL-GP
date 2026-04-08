#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLINK_BIN_DEFAULT="$REPO_DIR/Scripts/Data_preparation/Genotype/plink"
PLINK2_BIN_DEFAULT="$REPO_DIR/Scripts/Data_preparation/Genotype/plink2"
TABIX_BIN_DEFAULT="$REPO_DIR/Scripts/Data_preparation/Genotype/tabix"
BEAGLE_JAR_DEFAULT="$REPO_DIR/Scripts/Data_preparation/Genotype/beagle.06Aug24.a91.jar"
BCFTOOLS_BIN_DEFAULT="${BCFTOOLS_BIN:-bcftools}"
ADMIXTURE_BIN_DEFAULT="$REPO_DIR/Scripts/Data_preparation/Genotype/admixture"
SAMTOOLS_BIN_DEFAULT="${SAMTOOLS_BIN:-samtools}"

usage() {
  cat <<'EOF'
EcoPopDL-GP end-to-end launcher

Usage:
  ./run.sh [options]

Core options:
  --mode diploid|polyploid           Model family to run.
  --workdir PATH                     Output working directory.
  --stages LIST                      Comma-separated stages.
                                     Available: genotype,phenotype,admixture,env,tensors,train,all
  --config FILE                      Shell-style config file (KEY=VALUE per line).
  --force                            Re-run stages even when outputs already exist.

Genotype entry (recommended):
  --genotype-source PATH             Single genotype input path or prefix.
  --genotype-source-type TYPE        auto|axiom|plink|pfile|vcf (default: auto)
                                     Aliases: bed/bfile/link -> plink, pgen -> pfile.

Legacy genotype inputs (still supported):
  --axiom-file FILE                  Raw Axiom genotype table for full preprocessing.
  --existing-plink-prefix PREFIX     Existing PLINK BED/BIM/FAM prefix to start from.
  --existing-pfile-prefix PREFIX     Existing PLINK2 PGEN/PVAR/PSAM prefix to start from.
  --existing-vcf FILE                Optional companion VCF for PLINK/PFILE starts, or legacy VCF input.
  --existing-hap-blocks FILE         Optional existing *.blocks.det file.

Genotype processing:
  --ref-mode none|genome|founders    REF/ALT strategy for Axiom conversion.
  --genotype-output-prefix NAME      Prefix for raw-genotype intermediate files.
  --fasta FILE                       Reference FASTA (and .fai if available).
  --founders CSV                     Comma-separated founder sample names for founder-based REF.
  --run-beagle-imputation            Enable QC + Beagle imputation for PLINK/PFILE/VCF starts.
  --threads INT                      CPU threads.
  --java-mem-gb INT                  Java memory for Beagle.
  --qc-geno FLOAT                    PLINK --geno threshold.
  --qc-mind FLOAT                    PLINK --mind threshold.
  --qc-maf FLOAT                     PLINK --maf threshold.
  --dr2-threshold FLOAT              Post-imputation DR2 threshold.
  --hard-call-threshold FLOAT        PLINK2 hard-call threshold after dosage import.

Phenotype:
  --trait-file FILE                  Raw phenotype CSV/TSV.
  --trait-col NAME                   Trait column in phenotype file.
  --target-col NAME                  Target column name used by the model (default: same as --trait-col).
  --sample-col NAME                  Sample/genotype ID column (default: Name).
  --location-col NAME                Location column (default: Location).
  --year-col NAME                    Year column (default: Year).
  --pop-col NAME                     Population column when present (default: Pop).
  --group-cols CSV                   Grouping columns before averaging replicates.
  --covariate-cols CSV               Covariate columns for per-environment covariate files.

Environment:
  --trial-file FILE                  Trial window file for NASA POWER extraction.
  --env-source-file FILE             Simpler phenotype-like file used to build a trial file when --trial-file is absent.
  --env-sample-col NAME              Sample column for generated trial files.
  --env-location-col NAME            Location/site column for generated trial files.
  --env-year-col NAME                Year column for generated trial files.
  --env-lat-col NAME                 Latitude column for generated trial files.
  --env-lon-col NAME                 Longitude column for generated trial files.
  --env-place-col NAME               Place column for generated trial files and geocoding.
  --env-start-col NAME               Explicit season-start column for generated trial files.
  --env-end-col NAME                 Explicit season-end column for generated trial files.
  --env-season-start-mm-dd MM-DD     Season start when explicit dates are absent.
  --env-season-end-mm-dd MM-DD       Season end when explicit dates are absent.
  --env-season-length-days INT       Season length when explicit end dates are absent.
  --env-windows INT                  Number of temporal windows (diploid default 32, polyploid 16).
  --env-align calendar|thermal       Window alignment mode.
  --env-vars CSV                     Environment variables list.
  --env-workers INT                  Parallel workers for API fetches.

Annotations / tensors:
  --gene-gff FILE                    Gene annotation GFF/GFF3.
  --te-annotation FILE               TE annotation TSV.
  --te-gff FILE                      TE GFF/GFF3/RepeatMasker GFF; converted for ChromoMap.
  --homoeolog-pairs FILE             Required for polyploid tensor generation.
  --color-mode dosage|allele_combination

Training overrides:
  --epochs INT                       Override NUM_EPOCHS in the training script.
  --batch-size INT                   Override BATCH_SIZE.
  --learning-rate FLOAT              Override LEARNING_RATE.
  --cv-folds INT                     Override CV_FOLDS.

ADMIXTURE:
  --skip-admixture                   Disable population-cluster inference.
  --admixture-k INT                  Use a fixed K.
  --admixture-k-range MIN:MAX        Try a K range and pick the best K from CV logs.
  --admixture-selection-method METHOD
                                     K-selection method for CV summary: elbow|min_cv (default: elbow).

Examples:
  ./run.sh \
    --mode diploid \
    --workdir runs/d1_yield \
    --axiom-file data/axiom_export.tsv \
    --ref-mode genome \
    --fasta data/reference.fa \
    --trait-file data/Yield_D1.csv \
    --trait-col Yield \
    --target-col Yield \
    --env-source-file data/Yield_D1.csv \
    --env-season-start-mm-dd 05-01 \
    --env-season-end-mm-dd 09-30 \
    --gene-gff data/genes.gff3 \
    --te-annotation data/te_gene_annotation.tsv

  ./run.sh \
    --mode polyploid \
    --workdir runs/d4_oil \
    --existing-plink-prefix data/imp.qc.all.withdc.clean.fixed \
    --trait-file data/D4_OIL_DB.csv \
    --trait-col OIL_DB \
    --target-col OIL_DB \
    --trial-file data/trial_data.csv \
    --gene-gff data/Bnapus_3DH.genes.gff3 \
    --homoeolog-pairs data/Bnapus_A_C_homoeolog_pairs.tsv
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

abspath() {
  "$PYTHON_BIN" - <<'PY' "$1"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

normalize_csv_list() {
  local raw="$1"
  raw="${raw// /}"
  raw="${raw#,}"
  raw="${raw%,}"
  printf '%s' "$raw"
}

slugify_name() {
  "$PYTHON_BIN" - <<'PY' "$1"
import os, re, sys
value = os.path.basename(sys.argv[1].strip())
value = re.sub(r'\.(vcf(\.gz|\.bgz)?|txt|tsv|csv|ped|map|bed|bim|fam|pgen|pvar|psam)$', '', value, flags=re.I)
value = re.sub(r'[^A-Za-z0-9_]+', '_', value)
value = re.sub(r'_+', '_', value).strip('_')
print(value or 'genotype_input')
PY
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Required directory not found: $path"
}

require_tool() {
  local tool="$1"
  if [[ "$tool" == */* ]]; then
    [[ -x "$tool" ]] || die "Required executable not found or not executable: $tool"
  else
    command -v "$tool" >/dev/null 2>&1 || die "Required command not found in PATH: $tool"
  fi
}

csv_has_column() {
  "$PYTHON_BIN" - <<'PY' "$1" "$2"
import pandas as pd, sys
path, col = sys.argv[1], sys.argv[2]
try:
    df = pd.read_csv(path, sep=None, engine='python', nrows=0)
except Exception:
    df = pd.read_csv(path, nrows=0)
print('1' if col in df.columns else '0')
PY
}

infer_chr_json_from_map() {
  "$PYTHON_BIN" - <<'PY' "$1"
import json, pandas as pd, sys
map_path = sys.argv[1]
df = pd.read_csv(map_path, sep=r'\s+', header=None, usecols=[0,3], names=['CHR','BP'])
df['CHR'] = df['CHR'].astype(str)
chr_max = df.groupby('CHR')['BP'].max().to_dict()
chr_max = {str(k): int(v) for k, v in chr_max.items()}
print(json.dumps(chr_max, separators=(',', ':')))
PY
}

normalize_admixture_bim_chromosomes() {
  "$PYTHON_BIN" - <<'PY' "$1" "$2"
import re
import sys
from pathlib import Path

bim_path = Path(sys.argv[1])
map_path = Path(sys.argv[2])

if not bim_path.exists():
    raise FileNotFoundError(f"Missing BIM file for ADMIXTURE normalization: {bim_path}")

rows = []
labels = []
with bim_path.open() as handle:
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Expected at least 6 BIM columns in {bim_path}, got: {line}")
        rows.append(parts)
        labels.append(parts[0])

def natural_key(value: str):
    out = []
    for chunk in re.split(r"(\d+)", value):
        if not chunk:
            continue
        if chunk.isdigit():
            out.append((0, int(chunk)))
        else:
            out.append((1, chunk.lower()))
    return out

sorted_labels = sorted(set(labels), key=natural_key)
mapping = {}
used_codes = set()

for label in sorted_labels:
    if re.fullmatch(r"[1-9]\d*", label):
        code = int(label)
        mapping[label] = str(code)
        used_codes.add(code)

next_code = 1
for label in sorted_labels:
    if label in mapping:
        continue
    while next_code in used_codes:
        next_code += 1
    mapping[label] = str(next_code)
    used_codes.add(next_code)

with bim_path.open("w") as handle:
    for parts in rows:
        parts[0] = mapping[parts[0]]
        handle.write("\t".join(parts[:6]) + "\n")

with map_path.open("w") as handle:
    handle.write("original_chr\tadmixture_chr\n")
    for label in sorted_labels:
        handle.write(f"{label}\t{mapping[label]}\n")
PY
}

run_admixture_job() {
  local admixture_dir="$1"
  local bed_basename="$2"
  local k="$3"
  local log_file="$4"
  (
    cd "$admixture_dir"
    "$ADMIXTURE_BIN" --cv -j"$THREADS" "${bed_basename}.bed" "$k" | tee "$log_file"
  )
}

convert_te_gff_to_annotation_tsv() {
  "$PYTHON_BIN" - <<'PY' "$1" "$2"
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

if not src.exists():
    raise FileNotFoundError(f"TE GFF not found: {src}")

cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attr"]
df = pd.read_csv(
    src,
    sep="\t",
    comment="#",
    names=cols,
    usecols=range(9),
    dtype=str,
    low_memory=False,
)

if df.empty:
    pd.DataFrame(columns=["chr", "start", "end", "ID", "region", "gene_id"]).to_csv(dst, sep="\t", index=False)
    raise SystemExit(0)

def parse_attr(raw: str) -> dict[str, str]:
    out = {}
    for field in str(raw or "").split(";"):
        field = field.strip()
        if not field or "=" not in field:
            continue
        key, value = field.split("=", 1)
        out[key.strip()] = value.strip()
    return out

attrs = df["attr"].map(parse_attr)
te = pd.DataFrame({
    "chr": df["seqid"].astype(str).str.strip(),
    "start": pd.to_numeric(df["start"], errors="coerce"),
    "end": pd.to_numeric(df["end"], errors="coerce"),
    "ID": attrs.map(lambda d: d.get("ID") or d.get("Name") or ""),
    "region": attrs.map(lambda d: d.get("Class") or d.get("Type") or "").replace("", pd.NA),
})
te["region"] = te["region"].fillna(df["type"].astype(str).str.strip()).replace("", "TE")
te = te.dropna(subset=["chr", "start", "end"]).copy()
te["start"] = te["start"].astype(int)
te["end"] = te["end"].astype(int)
te = te[te["end"] >= te["start"]].copy()
te = te.reset_index(drop=True)

missing_ids = te["ID"].astype(str).str.strip() == ""
if missing_ids.any():
    te.loc[missing_ids, "ID"] = [f"TE_{i+1}" for i in range(int(missing_ids.sum()))]
te["gene_id"] = te["ID"]

te = te[["chr", "start", "end", "ID", "region", "gene_id"]].sort_values(["chr", "start", "end", "ID"])
dst.parent.mkdir(parents=True, exist_ok=True)
te.to_csv(dst, sep="\t", index=False)
PY
}

ensure_fasta_index() {
  local fasta="$1"
  [[ -n "$fasta" ]] || return 0
  [[ -f "$fasta" ]] || return 0
  if [[ -f "${fasta}.fai" ]]; then
    return 0
  fi
  if command -v "$SAMTOOLS_BIN" >/dev/null 2>&1; then
    log "Creating FASTA index with $SAMTOOLS_BIN faidx"
    "$SAMTOOLS_BIN" faidx "$fasta"
  else
    log "FASTA index not found and samtools is unavailable; contig reheader will be skipped."
  fi
}

stage_enabled() {
  local name="$1"
  [[ "$STAGES" == "all" ]] && return 0
  local needle=",${STAGES},"
  [[ "$needle" == *",${name},"* ]]
}

expand_stage_list() {
  [[ "$STAGES" == "all" ]] && return 0
  local expanded=""
  local add_stage
  add_stage() {
    local stage_name="$1"
    [[ -n "$stage_name" ]] || return 0
    [[ ",$expanded," == *",${stage_name},"* ]] || expanded="${expanded:+$expanded,}$stage_name"
  }
  local part
  local -a requested=()
  IFS=',' read -r -a requested <<< "$STAGES"
  for part in "${requested[@]}"; do
    case "$part" in
      genotype) add_stage genotype ;;
      phenotype) add_stage phenotype ;;
      admixture) add_stage genotype; add_stage phenotype; add_stage admixture ;;
      env) add_stage env ;;
      tensors) add_stage genotype; add_stage tensors ;;
      train) add_stage genotype; add_stage phenotype; add_stage admixture; add_stage env; add_stage tensors; add_stage train ;;
      all) STAGES="all"; return 0 ;;
      "") ;;
      *) die "Unknown stage name in --stages: $part" ;;
    esac
  done
  STAGES="$expanded"
}

split_founders() {
  FOUNDERS_CSV="$(normalize_csv_list "$FOUNDERS_CSV")"
  FOUNDERS_ARR=()
  if [[ -n "$FOUNDERS_CSV" ]]; then
    IFS=',' read -r -a FOUNDERS_ARR <<< "$FOUNDERS_CSV"
  fi
}

split_env_vars() {
  ENV_VARS="$(normalize_csv_list "$ENV_VARS")"
  ENV_VARS_ARR=()
  IFS=',' read -r -a ENV_VARS_ARR <<< "$ENV_VARS"
}

clean_vcf_sample_ids() {
  local in_vcf="$1"
  local out_vcf="$2"
  local samples_txt="$TMP_DIR/samples.txt"
  local cleaned_txt="$TMP_DIR/samples_clean.txt"
  "$BCFTOOLS_BIN" query -l "$in_vcf" > "$samples_txt"
  "$PYTHON_BIN" - <<'PY' "$samples_txt" "$cleaned_txt"
import sys
from collections import Counter

src, dst = sys.argv[1], sys.argv[2]

def collapse_repeated_sample_id(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    for idx, ch in enumerate(value):
        if ch != "_":
            continue
        left = value[:idx]
        right = value[idx + 1 :]
        if left and left == right:
            return left
    return value

original = []
clean = []
with open(src) as handle:
    for value in map(str.strip, handle):
        original.append(value)
        clean.append(collapse_repeated_sample_id(value))

duplicates = [sid for sid, count in Counter(clean).items() if count > 1]
if duplicates:
    raise SystemExit(
        "Sample-ID cleanup created duplicate IDs: "
        + ", ".join(sorted(duplicates)[:10])
    )

with open(dst, 'w') as handle:
    handle.write('\n'.join(clean) + '\n')

changed = sum(before != after for before, after in zip(original, clean))
if changed:
    print(f"Cleaned {changed} duplicated VCF sample IDs.", file=sys.stderr)
PY
  "$BCFTOOLS_BIN" reheader -s "$cleaned_txt" -o "$out_vcf" "$in_vcf"
  "$TABIX_BIN" -f -p vcf "$out_vcf"
}

run_admixture_stage() {
  stage_enabled admixture || return 0
  [[ "$RUN_ADMIXTURE" -eq 1 ]] || { log "ADMIXTURE stage disabled."; return 0; }

  require_tool "$PLINK_BIN"
  require_tool "$ADMIXTURE_BIN"
  require_file "${FINAL_PLINK_PREFIX}.bed"
  require_file "$MODEL_METADATA_BASE"

  mkdir -p "$ADMIX_DIR"
  local pruned_prefix="$ADMIX_DIR/admixture_pruned"
  local input_prefix="$ADMIX_DIR/admixture_input"
  local chrom_map_tsv="$ADMIX_DIR/admixture_input.chrom_map.tsv"
  local final_cluster_csv="$ADMIX_DIR/sample_clusters.csv"
  local merged_metadata="$PHENO_DIR/metadata_${TRAIT_SLUG}_with_admixture.csv"
  local merged_mean="$PHENO_DIR/${TRAIT_SLUG}_mean_with_admixture.txt"

  if [[ "$FORCE" -eq 1 || ! -f "$final_cluster_csv" ]]; then
    log "Running LD pruning for ADMIXTURE input"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
      --indep-pairwise 50 5 0.2 --out "$pruned_prefix"

    log "Creating pruned BED for ADMIXTURE"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
      --extract "${pruned_prefix}.prune.in" \
      --make-bed --out "$input_prefix"
    log "Normalizing ADMIXTURE chromosome codes to integers"
    normalize_admixture_bim_chromosomes "${input_prefix}.bim" "$chrom_map_tsv"

    local chosen_k=""
    if [[ -n "$ADMIXTURE_K" ]]; then
      chosen_k="$ADMIXTURE_K"
      log "Running ADMIXTURE with fixed K=$chosen_k"
      run_admixture_job "$ADMIX_DIR" "admixture_input" "$chosen_k" "admixture_K${chosen_k}.out"
    else
      local min_k max_k
      IFS=':' read -r min_k max_k <<< "$ADMIXTURE_K_RANGE"
      [[ -n "$min_k" && -n "$max_k" ]] || die "Invalid --admixture-k-range. Expected MIN:MAX"
      for ((k=min_k; k<=max_k; k++)); do
        log "Running ADMIXTURE with K=$k"
        run_admixture_job "$ADMIX_DIR" "admixture_input" "$k" "admixture_K${k}.out"
      done
      local best_line
      best_line=$("$PYTHON_BIN" "$REPO_DIR/Scripts/utils/parse_admixture_cv.py" \
        --log-dir "$ADMIX_DIR" \
        --selection-method "$ADMIXTURE_SELECTION_METHOD" \
        --out-csv "$ADMIX_DIR/admixture_cv_summary.csv" \
        --out-plot "$ADMIX_DIR/admixture_cv_curve.png" | awk -F= '/Best_K=/{print $2; exit}')
      [[ -n "$best_line" ]] || die "Could not determine the best ADMIXTURE K from logs."
      chosen_k="$best_line"
      log "Selected ADMIXTURE K=$chosen_k using method=$ADMIXTURE_SELECTION_METHOD"
    fi

    local q_file="${input_prefix}.${chosen_k}.Q"
    require_file "$q_file"
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/admixture_assign_cli.py" \
      --q-file "$q_file" \
      --plink-prefix "$input_prefix" \
      --out "$final_cluster_csv"
  else
    log "Skipping ADMIXTURE; found $final_cluster_csv"
  fi

  "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/merge_population_clusters.py" \
    --metadata "$MODEL_METADATA_BASE" \
    --clusters "$final_cluster_csv" \
    --out "$merged_metadata"

  if [[ -f "$PHENO_MEAN_BASE" ]]; then
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/merge_population_clusters.py" \
      --metadata "$PHENO_MEAN_BASE" \
      --clusters "$final_cluster_csv" \
      --out "$merged_mean"
  fi

  MODEL_METADATA="$merged_metadata"
}

run_phenotype_stage() {
  stage_enabled phenotype || return 0
  require_file "$TRAIT_FILE"
  mkdir -p "$PHENO_DIR"

  local has_pop
  has_pop=$(csv_has_column "$TRAIT_FILE" "$POP_COL")

  local cmd=(
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/make_pheno_files_cli.py"
    --input "$TRAIT_FILE"
    --outdir "$PHENO_DIR"
    --sample-col "$SAMPLE_COL"
    --location-col "$LOCATION_COL"
    --year-col "$YEAR_COL"
    --trait-col "$TRAIT_COL"
    --target-col "$TARGET_COL"
    --metadata-name "metadata_${TRAIT_SLUG}.csv"
    --write-overall
    --write-covariates
  )
  if [[ "$has_pop" == "1" ]]; then
    cmd+=(--pop-col "$POP_COL")
  fi
  if [[ -n "$GROUP_COLS" ]]; then
    cmd+=(--group-cols "$GROUP_COLS")
  fi
  if [[ -n "$COVARIATE_COLS" ]]; then
    cmd+=(--covariate-cols "$COVARIATE_COLS")
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$MODEL_METADATA_BASE" || ! -f "$PHENO_MEAN_BASE" ]]; then
    log "Preparing phenotype metadata"
    "${cmd[@]}"
  else
    log "Skipping phenotype stage; found $MODEL_METADATA_BASE"
  fi

  MODEL_METADATA="$MODEL_METADATA_BASE"
}

prepare_trial_file() {
  if [[ -n "$TRIAL_FILE" ]]; then
    require_file "$TRIAL_FILE"
    return 0
  fi

  local env_source_file="${ENV_SOURCE_FILE:-$TRAIT_FILE}"
  [[ -n "$env_source_file" ]] || die "Provide --trial-file or --env-source-file."
  require_file "$env_source_file"

  local generated_trial="$ENV_DIR/trial_file.generated.csv"
  local env_sample_col="${ENV_SAMPLE_COL:-$SAMPLE_COL}"
  local env_location_col="${ENV_LOCATION_COL:-$LOCATION_COL}"
  local env_year_col="${ENV_YEAR_COL:-$YEAR_COL}"
  local cmd=(
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/build_trial_file_cli.py"
    --input "$env_source_file"
    --output "$generated_trial"
    --sample-col "$env_sample_col"
  )

  [[ -n "$env_location_col" ]] && cmd+=(--location-col "$env_location_col")
  [[ -n "$env_year_col" ]] && cmd+=(--year-col "$env_year_col")
  [[ -n "$ENV_LAT_COL" ]] && cmd+=(--lat-col "$ENV_LAT_COL")
  [[ -n "$ENV_LON_COL" ]] && cmd+=(--lon-col "$ENV_LON_COL")
  [[ -n "$ENV_PLACE_COL" ]] && cmd+=(--place-col "$ENV_PLACE_COL")
  [[ -n "$ENV_START_COL" ]] && cmd+=(--start-col "$ENV_START_COL")
  [[ -n "$ENV_END_COL" ]] && cmd+=(--end-col "$ENV_END_COL")
  [[ -n "$ENV_SEASON_START_MM_DD" ]] && cmd+=(--season-start-mm-dd "$ENV_SEASON_START_MM_DD")
  [[ -n "$ENV_SEASON_END_MM_DD" ]] && cmd+=(--season-end-mm-dd "$ENV_SEASON_END_MM_DD")
  [[ -n "$ENV_SEASON_LENGTH_DAYS" ]] && cmd+=(--season-length-days "$ENV_SEASON_LENGTH_DAYS")

  log "Building trial file from $env_source_file"
  "${cmd[@]}"
  TRIAL_FILE="$generated_trial"
}

run_environment_stage() {
  stage_enabled env || return 0
  mkdir -p "$ENV_DIR"
  prepare_trial_file

  if [[ "$FORCE" -eq 1 || ! -f "$ENV_MATRIX" ]]; then
    split_env_vars
    log "Building environment matrix from NASA POWER"
    (
      cd "$REPO_DIR/Scripts/Data_preparation/Env"
      PYTHONPATH="$REPO_DIR/Scripts/Data_preparation/Env${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" build_env_matrix_v2.py \
        --trials "$TRIAL_FILE" \
        --L "$ENV_WINDOWS" \
        --align "$ENV_ALIGN" \
        --out-matrix "$ENV_MATRIX" \
        --out-meta "$ENV_META" \
        --cache-dir "$ENV_CACHE_DIR" \
        --geocode-cache "$GEOCODE_CACHE" \
        --workers "$ENV_WORKERS" \
        --vars "${ENV_VARS_ARR[@]}"
    )
  else
    log "Skipping environment stage; found $ENV_MATRIX"
  fi
}

run_tensor_stage() {
  stage_enabled tensors || return 0
  require_file "$FINAL_PED"
  require_file "$FINAL_MAP"
  require_file "$HAP_BLOCK_FILE"
  mkdir -p "$TENSOR_DIR_OUT" "$TMP_DIR"
  [[ -n "$TE_GFF" && -n "$TE_ANNOTATION" ]] && die "Provide either --te-gff or --te-annotation, not both."
  [[ -n "$TE_GFF" ]] && require_file "$TE_GFF"

  if [[ "$MODE" == "polyploid" ]]; then
    require_file "$HOMOEOLOG_PAIR_FILE"
    require_file "$GENE_GFF"
    local tensor_source="$REPO_DIR/Scripts/Chromomap_tensor_generation/Polyploid/integrated_tile_generation.py"
    local tensor_py="$TMP_DIR/integrated_tile_generation.polyploid.py"
  else
    local tensor_source="$REPO_DIR/Scripts/Chromomap_tensor_generation/Diploid/integrated_tile_generation.py"
    local tensor_py="$TMP_DIR/integrated_tile_generation.diploid.py"
  fi

  local chr_json
  chr_json=$(infer_chr_json_from_map "$FINAL_MAP")
  local tensor_te_annotation="${TE_ANNOTATION:-}"
  if [[ -n "$TE_GFF" ]]; then
    tensor_te_annotation="$TENSOR_DIR_OUT/te_annotation.from_gff.tsv"
    if [[ "$FORCE" -eq 1 || ! -f "$tensor_te_annotation" ]]; then
      log "Converting TE GFF to ChromoMap TE annotation TSV"
      convert_te_gff_to_annotation_tsv "$TE_GFF" "$tensor_te_annotation"
    fi
  fi

  local -a render_cmd=(
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/render_configured_script.py"
    --source "$tensor_source"
    --dest "$tensor_py"
    --top TE_GENE_ANNOTATION_TSV="${tensor_te_annotation:-None}"
    --top GENE_GFF_PATH="${GENE_GFF:-None}"
    --func main vcf_path="$FINAL_VCF"
    --func main ped_file="$FINAL_PED"
    --func main map_file="$FINAL_MAP"
    --func main haplotype_block_file="$HAP_BLOCK_FILE"
    --func main ENCODED_WITH_BLOCKS="$TENSOR_DIR_OUT/encoded_genotypes_with_haplotype_blocks.csv"
    --func main out_folder_combined_tensor="$TENSOR_DIR_OUT/tensors"
  )

  if [[ "$MODE" == "polyploid" ]]; then
    render_cmd+=(--top chr_info="$chr_json")
    render_cmd+=(--top USE_SUBGENOMES=true)
  else
    render_cmd+=(--top USE_SUBGENOMES=false)
    render_cmd+=(--insert-after "# Global variables" "chr_info=$chr_json")
  fi

  log "Rendering configured tensor-generation script"
  "${render_cmd[@]}"

  if [[ "$FORCE" -eq 0 ]] && find "$TENSOR_DIR_OUT/tensors" -name '*_tensor.npz' -print -quit 2>/dev/null | grep -q .; then
    log "Skipping tensor stage; tensors already exist in $TENSOR_DIR_OUT/tensors"
    return 0
  fi

  log "Generating ChromoMap tensors"
  local -a env_cmd=()
  if [[ "$MODE" == "polyploid" ]]; then
    env_cmd+=(HOMOEOLOG_PAIR_FILE="$HOMOEOLOG_PAIR_FILE")
  fi
  (
    cd "$TENSOR_DIR_OUT"
    env "${env_cmd[@]}" "$PYTHON_BIN" "$tensor_py" --color-mode "$COLOR_MODE"
  )
}

run_training_stage() {
  stage_enabled train || return 0
  require_file "$MODEL_METADATA"
  require_file "$ENV_MATRIX"
  require_file "${FINAL_PLINK_PREFIX}.bed"
  mkdir -p "$TRAIN_DIR" "$TMP_DIR"

  local model_source model_dir model_py
  if [[ "$MODE" == "polyploid" ]]; then
    model_source="$REPO_DIR/Scripts/Model/Polyploid/integrated_training_chromomap.py"
    model_dir="$REPO_DIR/Scripts/Model/Polyploid"
    model_py="$TMP_DIR/integrated_training_chromomap.polyploid.py"
  else
    model_source="$REPO_DIR/Scripts/Model/Diploid/integrated_training_chromomap.py"
    model_dir="$REPO_DIR/Scripts/Model/Diploid"
    model_py="$TMP_DIR/integrated_training_chromomap.diploid.py"
  fi

  local -a render_cmd=(
    "$PYTHON_BIN" "$REPO_DIR/Scripts/utils/render_configured_script.py"
    --source "$model_source"
    --dest "$model_py"
    --top METADATA_FILE="$MODEL_METADATA"
    --top ENVIRONMENT_FILE="$ENV_MATRIX"
    --top PLINK_PREFIX="$FINAL_PLINK_PREFIX"
    --top VCF_PATH="$FINAL_VCF"
    --top TARGET_COL="$TARGET_COL"
    --top TENSOR_DIR="$TENSOR_DIR_OUT/tensors"
    --top N_MONTHS="$ENV_WINDOWS"
    --top NUM_WORKERS="$THREADS"
  )
  [[ -n "$EPOCHS" ]] && render_cmd+=(--top NUM_EPOCHS="$EPOCHS")
  [[ -n "$BATCH_SIZE" ]] && render_cmd+=(--top BATCH_SIZE="$BATCH_SIZE")
  [[ -n "$LEARNING_RATE" ]] && render_cmd+=(--top LEARNING_RATE="$LEARNING_RATE")
  [[ -n "$CV_FOLDS" ]] && render_cmd+=(--top CV_FOLDS="$CV_FOLDS")

  log "Rendering configured training script"
  "${render_cmd[@]}"

  if [[ "$FORCE" -eq 0 && -f "$TRAIN_DIR/best_model_chromomap.pt" ]]; then
    log "Skipping training stage; found $TRAIN_DIR/best_model_chromomap.pt"
    return 0
  fi

  log "Training $MODE model"
  (
    cd "$TRAIN_DIR"
    PYTHONPATH="$model_dir${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" "$model_py"
  )
}


normalize_source_type_alias() {
  local raw="${1:-auto}"
  raw="${raw}"
  case "$raw" in
    ""|auto) printf 'auto' ;;
    axiom|axiom_raw|raw_axiom|raw|table|csv|tsv|txt) printf 'axiom' ;;
    plink|plink_prefix|plink-bed|plink_bed|bfile|bed|link) printf 'plink' ;;
    pfile|pgen|plink2|plink2_prefix) printf 'pfile' ;;
    vcf|vcf.gz|vcf.bgz|bgz) printf 'vcf' ;;
    *) die "Unsupported genotype source type: $1" ;;
  esac
}

looks_like_axiom_table() {
  "$PYTHON_BIN" - <<'PY' "$1"
import pandas as pd, sys
path = sys.argv[1]
try:
    df = pd.read_csv(path, sep=None, engine='python', nrows=0)
except Exception:
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception:
        print('0')
        raise SystemExit(0)
required = {'probeset_id', 'SNP Marker', 'Chr_id', 'Start'}
print('1' if required.issubset(df.columns) else '0')
PY
}

resolve_plink_prefix() {
  local src="$1"
  local prefix="$src"
  case "$src" in
    *.bed|*.bim|*.fam) prefix="${src%.*}" ;;
  esac
  require_file "${prefix}.bed"
  require_file "${prefix}.bim"
  require_file "${prefix}.fam"
  printf '%s' "$prefix"
}

resolve_pfile_prefix() {
  local src="$1"
  local prefix="$src"
  case "$src" in
    *.pgen|*.pvar|*.psam) prefix="${src%.*}" ;;
    *.pvar.zst) prefix="${src%.pvar.zst}" ;;
  esac
  require_file "${prefix}.pgen"
  if [[ ! -f "${prefix}.pvar" && ! -f "${prefix}.pvar.zst" ]]; then
    die "Required PFILE variant file not found: ${prefix}.pvar or ${prefix}.pvar.zst"
  fi
  require_file "${prefix}.psam"
  printf '%s' "$prefix"
}

infer_genotype_source_type() {
  local src="$1"
  local lower="${src}"
  if [[ "$lower" == *.vcf || "$lower" == *.vcf.gz || "$lower" == *.vcf.bgz ]]; then
    printf 'vcf'
    return 0
  fi
  if [[ "$lower" == *.bed || "$lower" == *.bim || "$lower" == *.fam ]]; then
    printf 'plink'
    return 0
  fi
  if [[ "$lower" == *.pgen || "$lower" == *.pvar || "$lower" == *.psam || "$lower" == *.pvar.zst ]]; then
    printf 'pfile'
    return 0
  fi
  if [[ -f "${src}.bed" && -f "${src}.bim" && -f "${src}.fam" ]]; then
    printf 'plink'
    return 0
  fi
  if [[ -f "${src}.pgen" && ( -f "${src}.pvar" || -f "${src}.pvar.zst" ) && -f "${src}.psam" ]]; then
    printf 'pfile'
    return 0
  fi
  if [[ "$lower" == *.csv || "$lower" == *.tsv || "$lower" == *.txt ]]; then
    printf 'axiom'
    return 0
  fi
  if [[ -f "$src" ]]; then
    local axiom_like
    axiom_like=$(looks_like_axiom_table "$src")
    if [[ "$axiom_like" == "1" ]]; then
      printf 'axiom'
      return 0
    fi
  fi
  die "Could not infer genotype source type from '$src'. Use --genotype-source-type explicitly."
}

resolve_genotype_inputs() {
  local legacy_count=0
  [[ -n "$AXIOM_FILE" ]] && ((legacy_count+=1)) || true
  [[ -n "$EXISTING_PLINK_PREFIX" ]] && ((legacy_count+=1)) || true
  [[ -n "$EXISTING_PFILE_PREFIX" ]] && ((legacy_count+=1)) || true
  [[ -n "$VCF_SOURCE" ]] && ((legacy_count+=1)) || true

  if [[ -n "$GENOTYPE_SOURCE" && "$legacy_count" -gt 0 ]]; then
    die "Use either --genotype-source/--genotype-source-type or the legacy genotype flags, not both."
  fi

  if [[ -n "$GENOTYPE_SOURCE" ]]; then
    local src_type
    src_type=$(normalize_source_type_alias "$GENOTYPE_SOURCE_TYPE")
    if [[ "$src_type" == "auto" ]]; then
      src_type=$(infer_genotype_source_type "$GENOTYPE_SOURCE")
    fi
    log "Resolved genotype source '$GENOTYPE_SOURCE' as type '$src_type'"
    case "$src_type" in
      axiom)
        AXIOM_FILE="$GENOTYPE_SOURCE"
        ;;
      plink)
        EXISTING_PLINK_PREFIX=$(resolve_plink_prefix "$GENOTYPE_SOURCE")
        ;;
      pfile)
        EXISTING_PFILE_PREFIX=$(resolve_pfile_prefix "$GENOTYPE_SOURCE")
        ;;
      vcf)
        require_file "$GENOTYPE_SOURCE"
        VCF_SOURCE="$GENOTYPE_SOURCE"
        ;;
      *)
        die "Unhandled genotype source type: $src_type"
        ;;
    esac
    GENOTYPE_SOURCE_TYPE="$src_type"
  else
    [[ -n "$EXISTING_PLINK_PREFIX" ]] && EXISTING_PLINK_PREFIX=$(resolve_plink_prefix "$EXISTING_PLINK_PREFIX")
    [[ -n "$EXISTING_PFILE_PREFIX" ]] && EXISTING_PFILE_PREFIX=$(resolve_pfile_prefix "$EXISTING_PFILE_PREFIX")
    if [[ -n "$VCF_SOURCE" ]]; then
      require_file "$VCF_SOURCE"
    fi
  fi
}

copy_or_index_vcf() {
  local src_vcf="$1"
  local dst_vcf="$2"
  require_tool "$TABIX_BIN"
  if [[ "$src_vcf" == *.vcf.gz || "$src_vcf" == *.vcf.bgz ]]; then
    cp -f "$src_vcf" "$dst_vcf"
    if [[ -f "${src_vcf}.tbi" ]]; then
      cp -f "${src_vcf}.tbi" "${dst_vcf}.tbi"
    elif [[ -f "${src_vcf}.csi" ]]; then
      cp -f "${src_vcf}.csi" "${dst_vcf}.csi"
    else
      "$TABIX_BIN" -f -p vcf "$dst_vcf"
    fi
  else
    require_tool "$BCFTOOLS_BIN"
    "$BCFTOOLS_BIN" view -Oz -o "$dst_vcf" "$src_vcf"
    "$TABIX_BIN" -f -p vcf "$dst_vcf"
  fi
}

run_plink_qc_for_imputation() {
  local input_prefix="$1"
  local qc_prefix="$2"
  log "Running PLINK QC before Beagle imputation"
  "$PLINK2_BIN" --bfile "$input_prefix" \
    --geno "$QC_GENO" --mind "$QC_MIND" --maf "$QC_MAF" \
    --allow-extra-chr --make-bed --out "$qc_prefix"
}

run_beagle_imputation_from_qc_bed() {
  local qc_prefix="$1"
  local final_pfile="$2"

  require_tool "$PLINK_BIN"
  require_tool "$PLINK2_BIN"
  require_tool "$TABIX_BIN"
  require_tool "$BCFTOOLS_BIN"
  require_file "$BEAGLE_JAR"
  require_file "${qc_prefix}.bed"
  require_file "${qc_prefix}.bim"
  require_file "${qc_prefix}.fam"

  if [[ -n "$FASTA" && -f "$FASTA" ]]; then
    ensure_fasta_index "$FASTA"
  fi

  log "Imputing missing genotypes with Beagle"
  mapfile -t CHR_LIST < <(awk '{print $1}' "${qc_prefix}.bim" | sort -uV)
  [[ "${#CHR_LIST[@]}" -gt 0 ]] || die "Could not determine chromosome labels from ${qc_prefix}.bim"

  local -a qc_vcfs=()
  local chr
  for chr in "${CHR_LIST[@]}"; do
    local chr_vcf="$GENO_IMPUTE_DIR/vcf_imputed_after_qc1.${chr}.vcf.gz"
    local imputed_prefix="$GENO_IMPUTE_DIR/imputed.${chr}"
    local fixed_vcf="$GENO_IMPUTE_DIR/imp.${chr}.fix.vcf.gz"
    local filtered_vcf="$GENO_IMPUTE_DIR/imp.qc.${chr}.vcf.gz"

    "$PLINK2_BIN" --bfile "$qc_prefix" --allow-extra-chr \
      --chr "$chr" --export vcf bgz ref-first --out "$GENO_IMPUTE_DIR/vcf_imputed_after_qc1.${chr}"
    "$TABIX_BIN" -f -p vcf "$chr_vcf"

    java -Xmx"${JAVA_MEM_GB}g" -jar "$BEAGLE_JAR" \
      gt="$chr_vcf" \
      out="$imputed_prefix" \
      nthreads="$THREADS"

    if [[ -n "$FASTA" && -f "${FASTA}.fai" ]]; then
      "$BCFTOOLS_BIN" reheader -f "${FASTA}.fai" -o "$fixed_vcf" "${imputed_prefix}.vcf.gz"
    else
      cp -f "${imputed_prefix}.vcf.gz" "$fixed_vcf"
    fi
    "$BCFTOOLS_BIN" index -f "$fixed_vcf"

    "$BCFTOOLS_BIN" view -i "TYPE=\"snp\" && (INFO/DR2=\".\" || INFO/DR2>=${DR2_THRESHOLD})" \
      "$fixed_vcf" -Oz -o "$filtered_vcf"
    "$BCFTOOLS_BIN" index -f "$filtered_vcf"
    qc_vcfs+=("$filtered_vcf")
  done

  "$BCFTOOLS_BIN" concat -Oz -o "$GENO_DIR/imp.qc.all.vcf.gz" "${qc_vcfs[@]}"
  "$BCFTOOLS_BIN" index -f "$GENO_DIR/imp.qc.all.vcf.gz"

  "$PYTHON_BIN" "$REPO_DIR/Scripts/Data_preparation/Genotype/add_ds_from_gt.py" \
    "$GENO_DIR/imp.qc.all.vcf.gz" \
    "$GENO_DIR/imp.qc.all.withds.vcf.gz"

  clean_vcf_sample_ids "$GENO_DIR/imp.qc.all.withds.vcf.gz" "$FINAL_VCF"

  "$PLINK2_BIN" --vcf "$FINAL_VCF" --allow-extra-chr \
    --make-pgen --out "$final_pfile"

  "$PLINK2_BIN" --pfile "$final_pfile" --allow-extra-chr \
    --hard-call-threshold "$HARD_CALL_THRESHOLD" \
    --make-bed --out "$FINAL_PLINK_PREFIX"

  "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
    --recode --out "$FINAL_PLINK_PREFIX"

  log "Generating haplotype blocks"
  "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
    --blocks no-pheno-req --out "$GENO_DIR/haplotype_blocks"

  : > "$GENO_IMPUTE_MARKER"
}

prepare_existing_pfile() {
  require_tool "$PLINK_BIN"
  require_tool "$PLINK2_BIN"
  require_tool "$TABIX_BIN"
  require_file "${EXISTING_PFILE_PREFIX}.pgen"
  if [[ ! -f "${EXISTING_PFILE_PREFIX}.pvar" && ! -f "${EXISTING_PFILE_PREFIX}.pvar.zst" ]]; then
    die "Required PFILE variant file not found: ${EXISTING_PFILE_PREFIX}.pvar or ${EXISTING_PFILE_PREFIX}.pvar.zst"
  fi
  require_file "${EXISTING_PFILE_PREFIX}.psam"

  mkdir -p "$GENO_DIR"
  local input_prefix="$GENO_DIR/input_genotype"
  local qc_prefix="$GENO_DIR/qc1"
  local final_pfile="$GENO_DIR/final_genotype_pfile"
  FINAL_PLINK_PREFIX="$GENO_DIR/final_genotype"
  FINAL_VCF="$GENO_DIR/final_genotype.vcf.gz"
  FINAL_PED="$GENO_DIR/final_genotype.ped"
  FINAL_MAP="$GENO_DIR/final_genotype.map"
  HAP_BLOCK_FILE="$GENO_DIR/haplotype_blocks.blocks.det"

  local -a pfile_args=(--pfile "$EXISTING_PFILE_PREFIX")
  [[ -f "${EXISTING_PFILE_PREFIX}.pvar.zst" ]] && pfile_args+=(vzs)

  if [[ "$RUN_BEAGLE_IMPUTATION" -eq 1 ]]; then
    [[ -n "$EXISTING_VCF" ]] && log "Ignoring EXISTING_VCF because Beagle imputation is enabled."
    [[ -n "$EXISTING_HAP_BLOCK_FILE" ]] && log "Ignoring EXISTING_HAP_BLOCK_FILE because haplotype blocks will be regenerated after imputation."
    if [[ "$FORCE" -eq 1 || ! -f "$GENO_IMPUTE_MARKER" || ! -f "${FINAL_PLINK_PREFIX}.bed" || ! -f "$FINAL_VCF" || ! -f "$HAP_BLOCK_FILE" ]]; then
      log "Converting existing PFILE to BED/BIM/FAM for Beagle imputation"
      "$PLINK2_BIN" "${pfile_args[@]}" --allow-extra-chr \
        --hard-call-threshold "$HARD_CALL_THRESHOLD" \
        --make-bed --out "$input_prefix"
      run_plink_qc_for_imputation "$input_prefix" "$qc_prefix"
      run_beagle_imputation_from_qc_bed "$qc_prefix" "$final_pfile"
    else
      log "Skipping Beagle imputation for existing PFILE input; found $GENO_IMPUTE_MARKER"
    fi
    return 0
  fi

  if [[ "$FORCE" -eq 1 || ! -f "${FINAL_PLINK_PREFIX}.bed" ]]; then
    log "Converting existing PFILE to BED/BIM/FAM"
    "$PLINK2_BIN" "${pfile_args[@]}" --allow-extra-chr       --hard-call-threshold "$HARD_CALL_THRESHOLD"       --make-bed --out "$FINAL_PLINK_PREFIX"
  fi

  if [[ -n "$EXISTING_VCF" ]]; then
    require_file "$EXISTING_VCF"
    if [[ "$FORCE" -eq 1 || ! -f "$FINAL_VCF" ]]; then
      copy_or_index_vcf "$EXISTING_VCF" "$FINAL_VCF"
    fi
  elif [[ "$FORCE" -eq 1 || ! -f "$FINAL_VCF" ]]; then
    log "Exporting VCF from existing PFILE"
    "$PLINK2_BIN" "${pfile_args[@]}" --allow-extra-chr       --export vcf bgz id-paste=fid ref-first --out "$GENO_DIR/final_genotype"
    "$TABIX_BIN" -f -p vcf "$FINAL_VCF"
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$FINAL_PED" || ! -f "$FINAL_MAP" ]]; then
    log "Exporting PED/MAP from existing PFILE-derived BED"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr       --recode --out "$GENO_DIR/final_genotype"
  fi

  if [[ -n "$EXISTING_HAP_BLOCK_FILE" ]]; then
    require_file "$EXISTING_HAP_BLOCK_FILE"
    if [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
      cp -f "$EXISTING_HAP_BLOCK_FILE" "$HAP_BLOCK_FILE"
    fi
  elif [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
    log "Generating haplotype blocks from existing PFILE-derived BED"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr       --blocks no-pheno-req --out "$GENO_DIR/haplotype_blocks"
  fi
}

prepare_vcf_genotype() {
  require_tool "$PLINK_BIN"
  require_tool "$PLINK2_BIN"
  require_tool "$TABIX_BIN"
  require_file "$VCF_SOURCE"

  mkdir -p "$GENO_DIR"
  local input_prefix="$GENO_DIR/input_genotype"
  local final_pfile="$GENO_DIR/final_genotype_pfile"
  FINAL_PLINK_PREFIX="$GENO_DIR/final_genotype"
  FINAL_VCF="$GENO_DIR/final_genotype.vcf.gz"
  FINAL_PED="$GENO_DIR/final_genotype.ped"
  FINAL_MAP="$GENO_DIR/final_genotype.map"
  HAP_BLOCK_FILE="$GENO_DIR/haplotype_blocks.blocks.det"

  if [[ "$RUN_BEAGLE_IMPUTATION" -eq 1 ]]; then
    [[ -n "$EXISTING_HAP_BLOCK_FILE" ]] && log "Ignoring EXISTING_HAP_BLOCK_FILE because haplotype blocks will be regenerated after imputation."
    if [[ "$FORCE" -eq 1 || ! -f "$GENO_IMPUTE_MARKER" || ! -f "${FINAL_PLINK_PREFIX}.bed" || ! -f "$FINAL_VCF" || ! -f "$HAP_BLOCK_FILE" ]]; then
      if [[ "$FORCE" -eq 1 || ! -f "$input_prefix.bed" ]]; then
        log "Converting VCF to BED/BIM/FAM for Beagle imputation"
        "$PLINK2_BIN" --vcf "$VCF_SOURCE" --allow-extra-chr \
          --make-pgen --out "$GENO_DIR/input_genotype_pfile"
        "$PLINK2_BIN" --pfile "$GENO_DIR/input_genotype_pfile" --allow-extra-chr \
          --hard-call-threshold "$HARD_CALL_THRESHOLD" \
          --make-bed --out "$input_prefix"
      fi
      run_plink_qc_for_imputation "$input_prefix" "$GENO_DIR/qc1"
      run_beagle_imputation_from_qc_bed "$GENO_DIR/qc1" "$final_pfile"
    else
      log "Skipping Beagle imputation for VCF input; found $GENO_IMPUTE_MARKER"
    fi
    return 0
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$FINAL_VCF" ]]; then
    log "Copying/compressing VCF into workdir"
    copy_or_index_vcf "$VCF_SOURCE" "$FINAL_VCF"
  fi

  if [[ "$FORCE" -eq 1 || ! -f "${FINAL_PLINK_PREFIX}.bed" ]]; then
    log "Converting VCF to PLINK files"
    "$PLINK2_BIN" --vcf "$FINAL_VCF" --allow-extra-chr       --make-pgen --out "$GENO_DIR/final_genotype_pfile"
    "$PLINK2_BIN" --pfile "$GENO_DIR/final_genotype_pfile" --allow-extra-chr       --hard-call-threshold "$HARD_CALL_THRESHOLD"       --make-bed --out "$FINAL_PLINK_PREFIX"
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$FINAL_PED" || ! -f "$FINAL_MAP" ]]; then
    log "Exporting PED/MAP from VCF-derived BED"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr       --recode --out "$GENO_DIR/final_genotype"
  fi

  if [[ -n "$EXISTING_HAP_BLOCK_FILE" ]]; then
    require_file "$EXISTING_HAP_BLOCK_FILE"
    if [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
      cp -f "$EXISTING_HAP_BLOCK_FILE" "$HAP_BLOCK_FILE"
    fi
  elif [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
    log "Generating haplotype blocks from VCF-derived BED"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr       --blocks no-pheno-req --out "$GENO_DIR/haplotype_blocks"
  fi
}

prepare_existing_genotype() {
  require_tool "$PLINK_BIN"
  require_tool "$PLINK2_BIN"
  require_tool "$TABIX_BIN"
  require_file "${EXISTING_PLINK_PREFIX}.bed"
  require_file "${EXISTING_PLINK_PREFIX}.bim"
  require_file "${EXISTING_PLINK_PREFIX}.fam"

  mkdir -p "$GENO_DIR"
  local input_prefix="$GENO_DIR/input_genotype"
  local qc_prefix="$GENO_DIR/qc1"
  local final_pfile="$GENO_DIR/final_genotype_pfile"
  FINAL_PLINK_PREFIX="$GENO_DIR/final_genotype"
  FINAL_VCF="$GENO_DIR/final_genotype.vcf.gz"
  FINAL_PED="$GENO_DIR/final_genotype.ped"
  FINAL_MAP="$GENO_DIR/final_genotype.map"
  HAP_BLOCK_FILE="$GENO_DIR/haplotype_blocks.blocks.det"

  if [[ "$RUN_BEAGLE_IMPUTATION" -eq 1 ]]; then
    [[ -n "$EXISTING_VCF" ]] && log "Ignoring EXISTING_VCF because Beagle imputation is enabled."
    [[ -n "$EXISTING_HAP_BLOCK_FILE" ]] && log "Ignoring EXISTING_HAP_BLOCK_FILE because haplotype blocks will be regenerated after imputation."
    if [[ "$FORCE" -eq 1 || ! -f "$GENO_IMPUTE_MARKER" || ! -f "${FINAL_PLINK_PREFIX}.bed" || ! -f "$FINAL_VCF" || ! -f "$HAP_BLOCK_FILE" ]]; then
      log "Using existing PLINK BED/BIM/FAM input; QC and Beagle imputation will run."
      if [[ "$FORCE" -eq 1 || ! -f "${input_prefix}.bed" ]]; then
        log "Copying existing PLINK files into workdir for Beagle imputation"
        cp -f "${EXISTING_PLINK_PREFIX}.bed" "${input_prefix}.bed"
        cp -f "${EXISTING_PLINK_PREFIX}.bim" "${input_prefix}.bim"
        cp -f "${EXISTING_PLINK_PREFIX}.fam" "${input_prefix}.fam"
      fi
      run_plink_qc_for_imputation "$input_prefix" "$qc_prefix"
      run_beagle_imputation_from_qc_bed "$qc_prefix" "$final_pfile"
    else
      log "Skipping Beagle imputation for existing PLINK input; found $GENO_IMPUTE_MARKER"
    fi
    return 0
  fi

  log "Using existing PLINK BED/BIM/FAM input; skipping raw QC and Beagle imputation."

  if [[ "$FORCE" -eq 1 || ! -f "${FINAL_PLINK_PREFIX}.bed" ]]; then
    log "Copying existing PLINK files into workdir"
    cp -f "${EXISTING_PLINK_PREFIX}.bed" "${FINAL_PLINK_PREFIX}.bed"
    cp -f "${EXISTING_PLINK_PREFIX}.bim" "${FINAL_PLINK_PREFIX}.bim"
    cp -f "${EXISTING_PLINK_PREFIX}.fam" "${FINAL_PLINK_PREFIX}.fam"
  fi

  if [[ -n "$EXISTING_VCF" ]]; then
    require_file "$EXISTING_VCF"
    if [[ "$FORCE" -eq 1 || ! -f "$FINAL_VCF" ]]; then
      cp -f "$EXISTING_VCF" "$FINAL_VCF"
      [[ -f "${EXISTING_VCF}.tbi" ]] && cp -f "${EXISTING_VCF}.tbi" "${FINAL_VCF}.tbi" || true
    fi
  elif [[ "$FORCE" -eq 1 || ! -f "$FINAL_VCF" ]]; then
    log "Exporting VCF from existing BED/BIM/FAM"
    "$PLINK2_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
      --export vcf bgz id-paste=fid ref-first --out "$GENO_DIR/final_genotype"
    "$TABIX_BIN" -f -p vcf "$FINAL_VCF"
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$FINAL_PED" || ! -f "$FINAL_MAP" ]]; then
    log "Exporting PED/MAP from existing BED/BIM/FAM"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
      --recode --out "$GENO_DIR/final_genotype"
  fi

  if [[ -n "$EXISTING_HAP_BLOCK_FILE" ]]; then
    require_file "$EXISTING_HAP_BLOCK_FILE"
    if [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
      cp -f "$EXISTING_HAP_BLOCK_FILE" "$HAP_BLOCK_FILE"
    fi
  elif [[ "$FORCE" -eq 1 || ! -f "$HAP_BLOCK_FILE" ]]; then
    log "Generating haplotype blocks from existing BED/BIM/FAM"
    "$PLINK_BIN" --bfile "$FINAL_PLINK_PREFIX" --allow-extra-chr \
      --blocks no-pheno-req --out "$GENO_DIR/haplotype_blocks"
  fi
}

prepare_axiom_genotype() {
  require_tool "$PLINK_BIN"
  require_tool "$PLINK2_BIN"
  require_tool "$TABIX_BIN"
  require_tool "$BCFTOOLS_BIN"
  require_file "$AXIOM_FILE"
  require_file "$BEAGLE_JAR"
  log "Using raw Axiom input; QC and Beagle imputation will run."
  if [[ "$REF_MODE" == "genome" ]]; then
    require_file "$FASTA"
    ensure_fasta_index "$FASTA"
  fi

  mkdir -p "$GENO_DIR" "$GENO_IMPUTE_DIR" "$TMP_DIR"
  local prefix_slug
  prefix_slug="$(slugify_name "${GENOTYPE_OUTPUT_PREFIX:-$AXIOM_FILE}")"
  local out_prefix="$GENO_DIR/$prefix_slug"
  local raw_prefix="$GENO_DIR/${prefix_slug}_raw"
  local refa2_prefix="$GENO_DIR/${prefix_slug}_raw_refA2"
  local qc_prefix="$GENO_DIR/qc1"
  local final_pfile="$GENO_DIR/imp.qc.all.withds.clean"

  FINAL_PLINK_PREFIX="$final_pfile"
  FINAL_VCF="$GENO_DIR/imp.qc.all.withds.clean.vcf.gz"
  FINAL_PED="$GENO_DIR/imp.qc.all.withds.clean.ped"
  FINAL_MAP="$GENO_DIR/imp.qc.all.withds.clean.map"
  HAP_BLOCK_FILE="$GENO_DIR/haplotype_blocks.blocks.det"

  if [[ "$FORCE" -eq 1 || ! -f "${FINAL_PLINK_PREFIX}.bed" || ! -f "$FINAL_VCF" || ! -f "$HAP_BLOCK_FILE" ]]; then
    log "Using raw-genotype output prefix: $prefix_slug"
    log "Converting Axiom genotype table to PLINK PED/MAP"
    local -a axiom_cmd=(
      "$PYTHON_BIN" "$REPO_DIR/Scripts/Data_preparation/Genotype/axiom_snp_to_plink_refalt.py"
      --in "$AXIOM_FILE"
      --sep "$AXIOM_SEP"
      --out_prefix "$out_prefix"
      --ref_mode "$REF_MODE"
    )
    if [[ "$REF_MODE" == "genome" ]]; then
      axiom_cmd+=(--fasta "$FASTA")
    elif [[ "$REF_MODE" == "founders" ]]; then
      split_founders
      [[ "${#FOUNDERS_ARR[@]}" -gt 0 ]] || die "--founders is required when --ref-mode founders"
      axiom_cmd+=(--founders "${FOUNDERS_ARR[@]}")
    fi
    "${axiom_cmd[@]}"

    log "Making binary PLINK files from PED/MAP"
    "$PLINK_BIN" --file "$out_prefix" --allow-extra-chr \
      --make-bed --out "$raw_prefix"
    if [[ -f "${out_prefix}.alleles_refalt.txt" ]]; then
      "$PLINK2_BIN" --bfile "$raw_prefix" \
        --a2-allele "${out_prefix}.alleles_refalt.txt" 1 2 \
        --keep-allele-order --make-bed --out "$refa2_prefix"
    else
      cp -f "${raw_prefix}.bed" "${refa2_prefix}.bed"
      cp -f "${raw_prefix}.bim" "${refa2_prefix}.bim"
      cp -f "${raw_prefix}.fam" "${refa2_prefix}.fam"
    fi

    log "Running PLINK QC"
    run_plink_qc_for_imputation "$refa2_prefix" "$qc_prefix"

    "$PLINK_BIN" --bfile "$qc_prefix" --allow-extra-chr \
      --recode --out "$qc_prefix"

    "$PLINK2_BIN" --bfile "$qc_prefix" --allow-extra-chr \
      --export vcf bgz id-paste=fid --out "$qc_prefix"
    "$TABIX_BIN" -f -p vcf "${qc_prefix}.vcf.gz"

    run_beagle_imputation_from_qc_bed "$qc_prefix" "$final_pfile"
  else
    log "Skipping raw genotype preprocessing; found final genotype outputs in $GENO_DIR"
  fi
}

run_genotype_stage() {
  stage_enabled genotype || return 0
  mkdir -p "$GENO_DIR" "$GENO_IMPUTE_DIR" "$TMP_DIR"
  resolve_genotype_inputs
  if [[ -n "$EXISTING_PLINK_PREFIX" ]]; then
    prepare_existing_genotype
  elif [[ -n "$EXISTING_PFILE_PREFIX" ]]; then
    prepare_existing_pfile
  elif [[ -n "$VCF_SOURCE" ]]; then
    prepare_vcf_genotype
  elif [[ -n "$AXIOM_FILE" ]]; then
    prepare_axiom_genotype
  else
    die "Provide a genotype entry point with --genotype-source/--genotype-source-type, or use the legacy genotype flags."
  fi

  require_file "${FINAL_PLINK_PREFIX}.bed"
  require_file "$FINAL_PED"
  require_file "$FINAL_MAP"
  require_file "$FINAL_VCF"
  require_file "$HAP_BLOCK_FILE"
}

# ----------------------------
# Defaults
# ----------------------------
MODE="diploid"
WORKDIR="$REPO_DIR/runs/default"
STAGES="all"
CONFIG_FILE=""
FORCE=0
RUN_ADMIXTURE=1

GENOTYPE_SOURCE=""
GENOTYPE_SOURCE_TYPE="auto"
AXIOM_FILE=""
EXISTING_PLINK_PREFIX=""
EXISTING_PFILE_PREFIX=""
VCF_SOURCE=""
EXISTING_VCF=""
EXISTING_HAP_BLOCK_FILE=""
AXIOM_SEP="tab"
REF_MODE="none"
FASTA=""
FOUNDERS_CSV=""
GENOTYPE_OUTPUT_PREFIX=""
THREADS=8
JAVA_MEM_GB=16
QC_GENO=0.02
QC_MIND=0.02
QC_MAF=0.01
DR2_THRESHOLD=0.8
HARD_CALL_THRESHOLD=0.1
RUN_BEAGLE_IMPUTATION=0

TRAIT_FILE=""
TRAIT_COL=""
TARGET_COL=""
SAMPLE_COL="Name"
LOCATION_COL="Location"
YEAR_COL="Year"
POP_COL="Pop"
GROUP_COLS=""
COVARIATE_COLS=""

TRIAL_FILE=""
ENV_SOURCE_FILE=""
ENV_SAMPLE_COL=""
ENV_LOCATION_COL=""
ENV_YEAR_COL=""
ENV_LAT_COL=""
ENV_LON_COL=""
ENV_PLACE_COL=""
ENV_START_COL=""
ENV_END_COL=""
ENV_SEASON_START_MM_DD=""
ENV_SEASON_END_MM_DD=""
ENV_SEASON_LENGTH_DAYS=""
ENV_WINDOWS=""
ENV_ALIGN="calendar"
ENV_VARS="tmax_C,tmin_C,precip_mm,par_allsky,srad_allsky,wind_m_s,vpd_kPa,gdd,cloud_pct,kt,daylength_h"
ENV_WORKERS=4

GENE_GFF=""
TE_ANNOTATION=""
TE_GFF=""
HOMOEOLOG_PAIR_FILE=""
COLOR_MODE="allele_combination"

ADMIXTURE_K=""
ADMIXTURE_K_RANGE="2:10"
ADMIXTURE_SELECTION_METHOD="elbow"

EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
CV_FOLDS=""

PLINK_BIN="$PLINK_BIN_DEFAULT"
PLINK2_BIN="$PLINK2_BIN_DEFAULT"
TABIX_BIN="$TABIX_BIN_DEFAULT"
BEAGLE_JAR="$BEAGLE_JAR_DEFAULT"
BCFTOOLS_BIN="$BCFTOOLS_BIN_DEFAULT"
ADMIXTURE_BIN="$ADMIXTURE_BIN_DEFAULT"
SAMTOOLS_BIN="$SAMTOOLS_BIN_DEFAULT"

# ----------------------------
# Config pre-pass
# ----------------------------
ARGV=("$@")
for ((i=0; i<${#ARGV[@]}; i++)); do
  if [[ "${ARGV[$i]}" == "--config" ]]; then
    (( i + 1 < ${#ARGV[@]} )) || die "Missing value after --config"
    CONFIG_FILE="${ARGV[$((i+1))]}"
    break
  fi
done
if [[ -n "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$(abspath "$CONFIG_FILE")"
  require_file "$CONFIG_FILE"
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi

# ----------------------------
# CLI parsing
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;

    --genotype-source) GENOTYPE_SOURCE="$2"; shift 2 ;;
    --genotype-source-type) GENOTYPE_SOURCE_TYPE="$2"; shift 2 ;;
    --axiom-file) AXIOM_FILE="$2"; shift 2 ;;
    --existing-plink-prefix) EXISTING_PLINK_PREFIX="$2"; shift 2 ;;
    --existing-pfile-prefix) EXISTING_PFILE_PREFIX="$2"; shift 2 ;;
    --vcf-source) VCF_SOURCE="$2"; shift 2 ;;
    --existing-vcf) EXISTING_VCF="$2"; shift 2 ;;
    --existing-hap-blocks) EXISTING_HAP_BLOCK_FILE="$2"; shift 2 ;;
    --axiom-sep) AXIOM_SEP="$2"; shift 2 ;;
    --ref-mode) REF_MODE="$2"; shift 2 ;;
    --genotype-output-prefix) GENOTYPE_OUTPUT_PREFIX="$2"; shift 2 ;;
    --fasta) FASTA="$2"; shift 2 ;;
    --founders) FOUNDERS_CSV="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --java-mem-gb) JAVA_MEM_GB="$2"; shift 2 ;;
    --qc-geno) QC_GENO="$2"; shift 2 ;;
    --qc-mind) QC_MIND="$2"; shift 2 ;;
    --qc-maf) QC_MAF="$2"; shift 2 ;;
    --dr2-threshold) DR2_THRESHOLD="$2"; shift 2 ;;
    --hard-call-threshold) HARD_CALL_THRESHOLD="$2"; shift 2 ;;
    --run-beagle-imputation) RUN_BEAGLE_IMPUTATION=1; shift ;;

    --trait-file) TRAIT_FILE="$2"; shift 2 ;;
    --trait-col) TRAIT_COL="$2"; shift 2 ;;
    --target-col) TARGET_COL="$2"; shift 2 ;;
    --sample-col) SAMPLE_COL="$2"; shift 2 ;;
    --location-col) LOCATION_COL="$2"; shift 2 ;;
    --year-col) YEAR_COL="$2"; shift 2 ;;
    --pop-col) POP_COL="$2"; shift 2 ;;
    --group-cols) GROUP_COLS="$2"; shift 2 ;;
    --covariate-cols) COVARIATE_COLS="$2"; shift 2 ;;

    --trial-file) TRIAL_FILE="$2"; shift 2 ;;
    --env-source-file) ENV_SOURCE_FILE="$2"; shift 2 ;;
    --env-sample-col) ENV_SAMPLE_COL="$2"; shift 2 ;;
    --env-location-col) ENV_LOCATION_COL="$2"; shift 2 ;;
    --env-year-col) ENV_YEAR_COL="$2"; shift 2 ;;
    --env-lat-col) ENV_LAT_COL="$2"; shift 2 ;;
    --env-lon-col) ENV_LON_COL="$2"; shift 2 ;;
    --env-place-col) ENV_PLACE_COL="$2"; shift 2 ;;
    --env-start-col) ENV_START_COL="$2"; shift 2 ;;
    --env-end-col) ENV_END_COL="$2"; shift 2 ;;
    --env-season-start-mm-dd) ENV_SEASON_START_MM_DD="$2"; shift 2 ;;
    --env-season-end-mm-dd) ENV_SEASON_END_MM_DD="$2"; shift 2 ;;
    --env-season-length-days) ENV_SEASON_LENGTH_DAYS="$2"; shift 2 ;;
    --env-windows) ENV_WINDOWS="$2"; shift 2 ;;
    --env-align) ENV_ALIGN="$2"; shift 2 ;;
    --env-vars) ENV_VARS="$2"; shift 2 ;;
    --env-workers) ENV_WORKERS="$2"; shift 2 ;;

    --gene-gff) GENE_GFF="$2"; shift 2 ;;
    --te-annotation) TE_ANNOTATION="$2"; shift 2 ;;
    --te-gff) TE_GFF="$2"; shift 2 ;;
    --homoeolog-pairs) HOMOEOLOG_PAIR_FILE="$2"; shift 2 ;;
    --color-mode) COLOR_MODE="$2"; shift 2 ;;

    --skip-admixture) RUN_ADMIXTURE=0; shift ;;
    --admixture-k) ADMIXTURE_K="$2"; shift 2 ;;
    --admixture-k-range) ADMIXTURE_K_RANGE="$2"; shift 2 ;;
    --admixture-selection-method) ADMIXTURE_SELECTION_METHOD="$2"; shift 2 ;;

    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --cv-folds) CV_FOLDS="$2"; shift 2 ;;

    --plink) PLINK_BIN="$2"; shift 2 ;;
    --plink2) PLINK2_BIN="$2"; shift 2 ;;
    --tabix) TABIX_BIN="$2"; shift 2 ;;
    --bcftools) BCFTOOLS_BIN="$2"; shift 2 ;;
    --beagle-jar) BEAGLE_JAR="$2"; shift 2 ;;
    --admixture-bin) ADMIXTURE_BIN="$2"; shift 2 ;;
    --samtools) SAMTOOLS_BIN="$2"; shift 2 ;;

    *) die "Unknown option: $1" ;;
  esac
done

MODE="${MODE}"
case "$MODE" in
  diploid|polyploid) ;;
  *) die "--mode must be diploid or polyploid" ;;
esac

STAGES="${STAGES}"
if [[ -z "$ENV_WINDOWS" ]]; then
  if [[ "$MODE" == "polyploid" ]]; then
    ENV_WINDOWS=16
  else
    ENV_WINDOWS=32
  fi
fi
TARGET_COL="${TARGET_COL:-$TRAIT_COL}"
[[ -n "$TARGET_COL" ]] || [[ "$STAGES" != "all" && "$STAGES" != *train* && "$STAGES" != *phenotype* ]] || die "--trait-col is required when phenotype or train stages are enabled."

WORKDIR="$(abspath "$WORKDIR")"
mkdir -p "$WORKDIR"
GENO_DIR="$WORKDIR/00_genotype"
GENO_IMPUTE_DIR="$GENO_DIR/imputation"
GENO_IMPUTE_MARKER="$GENO_IMPUTE_DIR/beagle_imputation.complete"
ADMIX_DIR="$WORKDIR/01_admixture"
PHENO_DIR="$WORKDIR/02_phenotype"
ENV_DIR="$WORKDIR/03_environment"
TENSOR_DIR_OUT="$WORKDIR/04_tensors"
TRAIN_DIR="$WORKDIR/05_train/$MODE"
TMP_DIR="$WORKDIR/tmp"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$TMP_DIR" "$LOG_DIR"

GENOTYPE_SOURCE_TYPE="$(normalize_source_type_alias "$GENOTYPE_SOURCE_TYPE")"
if [[ -n "$GENOTYPE_SOURCE" ]]; then GENOTYPE_SOURCE="$(abspath "$GENOTYPE_SOURCE")"; fi
if [[ -n "$AXIOM_FILE" ]]; then AXIOM_FILE="$(abspath "$AXIOM_FILE")"; fi
if [[ -n "$EXISTING_PLINK_PREFIX" ]]; then EXISTING_PLINK_PREFIX="$(abspath "$EXISTING_PLINK_PREFIX")"; fi
if [[ -n "$EXISTING_PFILE_PREFIX" ]]; then EXISTING_PFILE_PREFIX="$(abspath "$EXISTING_PFILE_PREFIX")"; fi
if [[ -n "$VCF_SOURCE" ]]; then VCF_SOURCE="$(abspath "$VCF_SOURCE")"; fi
if [[ -n "$EXISTING_VCF" ]]; then EXISTING_VCF="$(abspath "$EXISTING_VCF")"; fi
if [[ -n "$EXISTING_HAP_BLOCK_FILE" ]]; then EXISTING_HAP_BLOCK_FILE="$(abspath "$EXISTING_HAP_BLOCK_FILE")"; fi
if [[ -n "$FASTA" ]]; then FASTA="$(abspath "$FASTA")"; fi
if [[ -n "$TRAIT_FILE" ]]; then TRAIT_FILE="$(abspath "$TRAIT_FILE")"; fi
if [[ -n "$TRIAL_FILE" ]]; then TRIAL_FILE="$(abspath "$TRIAL_FILE")"; fi
if [[ -n "$ENV_SOURCE_FILE" ]]; then ENV_SOURCE_FILE="$(abspath "$ENV_SOURCE_FILE")"; fi
if [[ -n "$GENE_GFF" ]]; then GENE_GFF="$(abspath "$GENE_GFF")"; fi
if [[ -n "$TE_ANNOTATION" ]]; then TE_ANNOTATION="$(abspath "$TE_ANNOTATION")"; fi
if [[ -n "$TE_GFF" ]]; then TE_GFF="$(abspath "$TE_GFF")"; fi
if [[ -n "$HOMOEOLOG_PAIR_FILE" ]]; then HOMOEOLOG_PAIR_FILE="$(abspath "$HOMOEOLOG_PAIR_FILE")"; fi
if [[ -n "$BEAGLE_JAR" && "$BEAGLE_JAR" == */* ]]; then BEAGLE_JAR="$(abspath "$BEAGLE_JAR")"; fi

TRAIT_SLUG="$($PYTHON_BIN - <<'PY' "$TARGET_COL"
import re, sys
value = re.sub(r'[^A-Za-z0-9_]+', '_', sys.argv[1].strip())
value = re.sub(r'_+', '_', value).strip('_')
print(value.lower() or 'trait')
PY
)"
MODEL_METADATA_BASE="$PHENO_DIR/metadata_${TRAIT_SLUG}.csv"
PHENO_MEAN_BASE="$PHENO_DIR/${TRAIT_SLUG}_mean.txt"
MODEL_METADATA="$MODEL_METADATA_BASE"
ENV_MATRIX="$ENV_DIR/env_matrix.csv"
ENV_META="$ENV_DIR/env_meta.json"
ENV_CACHE_DIR="$ENV_DIR/.env_cache"
GEOCODE_CACHE="$ENV_DIR/.geocode_cache.json"

FINAL_PLINK_PREFIX=""
FINAL_VCF=""
FINAL_PED=""
FINAL_MAP=""
HAP_BLOCK_FILE=""

expand_stage_list

log "EcoPopDL-GP launcher"
log "Mode: $MODE"
log "Stages: $STAGES"
log "Workdir: $WORKDIR"

case "$COLOR_MODE" in
  dosage|allele_combination) ;;
  *) die "--color-mode must be dosage or allele_combination" ;;
esac

if [[ "$MODE" == "polyploid" ]]; then
  if stage_enabled tensors; then
    [[ -n "$HOMOEOLOG_PAIR_FILE" ]] || die "--homoeolog-pairs is required for polyploid tensor generation."
  fi
fi

run_genotype_stage
run_phenotype_stage
run_admixture_stage
run_environment_stage
run_tensor_stage
run_training_stage

cat <<EOF

Pipeline finished.

Key outputs:
  Final genotype prefix : ${FINAL_PLINK_PREFIX:-N/A}
  Final VCF            : ${FINAL_VCF:-N/A}
  Phenotype metadata   : ${MODEL_METADATA:-N/A}
  Environment matrix   : ${ENV_MATRIX:-N/A}
  Tensor directory     : ${TENSOR_DIR_OUT:-N/A}/tensors
  Training directory   : ${TRAIN_DIR:-N/A}
EOF
