import logging
import numpy as np, math, os, pandas as pd, tempfile, zipfile
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import random
import glob
from collections import Counter
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.txt"),
        logging.StreamHandler()
    ]
)

_LOGGED_OPTIONAL_CHANNEL_GROUPS = set()


def _log_optional_channel_group_once(exporter: str, group: str, channels: list[str], reason: str) -> None:
    key = (exporter, group)
    if key in _LOGGED_OPTIONAL_CHANNEL_GROUPS:
        return
    _LOGGED_OPTIONAL_CHANNEL_GROUPS.add(key)
    if channels:
        logging.info(f"[{exporter}] Added {group} channels: {', '.join(channels)}")
    else:
        logging.warning(f"[{exporter}] {group} channels were not added. {reason}")

# Atomic NPZ writer to avoid partially written tiles when readers are active
def safe_save_npz(path: str, **arrays):
    """
    Write NPZ atomically: save to a temp in the same directory, fsync, then replace.
    Prevents readers from seeing half-written files.
    """
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_size = os.path.getsize(tmp)
        if tmp_size <= 0:
            raise RuntimeError(f"np.savez_compressed produced zero bytes for temp file: {tmp}")
        os.replace(tmp, path)
        final_size = os.path.getsize(path)
        if final_size <= 0:
            logging.warning(f"Target NPZ is zero bytes after replace: {path}")
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

# Save NPZ and immediately verify integrity (zip + np.load); retry if needed.
def write_and_verify_npz(path: str, retries: int = 1, **arrays):
    for attempt in range(retries + 1):
        safe_save_npz(path, **arrays)
        try:
            size = os.path.getsize(path)
            if size <= 0:
                raise RuntimeError("zero-byte NPZ after save")
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"zip error at {bad}")
            # make sure numpy can read all members
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    _ = z[key].shape
            return
        except Exception as e:
            logging.warning(f"NPZ verify failed (attempt {attempt+1}/{retries+1}) for {path}: {e}")
            try:
                os.remove(path)
            except OSError:
                pass
            if attempt >= retries:
                raise

# Global variables

# Chickpea
# chr_info = {
#     '1': 48360000,
#     '2': 36630000,
#     '3': 39990000,
#     '4': 49190000,
#     '5': 48170000,
#     '6': 59460000,
#     '7': 48960000,
#     '8': 16480000
# }
# Rice (12 chromosomes)
# chr_info = {
#     '1': 45050000,
#     '2': 36780000,
#     '3': 37370000,
#     '4': 36150000,
#     '5': 30000000,
#     '6': 31600000,
#     '7': 30280000,
#     '8': 28570000,
#     '9': 30530000,
#     '10': 23960000,
#     '11': 30760000,
#     '12': 27770000
# }

# B. napus D4 chromosome lengths inferred from the actual MAP + TE + GFF inputs.
chr_info = {
    '1': 50462719,
    '2': 63736844,
    '3': 58438715,
    '4': 68218497,
    '5': 60570340,
    '6': 59681604,
    '7': 51041440,
    '8': 73190948,
    '9': 72814788,
    '10': 63378997,
    '11': 56714362,
    '12': 65577383,
    '13': 78504609,
    '14': 69431502,
    '15': 56309783,
    '16': 49621279,
    '17': 61103310,
    '18': 53717154,
    '19': 64999797,
}

USE_SUBGENOMES = True  # <-- flip to True for A/C, False for diploid

# # Add near the top with other globals
SUBGENOME_LABELS = {'A': 1.0, 'C': 0.0}

# # ---- Subgenome config (B. napus example) ----
NUM_TO_LABEL = {
    '1':'A01','2':'A02','3':'A03','4':'A04','5':'A05',
    '6':'A06','7':'A07','8':'A08','9':'A09','10':'A10',
    '11':'C01','12':'C02','13':'C03','14':'C04','15':'C05',
    '16':'C06','17':'C07','18':'C08','19':'C09'
}
SUBGENOME_CHRS = {
    'A': [f'A{str(i).zfill(2)}' for i in range(1, 11)],
    'C': [f'C{str(i).zfill(2)}' for i in range(1, 10)],
}
# If you're in SG mode, NUM_TO_LABEL must cover all numeric chr IDs.
# NUM_TO_LABEL = {'1':'A01', ... '19':'C09'}  # <-- ensure this is defined in SG mode

# Fail fast when SG mapping is required
def assert_subgenome_config_valid():
    if USE_SUBGENOMES:
        if 'NUM_TO_LABEL' not in globals() or not NUM_TO_LABEL:
            raise ValueError("In subgenome mode you must define NUM_TO_LABEL.")
        if not (SUBGENOME_CHRS and all(isinstance(v, list) and len(v) for v in SUBGENOME_CHRS.values())):
            raise ValueError("Define SUBGENOME_CHRS = {'A':[...], 'C':[...]} (or A/B/D, etc.).")

WRITE_FEATURE_TILES = True   # set False if you only want the color PNG/NPZ you already had
INCLUDE_SG_CHANNEL  = False   # adds the 'is_A' channel as Ch3/Ch4
INCLUDE_HOMOLOGY = USE_SUBGENOMES  # diploid -> False, polyploid -> True (override if needed)
HOMOEOLOG_PAIR_FILE = os.environ.get(
    "HOMOEOLOG_PAIR_FILE",
    "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/Chromomap/Bnapus/Bnapus_A_C_homoeolog_pairs.tsv",
)
HOM_HASH_K = int(os.environ.get("HOM_HASH_K", "32"))
HOMOEOLOG_ANCHOR_WINDOW_BP = int(os.environ.get("HOMOEOLOG_ANCHOR_WINDOW_BP", "1000000"))

PLOIDY = 2  # set 2 for diploid datasets; if you ever render polyploids, change accordingly
# Quality channel choice for SNP tiles:
#   'maf'       -> MAF (compute on train split only to avoid leakage)
#   'callrate'  -> fraction of called genotypes per SNP (default; leak-safe)
#   'missing'   -> 1 - callrate
SNP_QUALITY_CHANNEL = 'callrate'  # 'maf'|'callrate'|'missing'

# Include a per-row local SNP density channel
INCLUDE_DENSITY_CHANNEL = True

# TE annotation input/output (set to None to skip).
# run.sh fills this in from --te-annotation or --te-gff.
TE_GENE_ANNOTATION_TSV = None
SNP_TE_ANNOTATION_OUT = "snps_with_te_annotation.tsv"

# Gene annotation input/output (set to None to skip)
GENE_GFF_PATH = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Bnapus_3DH.genes_20211001.gff3"
SNP_GENE_ANNOTATION_OUT = "snps_with_gene_annotation.tsv"
PROMOTER_BP = 2000

# Optional hashed block-identity channels K (0 disables)
HASH_BLOCK_ID_K = 32  # e.g., 8 or 16 if you want hashed ID channels

# Positional encoding width for hierarchical tensors
POSITION_ENCODING_DIM = 32

# Tile/global sizing defaults (used by both SNP and haplo writers)
DESIRED_MAX_WIDTH_PX = 800_000
COLLISION_DEPTH = 64  # allow deeper stacking to eliminate overflow without big cost

# --- NEW: local genotype context channels for hierarchical tensor ---
INCLUDE_TOKEN_RANK_NORM = True
INCLUDE_LOCAL_DOSAGE_CONTEXT = True
LOCAL_DOSAGE_WINDOW = 11   # odd number recommended (e.g., 11, 21)


# Color mode controls whether SNP tiles are colored by dosage or by allele combinations.
# Set the `CHROMOMAP_COLOR_MODE` env var or edit the default below if you need the alternate palette.
DOSAGE_COLOR_MAP = {
    'dos0': (86, 180, 233),  # sky blue
    'dos1': (230, 159, 0),   # orange
    'dos2': (0, 0, 0),       # black
    '-1':   (255, 255, 255)  # white
}

ALLELE_COMBINATION_COLOR_MAP = {
    '00': (0, 0, 0), # black
    '11': (255, 0, 0), # red
    '22': (0, 255, 0), # green
    '33': (0, 0, 255), # blue
    '01': (0, 0, 128), # navy blue
    '10': (0, 0, 128), # navy blue
    '02': (0, 128, 0),
    '20': (0, 128, 0),
    '03': (128, 0, 0),
    '30': (128, 0, 0),
    '12': (255, 255, 0),
    '21': (255, 255, 0),
    '13': (255, 0, 255),
    '31': (255, 0, 255),
    '23': (0, 255, 255),
    '32': (0, 255, 255),
    '-1': (200, 200, 200) # light gray for missing
}

CHROMOMAP_COLOR_MODE = os.environ.get('CHROMOMAP_COLOR_MODE')
VALID_COLOR_MODES = {'dosage', 'allele_combination'}

def _normalize_color_mode(mode: str) -> str:
    """Lowercase and validate a requested chromomap color mode."""
    value = (mode or '').strip().lower()
    if not value:
        value = 'dosage'
    if value not in VALID_COLOR_MODES:
        raise ValueError(f"Unsupported CHROMOMAP_COLOR_MODE={mode!r}; valid options: {sorted(VALID_COLOR_MODES)}")
    return value


COLOR_MODE = 'allele_combination'  # file default when no env/CLI override
color_map = None

def set_color_mode(mode: str):
    """Update the global mode and color map (used by CLI args or env defaults)."""
    global COLOR_MODE, color_map
    COLOR_MODE = _normalize_color_mode(mode)
    color_map = ALLELE_COMBINATION_COLOR_MAP if COLOR_MODE == 'allele_combination' else DOSAGE_COLOR_MAP


initial_color_mode = CHROMOMAP_COLOR_MODE or COLOR_MODE  # env wins, else file default
set_color_mode(initial_color_mode)


# --- MODIFY canonical_pair so it recognizes 0/1/2 (or GT strings) ---
MISSING_TOKENS = {"-1", "99", "", ".", "./.", ".|.", "nan", "NaN", "None"}
PED_MISSING_ALLELES = {"0", "N", "n", "-", ".", "?", "NA", ""}

_LOGGED_BAD_ALLELE_CODES = set()


def _log_unrecognized_allele_code(token: str):
    """Warn once per unexpected allele token so we don't silently drop them."""
    if token in _LOGGED_BAD_ALLELE_CODES:
        return
    _LOGGED_BAD_ALLELE_CODES.add(token)
    logging.warning(f"Unrecognized allele code {token!r} in allele_combination mode; treated as missing.")


def canonical_pair(code: str) -> str:
    if COLOR_MODE == 'allele_combination':
        return canonical_pair_allele(code)
    return canonical_pair_dosage(code)


def canonical_pair_dosage(code: str) -> str:
    s = str(code).strip()
    if s in MISSING_TOKENS:
        return "-1"

    try:
        x = float(s)
        if 0.0 <= x <= 1.0:
            x *= float(PLOIDY)
        if 0.0 <= x <= float(PLOIDY):
            return f"dos{int(np.rint(x))}"
    except Exception:
        pass

    m = re.match(r'^([0-9]+)[/|]([0-9]+)$', s)
    if m:
        a, b = m.groups()
        if a == '.' or b == '.':
            return "-1"
        alt = (int(a) != 0) + (int(b) != 0)
        return f"dos{min(PLOIDY, max(0, alt))}"

    if re.fullmatch(r'\d{2}', s):
        a, b = s[0], s[1]
        alt = (a != '0') + (b != '0')
        return f"dos{min(PLOIDY, alt)}"

    return "-1"


def canonical_pair_allele(code: str) -> str:
    s = str(code).strip()
    if s in MISSING_TOKENS:
        return "-1"
    if re.fullmatch(r'\d{2}', s) and all(ch in '0123' for ch in s):
        return "".join(sorted(s))
    _log_unrecognized_allele_code(s)
    return "-1"


# def canonical_pair(code: str) -> str:
#     s = str(code).strip()
#     if s in {"-1","99","","nan","NaN","None"}: return "-1"
#     if re.fullmatch(r'-?\d+(\.0+)?', s):
#         try: s = f"{int(float(s)):02d}"
#         except: return "-1"
#     if not re.fullmatch(r'\d{2}', s): return "-1"
#     a,b = sorted(s)
#     return a + b

def _normalize_chr_label_to_digits(label: str) -> str:
    parsed = _extract_chromosome_id(label)
    if parsed is not None:
        return parsed
    s = ''.join(ch for ch in str(label) if ch.isdigit())
    return str(int(s)) if s else str(label)

def _normalize_numeric_chr_token(token: str) -> str | None:
    """
    Normalize a single chromosome token into canonical digits:
    - '1', '01' -> '1'
    - 'Ca8', 'chr08', 'chromosome_8' -> '8'
    - 'N1', 'N01' -> '1'   (B. napus reference naming)
    """
    t = str(token).strip()
    if not t:
        return None
    m = re.fullmatch(r"0*([0-9]+)", t)
    if m:
        return str(int(m.group(1)))
    m = re.fullmatch(r"(?i)(?:chr|chromosome|ca|n)[_-]*0*([0-9]+)", t)
    if m:
        return str(int(m.group(1)))
    return None


def _extract_chromosome_id(label: str) -> str | None:
    """
    Extract canonical chromosome digits from labels like:
    - 'Ca1', 'chr1', '1'
    - 'cicar.CDCFrontier.gnm1.Ca8'
    Returns None for scaffold/contig identifiers to avoid false joins.
    """
    s = str(label).strip()
    if not s:
        return None

    direct = _normalize_numeric_chr_token(s)
    if direct is not None:
        return direct

    parts = re.split(r"[._:/\\-]+", s)
    for tok in reversed(parts):
        norm = _normalize_numeric_chr_token(tok)
        if norm is not None:
            return norm
    return None


def _extract_subgenome_chr_label(
    label: str,
    allowed_letters: set[str] | None = None,
) -> str | None:
    """
    Preserve subgenome-labelled chromosomes like A01/C01 or chrA01/chrC01.
    Returns None for labels that do not match a strict subgenome token.
    """
    s = str(label).strip()
    if not s:
        return None

    allowed = {str(x).upper() for x in allowed_letters} if allowed_letters else set()
    tokens = [s]
    tokens.extend(tok for tok in re.split(r"[._:/\\-]+", s) if tok)
    for tok in tokens:
        m = re.fullmatch(r"(?i)(?:chr|chromosome)?[_-]*([A-Za-z])[_-]*0*([0-9]+)", tok.strip())
        if not m:
            continue
        letter = m.group(1).upper()
        if allowed and letter not in allowed:
            continue
        return f"{letter}{int(m.group(2)):02d}"
    return None


def normalize_te_chr_label(label: str) -> str | None:
    """
    Normalize TE chromosome labels into canonical digits ('N1' -> '1', 'Ca8' -> '8').
    Non-chromosome scaffold labels return None.
    """
    return _extract_chromosome_id(label)

def normalize_gene_chr_label(label: str) -> str | None:
    """
    Normalize gene seqids by extracting main chromosome ids from labels like N1/chr1/Ca8/1.
    Returns None for non-chromosome contigs to avoid false overlaps.
    """
    return _extract_chromosome_id(label)


def normalize_chr_for_mode(label: str, id_normalizer) -> str | None:
    """
    Normalize chromosome labels to the current export mode.
    In subgenome mode preserve explicit A/C-style labels; otherwise map numeric-like
    labels such as N1/chr1/1 through the active id normalizer.
    """
    if USE_SUBGENOMES:
        sg = _extract_subgenome_chr_label(label, set(SUBGENOME_CHRS.keys()))
        if sg is not None:
            return sg
    base = _extract_chromosome_id(label)
    if base is None:
        return None
    return id_normalizer(base) if id_normalizer is not None else base


def normalize_chr_id(x: str) -> str:
    x = str(x)
    return NUM_TO_LABEL.get(x, x)  # if already 'A01'/'C01', this is a no-op

def id_norm_sg(x: str) -> str:
    """Subgenome-aware normalization 1..19 -> A01..C09 (B. napus map)."""
    s = _extract_chromosome_id(str(x)) or str(x)
    return NUM_TO_LABEL.get(s, s)

def id_norm_identity(x: str) -> str:
    """Diploid: canonicalize chromosome labels to plain digits when possible."""
    return str(_normalize_chr_label_to_digits(x))

def build_chr_info_for_mode():
    """
    Returns (local_chr_info, id_normalizer) depending on mode.
    - Subgenome mode: keys like A01..A10, C01..C09 (checks config)
    - Diploid mode:   keys as strings '1','2',... from chr_info
    """
    if USE_SUBGENOMES:
        assert_subgenome_config_valid()
        local = {
            id_norm_sg(k): v
            for k, v in chr_info.items()
            if id_norm_sg(k) is not None
        }
        return local, id_norm_sg
    else:
        local = {_normalize_chr_label_to_digits(k): v for k, v in chr_info.items()}
        return local, id_norm_identity

def _te_region_priority(value: str) -> int:
    rank = {"genic": 3, "promoter": 2, "intergenic": 1}
    return rank.get(str(value).strip().lower(), 0)

def _clean_te_token(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or v.lower() in {"none", "nan", "na", "-1"}:
        return None
    return v


def _collect_point_interval_overlaps(
    snps_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    payload_cols: list[str],
) -> pd.DataFrame:
    if snps_df.empty or intervals_df.empty:
        return pd.DataFrame(columns=["snp_id", *payload_cols])

    rows = []
    for chrom, snps_chr in snps_df.groupby("chr_norm", sort=False):
        intervals_chr = intervals_df[intervals_df["chr_norm"] == chrom]
        if intervals_chr.empty:
            continue

        intervals_chr = intervals_chr.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)
        starts = intervals_chr["start"].astype(int).to_numpy()
        ends = intervals_chr["end"].astype(int).to_numpy()
        payload = {col: intervals_chr[col].to_numpy(dtype=object) for col in payload_cols}

        active = []
        next_idx = 0
        points = snps_chr[["snp_id", "pos"]].copy()
        points["pos"] = points["pos"].astype(int)
        points = points.sort_values("pos", kind="mergesort")

        for snp_id, pos in zip(points["snp_id"].astype(str), points["pos"].to_numpy()):
            while next_idx < len(intervals_chr) and starts[next_idx] <= pos:
                active.append(next_idx)
                next_idx += 1
            if active:
                active = [idx for idx in active if ends[idx] >= pos]
            if not active:
                continue
            for idx in active:
                row = {"snp_id": snp_id}
                for col in payload_cols:
                    row[col] = payload[col][idx]
                rows.append(row)

    return pd.DataFrame(rows, columns=["snp_id", *payload_cols])


def annotate_snps_with_te(
    snps_df: pd.DataFrame,
    te_annotation_path: str | None,
    output_path: str | None = None,
    id_normalizer=normalize_chr_id,
) -> pd.DataFrame:
    if not te_annotation_path:
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out
    if not os.path.exists(te_annotation_path):
        logging.warning(f"TE annotation file not found: {te_annotation_path}")
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out

    pr = None
    try:
        import pyranges as pr
    except Exception as e:
        logging.info(f"PyRanges not available for TE annotation; using Python fallback: {e}")

    te = pd.read_csv(te_annotation_path, sep="\t")
    required = {"chr", "start", "end", "ID", "region", "gene_id"}
    missing = required - set(te.columns)
    if missing:
        logging.warning(f"TE annotation missing columns {missing}; skipping TE join.")
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out

    te = te.copy()
    te["chr"] = te["chr"].astype(str)
    if id_normalizer is not None:
        te["chr_norm"] = te["chr"].map(id_normalizer)
    else:
        te["chr_norm"] = te["chr"]
    te = te[te["chr_norm"].notna()].copy()
    if te.empty:
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out
    te["chr_norm"] = te["chr_norm"].astype(str)
    te["start0"] = (te["start"].astype(int) - 1).clip(lower=0)
    te["end0"] = te["end"].astype(int)

    snps = snps_df.copy()
    snps["chr"] = snps["chr"].astype(str)
    if id_normalizer is not None:
        snps["chr_norm"] = snps["chr"].map(id_normalizer)
    else:
        snps["chr_norm"] = snps["chr"]
    snps = snps[snps["chr_norm"].notna()].copy()
    if snps.empty:
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out
    snps["chr_norm"] = snps["chr_norm"].astype(str)
    snps["start0"] = (snps["pos"].astype(int) - 1).clip(lower=0)
    snps["end0"] = snps["pos"].astype(int)

    if pr is not None:
        gr_snps = pr.PyRanges(
            snps[["chr_norm", "start0", "end0", "snp_id"]].rename(
                columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
            )
        )
        gr_te = pr.PyRanges(
            te[["chr_norm", "start0", "end0", "ID", "region", "gene_id"]].rename(
                columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End", "ID": "te_id"}
            )
        )

        joined = gr_snps.join(gr_te, how="left").df
    else:
        te_join = te[["chr_norm", "start", "end", "ID", "region", "gene_id"]].rename(columns={"ID": "te_id"})
        joined = _collect_point_interval_overlaps(snps, te_join, ["te_id", "region", "gene_id"])
    if joined.empty:
        out = snps_df.copy()
        out["is_TE"] = False
        out["TE_region"] = "none"
        out["TE_gene"] = "none"
        return out

    grouped = joined.groupby("snp_id", sort=False)

    is_te = grouped["te_id"].apply(lambda s: s.notna().any())
    region = grouped["region"].apply(
        lambda s: max(
            (r for r in (_clean_te_token(v) for v in s) if r is not None),
            default=None,
            key=_te_region_priority,
        )
    )
    gene = grouped["gene_id"].apply(
        lambda s: ";".join(sorted({v for v in (_clean_te_token(x) for x in s) if v is not None}))
    )

    out = snps_df.copy()
    out["is_TE"] = out["snp_id"].map(is_te).fillna(False).astype(bool)
    out["TE_region"] = out["snp_id"].map(region).fillna("none")
    out["TE_gene"] = out["snp_id"].map(gene)
    out["TE_gene"] = out["TE_gene"].replace("", "none").fillna("none")

    if output_path:
        out.to_csv(output_path, sep="\t", index=False)
        logging.info(f"Wrote SNP TE annotations: {output_path}")

    return out

GFF_COLUMNS = [
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attr",
]

def _read_gff(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=GFF_COLUMNS,
        low_memory=False,
        dtype={
            "seqid": "string",
            "source": "string",
            "type": "string",
            "score": "string",
            "strand": "string",
            "phase": "string",
            "attr": "string",
        },
    )
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["seqid", "start", "end"])
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def _extract_gff_attr(attr: str, keys: list[str]) -> str | None:
    if not isinstance(attr, str):
        return None
    for field in attr.split(";"):
        if not field or "=" not in field:
            continue
        k, v = field.split("=", 1)
        if k in keys and v:
            return v
    return None


def _extract_gene_id(attr: str, fallback: str) -> str:
    return _extract_gff_attr(attr, ["ID", "gene_id", "Name"]) or fallback


def normalize_gene_id_token(x: str) -> str:
    """Normalize gene IDs across GFF and homoeolog pair tables."""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "na"}:
        return ""
    s = re.sub(r"\.\d+$", "", s)
    s = s.replace("gene:", "").replace("Gene:", "")
    return s


def load_gene_gff(
    gff_path: str,
    gene_types: list[str] | None = None,
    id_normalizer=normalize_gene_chr_label,
) -> pd.DataFrame:
    gene_types = gene_types or ["gene"]
    df = _read_gff(gff_path)
    df = df[df["type"].isin(gene_types)].copy()
    if df.empty:
        return df
    df["gene_id"] = df.apply(
        lambda r: _extract_gene_id(
            r["attr"], f"{r['seqid']}:{r['start']}-{r['end']}"
        ),
        axis=1,
    )
    df["chr"] = df["seqid"].astype(str)
    if id_normalizer:
        df["chr_norm"] = df["chr"].map(id_normalizer)
    else:
        df["chr_norm"] = df["chr"]
    df = df[df["chr_norm"].notna()]
    df["gene_id_norm"] = df["gene_id"].map(normalize_gene_id_token)
    df["start0"] = (df["start"] - 1).clip(lower=0)
    df["end0"] = df["end"]
    return df[["chr", "chr_norm", "start", "end", "start0", "end0", "strand", "gene_id", "gene_id_norm"]]


def load_homoeolog_pairs(path: str) -> pd.DataFrame:
    """
    Load a B. napus homoeolog pair table. The first two columns are treated as
    gene identifiers after normalization, regardless of separator or header.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Homoeolog pair file not found: {path}")

    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")

    if df.shape[1] < 2:
        raise ValueError("Homoeolog pair file must have at least 2 columns (gene1, gene2).")

    df = df.iloc[:, :2].copy()
    df.columns = ["gene1", "gene2"]
    df["gene1"] = df["gene1"].map(normalize_gene_id_token)
    df["gene2"] = df["gene2"].map(normalize_gene_id_token)

    header_tokens = {
        "gene1", "gene2", "gene_1", "gene_2", "genea", "genec",
        "query", "subject", "query_gene", "subject_gene",
        "id1", "id2", "geneid1", "geneid2", "gene_id1", "gene_id2",
        "a_gene", "c_gene", "agene", "cgene",
    }
    is_header = df["gene1"].str.lower().isin(header_tokens) & df["gene2"].str.lower().isin(header_tokens)
    df = df[~is_header]
    df = df[(df["gene1"] != "") & (df["gene2"] != "")]
    df = df[df["gene1"] != df["gene2"]]
    df = df.drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Homoeolog pair file has no usable gene pairs after normalization: {path}")
    return df


def build_homoeolog_groups(pairs: pd.DataFrame) -> tuple[dict[str, int], dict[int, int]]:
    """Build connected homoeolog groups from pairwise edges via union-find."""
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        rank.setdefault(x, 0)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in pairs[["gene1", "gene2"]].itertuples(index=False):
        union(a, b)

    comps = defaultdict(list)
    genes = set(pairs["gene1"]).union(set(pairs["gene2"]))
    for gene in genes:
        comps[find(gene)].append(gene)

    gene_to_gid: dict[str, int] = {}
    gid_to_size: dict[int, int] = {}
    components = []
    for members in comps.values():
        members_sorted = sorted(set(members))
        rep = members_sorted[0]
        components.append((rep, members_sorted))
    components.sort(key=lambda item: (-len(item[1]), item[0]))
    for gid, (_, members) in enumerate(components, start=1):
        for gene in members:
            gene_to_gid[gene] = gid
        gid_to_size[gid] = len(members)

    return gene_to_gid, gid_to_size


def map_snp_geneid_to_homology(
    snp_gene_id: pd.Series,
    gene_to_gid: dict[str, int],
    gid_to_size: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map per-SNP gene annotations to homoeolog groups.
    Returns:
      hom_gid (int32, 0 means none),
      hom_has (float32 0/1),
      hom_size_norm (float32 0..1)
    """
    n = len(snp_gene_id)
    hom_gid = np.zeros(n, dtype=np.int32)

    for i, raw in enumerate(snp_gene_id.astype(str).values):
        toks = [normalize_gene_id_token(tok) for tok in raw.split(";")]
        toks = [tok for tok in toks if tok]
        if not toks:
            continue
        gid = 0
        best_size = -1
        for gene in toks:
            cand_gid = gene_to_gid.get(gene, 0)
            if not cand_gid:
                continue
            cand_size = int(gid_to_size.get(cand_gid, 0))
            if cand_size > best_size or (cand_size == best_size and (gid == 0 or cand_gid < gid)):
                gid = int(cand_gid)
                best_size = cand_size
        hom_gid[i] = int(gid)

    hom_has = (hom_gid > 0).astype(np.float32)
    sizes = np.array([gid_to_size.get(int(gid), 0) for gid in hom_gid], dtype=np.float32)
    max_size = float(max(gid_to_size.values())) if gid_to_size else 1.0
    hom_size_norm = (np.log1p(sizes) / np.log1p(max_size)).astype(np.float32)
    hom_size_norm[hom_gid == 0] = 0.0
    return hom_gid, hom_has, hom_size_norm


def build_homology_spans_df(
    genes_df: pd.DataFrame | None,
    pairs: pd.DataFrame | None,
    gene_to_gid: dict[str, int],
) -> pd.DataFrame:
    """
    Build a stable interval table for homology span rasterization.

    Each pair edge contributes up to two rows, one per gene span, both sharing the
    same deterministic `hom_pair_id` and connected-component `hom_group_id`.
    """
    cols = ["Chromosome", "BP1", "BP2", "hom_group_id", "hom_pair_id", "gene_id", "partner_gene_id"]
    if genes_df is None or genes_df.empty or pairs is None or pairs.empty or not gene_to_gid:
        return pd.DataFrame(columns=cols)

    genes = genes_df.copy()
    if "gene_id_norm" not in genes.columns:
        genes["gene_id_norm"] = genes["gene_id"].map(normalize_gene_id_token)
    genes = genes[(genes["gene_id_norm"] != "") & genes["chr_norm"].notna()].copy()
    if genes.empty:
        return pd.DataFrame(columns=cols)

    genes = genes.sort_values(["gene_id_norm", "chr_norm", "start", "end"]).drop_duplicates(
        subset=["gene_id_norm"], keep="first"
    )
    gene_coord = genes.set_index("gene_id_norm")[["chr_norm", "start", "end"]].to_dict("index")

    pair_items: list[tuple[int, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for gene1, gene2 in pairs[["gene1", "gene2"]].itertuples(index=False):
        a = normalize_gene_id_token(gene1)
        b = normalize_gene_id_token(gene2)
        if not a or not b or a == b:
            continue
        left, right = sorted((a, b))
        key = (left, right)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        gids = {int(g) for g in (gene_to_gid.get(left, 0), gene_to_gid.get(right, 0)) if int(g) > 0}
        if not gids:
            continue
        pair_items.append((min(gids), left, right))

    pair_items.sort(key=lambda item: (item[0], item[1], item[2]))

    rows = []
    skipped_pairs = 0
    for hom_pair_id, (hom_group_id, left, right) in enumerate(pair_items, start=1):
        left_coord = gene_coord.get(left)
        right_coord = gene_coord.get(right)
        if left_coord is None or right_coord is None:
            skipped_pairs += 1
            continue
        rows.append((
            str(left_coord["chr_norm"]),
            int(left_coord["start"]),
            int(left_coord["end"]),
            int(hom_group_id),
            int(hom_pair_id),
            left,
            right,
        ))
        rows.append((
            str(right_coord["chr_norm"]),
            int(right_coord["start"]),
            int(right_coord["end"]),
            int(hom_group_id),
            int(hom_pair_id),
            right,
            left,
        ))

    out = pd.DataFrame(rows, columns=cols)
    logging.info(
        f"[HOMOLOGY] homology spans: rows={len(out)} pairs={len(out) // 2} skipped_pairs={skipped_pairs}"
    )
    return out


def compute_homoeolog_anchor_density(
    genes_df: pd.DataFrame | None,
    gene_to_gid: dict[str, int],
    snp_chr_norm: np.ndarray,
    snp_pos: np.ndarray,
    window_bp: int = HOMOEOLOG_ANCHOR_WINDOW_BP,
) -> np.ndarray:
    """
    Windowed density of homoeolog-anchor genes around each SNP, normalized to [0,1].
    Anchor positions use gene midpoints and a fixed genomic window.
    """
    out = np.zeros(len(snp_pos), dtype=np.float32)
    if genes_df is None or genes_df.empty or not gene_to_gid or len(snp_pos) == 0:
        return out

    genes = genes_df.copy()
    if "gene_id_norm" not in genes.columns:
        genes["gene_id_norm"] = genes["gene_id"].map(normalize_gene_id_token)
    anchor_genes = genes[genes["gene_id_norm"].isin(set(gene_to_gid.keys()))].copy()
    if anchor_genes.empty:
        return out

    anchor_genes["anchor_bp"] = (
        (anchor_genes["start"].astype(np.int64) + anchor_genes["end"].astype(np.int64)) // 2
    ).astype(np.int64)
    anchor_genes = anchor_genes.drop_duplicates(subset=["chr_norm", "gene_id_norm", "anchor_bp"])

    anchor_lookup: dict[str, np.ndarray] = {}
    for chrom, sub in anchor_genes.groupby("chr_norm", sort=False):
        anchor_lookup[str(chrom)] = np.sort(sub["anchor_bp"].to_numpy(dtype=np.int64))

    half_window = max(1, int(window_bp) // 2)
    by_chr = defaultdict(list)
    for idx, chrom in enumerate(np.asarray(snp_chr_norm, dtype=object)):
        by_chr[str(chrom)].append(idx)

    for chrom, idxs in by_chr.items():
        anchors = anchor_lookup.get(str(chrom))
        if anchors is None or anchors.size == 0:
            continue
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        pos = np.asarray(snp_pos[idxs_arr], dtype=np.int64)
        left = np.searchsorted(anchors, pos - half_window, side="left")
        right = np.searchsorted(anchors, pos + half_window, side="right")
        out[idxs_arr] = (right - left).astype(np.float32)

    max_count = float(out.max()) if out.size else 0.0
    if max_count > 0:
        out /= max_count
    return out.astype(np.float32, copy=False)


def _build_promoters(genes: pd.DataFrame, promoter_bp: int) -> pd.DataFrame:
    rows = []
    for row in genes.itertuples(index=False):
        if row.strand not in ("+", "-"):
            continue
        if row.strand == "+":
            p_start = max(int(row.start) - promoter_bp, 1)
            p_end = int(row.start)
        else:
            p_start = int(row.end)
            p_end = int(row.end) + promoter_bp
        if p_end < p_start:
            continue
        rows.append((row.chr, row.chr_norm, p_start, p_end, p_start - 1, p_end, row.gene_id))
    return pd.DataFrame(
        rows,
        columns=["chr", "chr_norm", "start", "end", "start0", "end0", "gene_id"],
    )


def annotate_snps_with_genes(
    snps_df: pd.DataFrame,
    gene_gff_path: str | None,
    promoter_bp: int = 2000,
    output_path: str | None = None,
    id_normalizer=normalize_gene_chr_label,
    genes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if genes_df is None:
        if not gene_gff_path:
            out = snps_df.copy()
            out["is_genic"] = False
            out["is_promoter"] = False
            out["gene_region"] = "intergenic"
            out["gene_id"] = "none"
            return out
        if not os.path.exists(gene_gff_path):
            logging.warning(f"Gene GFF not found: {gene_gff_path}")
            out = snps_df.copy()
            out["is_genic"] = False
            out["is_promoter"] = False
            out["gene_region"] = "intergenic"
            out["gene_id"] = "none"
            return out
        genes = load_gene_gff(gene_gff_path, id_normalizer=id_normalizer)
    else:
        genes = genes_df
    if genes is None or genes.empty:
        out = snps_df.copy()
        out["is_genic"] = False
        out["is_promoter"] = False
        out["gene_region"] = "intergenic"
        out["gene_id"] = "none"
        return out

    pr = None
    try:
        import pyranges as pr
    except Exception as e:
        logging.info(f"PyRanges not available for gene annotation; using Python fallback: {e}")
    if genes.empty:
        out = snps_df.copy()
        out["is_genic"] = False
        out["is_promoter"] = False
        out["gene_region"] = "intergenic"
        out["gene_id"] = "none"
        return out

    promoters = _build_promoters(genes, promoter_bp=promoter_bp)

    snps = snps_df.copy()
    snps["chr"] = snps["chr"].astype(str)
    if id_normalizer:
        snps["chr_norm"] = snps["chr"].map(id_normalizer)
    else:
        snps["chr_norm"] = snps["chr"]
    snps["chr_norm"] = snps["chr_norm"].fillna("NA")
    snps["start0"] = (snps["pos"].astype(int) - 1).clip(lower=0)
    snps["end0"] = snps["pos"].astype(int)

    if pr is not None:
        gr_snps = pr.PyRanges(
            snps[["chr_norm", "start0", "end0", "snp_id"]].rename(
                columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
            )
        )
        gr_genes = pr.PyRanges(
            genes[["chr_norm", "start0", "end0", "gene_id"]].rename(
                columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
            )
        )
        gr_prom = pr.PyRanges(
            promoters[["chr_norm", "start0", "end0", "gene_id"]].rename(
                columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
            )
        )

        genic = gr_snps.join(gr_genes, how="left").df
        prom = gr_snps.join(gr_prom, how="left").df
    else:
        genic = _collect_point_interval_overlaps(
            snps[["snp_id", "chr_norm", "pos"]],
            genes[["chr_norm", "start", "end", "gene_id"]],
            ["gene_id"],
        )
        prom = _collect_point_interval_overlaps(
            snps[["snp_id", "chr_norm", "pos"]],
            promoters[["chr_norm", "start", "end", "gene_id"]],
            ["gene_id"],
        )

    genic_grouped = genic.groupby("snp_id", sort=False)
    prom_grouped = prom.groupby("snp_id", sort=False)

    def _collect_ids(series: pd.Series) -> list[str]:
        return [v for v in (_clean_te_token(x) for x in series) if v is not None]

    genic_ids = genic_grouped["gene_id"].apply(lambda s: ";".join(sorted(set(_collect_ids(s)))))
    prom_ids = prom_grouped["gene_id"].apply(lambda s: ";".join(sorted(set(_collect_ids(s)))))

    is_genic = genic_ids.apply(lambda s: bool(s))
    is_prom = prom_ids.apply(lambda s: bool(s))

    out = snps_df.copy()
    out["is_genic"] = out["snp_id"].map(is_genic).fillna(False).astype(bool)
    out["is_promoter"] = out["snp_id"].map(is_prom).fillna(False).astype(bool)
    out.loc[out["is_genic"], "is_promoter"] = False

    out["gene_region"] = "intergenic"
    out.loc[out["is_genic"], "gene_region"] = "genic"
    out.loc[~out["is_genic"] & out["is_promoter"], "gene_region"] = "promoter"

    out["gene_id"] = "none"
    out.loc[out["is_genic"], "gene_id"] = out["snp_id"].map(genic_ids)
    out.loc[~out["is_genic"] & out["is_promoter"], "gene_id"] = out["snp_id"].map(prom_ids)
    out["gene_id"] = out["gene_id"].replace("", "none").fillna("none")

    if output_path:
        out.to_csv(output_path, sep="\t", index=False)
        logging.info(f"Wrote SNP gene annotations: {output_path}")

    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for start, end in sorted(intervals):
        if not merged:
            merged.append([int(start), int(end)])
            continue
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], int(end))
        else:
            merged.append([int(start), int(end)])
    return [(int(s), int(e)) for s, e in merged]


def _build_interval_lookup(
    df: pd.DataFrame,
    chr_col: str,
    start_col: str,
    end_col: str,
    normalizer=None
) -> dict[str, list[tuple[int, int]]]:
    if df is None or df.empty:
        return {}
    tmp = df[[chr_col, start_col, end_col]].copy()
    tmp[chr_col] = tmp[chr_col].astype(str)
    if normalizer is not None:
        tmp["chr_norm"] = tmp[chr_col].map(normalizer)
    else:
        tmp["chr_norm"] = tmp[chr_col]
    tmp = tmp.dropna(subset=["chr_norm", start_col, end_col])
    tmp[start_col] = pd.to_numeric(tmp[start_col], errors="coerce")
    tmp[end_col] = pd.to_numeric(tmp[end_col], errors="coerce")
    tmp = tmp.dropna(subset=[start_col, end_col])
    tmp[start_col] = tmp[start_col].astype(int)
    tmp[end_col] = tmp[end_col].astype(int)
    lookup = {}
    for chrom, sub in tmp.groupby("chr_norm", sort=False):
        intervals = [
            (int(s), int(e)) for s, e in zip(sub[start_col].values, sub[end_col].values)
            if int(s) <= int(e)
        ]
        lookup[str(chrom)] = _merge_intervals(intervals)
    return lookup


def _distances_to_intervals_for_chrom(
    positions: np.ndarray,
    intervals: list[tuple[int, int]],
    fallback: float
) -> np.ndarray:
    if positions.size == 0:
        return np.array([], dtype=np.float32)
    if not intervals:
        return np.full(positions.shape, float(fallback), dtype=np.float32)
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    dists = np.empty_like(positions, dtype=np.float32)
    idx = 0
    n = len(intervals)
    for i, pos in enumerate(positions):
        pos = int(pos)
        while idx < n and ends[idx] < pos:
            idx += 1
        best = None
        if idx < n:
            s, e = intervals[idx]
            if s <= pos <= e:
                dists[i] = 0.0
                continue
            if pos < s:
                best = s - pos
        if idx > 0:
            s_prev, e_prev = intervals[idx - 1]
            if s_prev <= pos <= e_prev:
                dists[i] = 0.0
                continue
            if pos > e_prev:
                prev_dist = pos - e_prev
                best = prev_dist if best is None else min(best, prev_dist)
        dists[i] = float(best if best is not None else fallback)
    return dists


def compute_distance_to_intervals(
    snp_chr_norm: np.ndarray,
    snp_pos: np.ndarray,
    interval_lookup: dict[str, list[tuple[int, int]]],
    chr_len_map: dict[str, int]
) -> np.ndarray:
    distances = np.full(snp_pos.shape, np.nan, dtype=np.float32)
    if snp_pos.size == 0:
        return distances
    fallback_global = float(max(chr_len_map.values())) if chr_len_map else float(np.nanmax(snp_pos))
    by_chr = defaultdict(list)
    for idx, chrom in enumerate(snp_chr_norm):
        by_chr[str(chrom)].append(idx)
    for chrom, idxs in by_chr.items():
        pos = snp_pos[idxs].astype(int)
        order = np.argsort(pos)
        pos_sorted = pos[order]
        fallback = float(chr_len_map.get(str(chrom), fallback_global))
        intervals = interval_lookup.get(str(chrom), [])
        dist_sorted = _distances_to_intervals_for_chrom(pos_sorted, intervals, fallback)
        distances[np.array(idxs, dtype=int)[order]] = dist_sorted
    distances = np.where(np.isfinite(distances), distances, fallback_global).astype(np.float32)
    return distances


def compute_block_region_features(
    haplotype_blocks: pd.DataFrame,
    haplo_row: np.ndarray,
    maf_row: np.ndarray,
    genes: pd.DataFrame | None,
    id_normalizer=None,
) -> dict[str, np.ndarray]:
    n = len(haplo_row)
    zeros = np.zeros(n, dtype=np.float32)
    if haplotype_blocks is None or haplotype_blocks.empty or n == 0:
        return {
            "block_gene_count_norm": zeros,
            "block_mean_maf_norm": zeros,
            "block_snp_density_norm": zeros,
        }

    blocks = haplotype_blocks.copy().reset_index(drop=True)
    blocks["block_id"] = np.arange(1, len(blocks) + 1, dtype=int)
    blocks["chr_norm"] = blocks["CHR"].astype(str)
    if id_normalizer is not None:
        blocks["chr_norm"] = blocks["chr_norm"].map(id_normalizer)
    blocks["start0"] = (blocks["BP1"].astype(int) - 1).clip(lower=0)
    blocks["end0"] = blocks["BP2"].astype(int)
    block_len_bp = (blocks["BP2"].astype(int) - blocks["BP1"].astype(int) + 1).clip(lower=1)
    block_len_map = pd.Series(block_len_bp.values, index=blocks["block_id"])
    blocks_for_gene_join = blocks[blocks["chr_norm"].notna()].copy()

    valid_blocks = haplo_row > 0
    block_ids = pd.Series(haplo_row[valid_blocks].astype(int))
    snp_counts = block_ids.value_counts()
    mean_maf = pd.Series(maf_row[valid_blocks]).groupby(block_ids).mean()

    gene_counts = pd.Series(dtype=float)
    if genes is not None and not genes.empty and not blocks_for_gene_join.empty:
        try:
            import pyranges as pr
            gr_blocks = pr.PyRanges(
                blocks_for_gene_join[["chr_norm", "start0", "end0", "block_id"]].rename(
                    columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
                )
            )
            gr_genes = pr.PyRanges(
                genes[["chr_norm", "start0", "end0", "gene_id"]].rename(
                    columns={"chr_norm": "Chromosome", "start0": "Start", "end0": "End"}
                )
            )
            joined = gr_blocks.join(gr_genes, how="left").df
            gene_counts = (
                joined.dropna(subset=["gene_id"])
                .groupby("block_id")["gene_id"]
                .nunique()
                .astype(float)
            )
        except Exception as e:
            logging.warning(f"Block gene counts skipped: {e}")

    max_gene = float(gene_counts.max()) if len(gene_counts) else 1.0
    gene_norm = gene_counts / max_gene if max_gene > 0 else gene_counts

    density = snp_counts / (block_len_map.reindex(snp_counts.index) / 1e6)
    max_density = float(density.max()) if len(density) else 1.0
    density_norm = density / max_density if max_density > 0 else density

    maf_norm = mean_maf * 2.0

    gene_norm_map = gene_norm.to_dict()
    density_norm_map = density_norm.to_dict()
    maf_norm_map = maf_norm.to_dict()

    block_gene_count_norm = pd.Series(haplo_row).map(gene_norm_map).fillna(0.0).to_numpy(np.float32)
    block_snp_density_norm = pd.Series(haplo_row).map(density_norm_map).fillna(0.0).to_numpy(np.float32)
    block_mean_maf_norm = pd.Series(haplo_row).map(maf_norm_map).fillna(0.0).to_numpy(np.float32)

    return {
        "block_gene_count_norm": block_gene_count_norm,
        "block_mean_maf_norm": block_mean_maf_norm,
        "block_snp_density_norm": block_snp_density_norm,
    }

def compute_call_rate_per_snp(encoded_df: pd.DataFrame) -> pd.Series:
    """
    Per-SNP call-rate across samples.
    - Numeric dosage tables: finite values in [0..PLOIDY] count as called.
    - Allele-code tables: require two valid digits '0'..'3' and exclude missing tokens.
    """
    ann = {'Chromosome','Position','Haplotype_Block','MAF','MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        return pd.Series([], index=encoded_df.index, dtype=float)

    if all(pd.api.types.is_numeric_dtype(encoded_df[c]) for c in sample_cols):
        X = encoded_df[sample_cols].to_numpy(dtype=np.float32, copy=False)
        valid = np.isfinite(X) & (X >= 0.0) & (X <= float(PLOIDY))
        called = valid.sum(axis=1)
        return pd.Series(called / max(1, X.shape[1]), index=encoded_df.index, dtype=float)

    vals = encoded_df[sample_cols].apply(pd.to_numeric, errors='coerce')
    X = vals.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(X)
    if finite.any():
        vmax = float(np.nanmax(X[finite]))
        vmin = float(np.nanmin(X[finite]))
        if vmin >= -1e-6 and vmax <= float(PLOIDY) + 1e-6:
            called = finite.sum(axis=1)
            return pd.Series(called / max(1, X.shape[1]), index=encoded_df.index, dtype=float)

    A = encoded_df[sample_cols].astype('S2').to_numpy(copy=False)
    A = np.ascontiguousarray(A)
    n_rows, n_cols = A.shape
    B = A.view(np.uint8).reshape(n_rows, n_cols, 2)
    b0, b1 = B[..., 0], B[..., 1]

    valid = (b0 >= 48) & (b0 <= 51) & (b1 >= 48) & (b1 <= 51)
    called = valid.sum(axis=1)
    return pd.Series(called / np.maximum(1, n_cols), index=encoded_df.index, dtype=float)


def compute_call_rate_per_sample(encoded_df: pd.DataFrame) -> pd.Series:
    """
    Per-sample call-rate across SNPs using the same missing-value rules as
    compute_call_rate_per_snp().
    """
    ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        return pd.Series(dtype=float)

    if all(pd.api.types.is_numeric_dtype(encoded_df[c]) for c in sample_cols):
        X = encoded_df[sample_cols].to_numpy(dtype=np.float32, copy=False)
        valid = np.isfinite(X) & (X >= 0.0) & (X <= float(PLOIDY))
        return pd.Series(valid.mean(axis=0), index=sample_cols, dtype=float)

    vals = encoded_df[sample_cols].apply(pd.to_numeric, errors='coerce')
    X = vals.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(X)
    if finite.any():
        vmax = float(np.nanmax(X[finite]))
        vmin = float(np.nanmin(X[finite]))
        if vmin >= -1e-6 and vmax <= float(PLOIDY) + 1e-6:
            return pd.Series(finite.mean(axis=0), index=sample_cols, dtype=float)

    A = encoded_df[sample_cols].astype('S2').to_numpy(copy=False)
    A = np.ascontiguousarray(A)
    n_rows, n_cols = A.shape
    B = A.view(np.uint8).reshape(n_rows, n_cols, 2)
    b0, b1 = B[..., 0], B[..., 1]
    valid = (b0 >= 48) & (b0 <= 51) & (b1 >= 48) & (b1 <= 51)
    return pd.Series(valid.mean(axis=0), index=sample_cols, dtype=float)

def build_sg_encoder(sorted_chr_labels: list[str], sg_map: dict[str, list[str]]):
    """
    Returns: (sg_names, row_to_onehot) where `row_to_onehot(row_label)` gives an N-hot vector.
    If sg_map is empty, returns ([], lambda *_: np.empty(0, dtype=np.float32))
    """
    if not sg_map:
        return [], (lambda *_: np.empty((0,), dtype=np.float32))
    sg_names = sorted(sg_map.keys())
    label_to_idx = {lab:i for i,lab in enumerate(sorted_chr_labels)}
    # Build a set per SG for fast membership
    sg_sets = {k: set(v) for k,v in sg_map.items()}

    def row_to_onehot(row_label: str) -> np.ndarray:
        out = np.zeros((len(sg_names),), dtype=np.float32)
        for i, sg in enumerate(sg_names):
            if row_label in sg_sets[sg]:
                out[i] = 1.0
        return out

    return sg_names, row_to_onehot


def _sg_dirs(base_snps_dir, base_hap_dir):
    return (os.path.join(base_snps_dir, "A"), os.path.join(base_snps_dir, "C"),
            os.path.join(base_hap_dir,  "A"), os.path.join(base_hap_dir,  "C"))

def _snp_index_path(sample, sg_dir, label):         # label 'A' or 'C'
    return os.path.join(sg_dir, f"{sample}_{label}_tile_index.csv")

def _hap_index_path(sample, sg_dir, label):
    return os.path.join(sg_dir, f"{sample}_{label}_haplo_tile_index.csv")

def _has_snp_done(sample, snp_dir_A, snp_dir_C):
    return os.path.exists(_snp_index_path(sample, snp_dir_A, "A")) and \
           os.path.exists(_snp_index_path(sample, snp_dir_C, "C"))

def _has_hap_done(sample, hap_dir_A, hap_dir_C):
    return os.path.exists(_hap_index_path(sample, hap_dir_A, "A")) and \
           os.path.exists(_hap_index_path(sample, hap_dir_C, "C"))

def _existing_W_from_npz(sample, sg_dir, label):
    """Read desired_max_width_px from any existing SNP *.npz tile (fast & exact)."""
    pat = os.path.join(sg_dir, f"{sample}_{label}_tile_*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    try:
        with np.load(files[0], allow_pickle=False) as z:
            meta = z["meta"]
        return int(meta[4])  # [H, W, strip_height, collision_depth, desired_max_width_px]
    except Exception:
        return None

def _existing_W_fallback(sample, sg_dir, label, tile_width=1024):
    """If no npz, infer width from max tile_idx + 1 (uses index CSV)."""
    idx_csv = _snp_index_path(sample, sg_dir, label)
    if not os.path.exists(idx_csv):
        return None
    try:
        df = pd.read_csv(idx_csv)
        if "tile_idx" in df.columns:
            ntiles = int(df["tile_idx"].max()) + 1
            return ntiles * tile_width
    except Exception:
        return None
    return None

def get_existing_W(sample, snp_dir_A, snp_dir_C, tile_width=1024):
    """Return max desired width among A and C SNP outputs (None if both missing)."""
    WA = _existing_W_from_npz(sample, snp_dir_A, "A") or _existing_W_fallback(sample, snp_dir_A, "A", tile_width)
    WC = _existing_W_from_npz(sample, snp_dir_C, "C") or _existing_W_fallback(sample, snp_dir_C, "C", tile_width)
    if WA is None and WC is None:
        return None
    return max([w for w in [WA, WC] if w is not None])

def load_plink_data(ped_file, map_file, chunk_size=10000):
    """
    Load PLINK .ped and .map files, encode alleles row-by-row,
    and merge Chromosome and Position data.
    Returns: final_df (SNPs x [samples, Chromosome, Position]), map_df, and raw genotype_data.
    """
    if not os.path.exists(ped_file):
        raise FileNotFoundError(f"{ped_file} not found.")
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"{map_file} not found.")

    logging.info("Loading .map file...")
    map_df = pd.read_csv(map_file, sep=r"\s+", header=None,
                         names=['Chromosome', 'SNP', 'Genetic_Distance', 'Position'])
    map_df['Chromosome'] = map_df['Chromosome'].astype(str).map(_normalize_chr_label_to_digits)
    logging.info(f"Loaded {len(map_df)} SNPs from map file.")

    genotype_columns_start = 6
    genotype_columns_end = 6 + 2 * len(map_df)

    logging.info("Loading .ped file in chunks...")
    sample_ids = []
    genotype_chunks = []
    with pd.read_csv(ped_file, sep=r"\s+", header=None,
                     usecols=[1] + list(range(genotype_columns_start, genotype_columns_end)),
                     chunksize=chunk_size, dtype=str) as reader:
        for i, chunk in enumerate(reader):
            sample_ids.extend(chunk.iloc[:, 0].values)
            genotype_chunks.append(chunk.iloc[:, 1:])
            logging.info(f"Processed chunk {i+1} with {len(chunk)} rows.")

    logging.info("Concatenating chunks...")
    genotype_data = pd.concat(genotype_chunks, axis=0)
    genotype_data.index = sample_ids
    genotype_data.columns = map_df['SNP'].repeat(2).values  # duplicate SNP names

    logging.info("Encoding genotypes by per-SNP allele frequency...")
    encoded_genotypes, allele_rank_df = encode_plink_by_frequency(genotype_data, map_df)
    encoded_genotypes.attrs['allele_rank'] = allele_rank_df

    final_df = encoded_genotypes[sample_ids + ['Chromosome', 'Position']]
    final_df.attrs['allele_rank'] = allele_rank_df

    out_path = "merged_genotype_data.csv"
    final_df.to_csv(out_path)
    logging.info(f"Final dataframe saved at {out_path}")

    return final_df, map_df, genotype_data

def _normalize_ped_allele(value):
    """Normalize PED alleles: uppercase, strip whitespace, treat known tokens as missing."""
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    if not s or s in PED_MISSING_ALLELES:
        return ""
    return s


def _dataframe_elementwise_map(df: pd.DataFrame, func):
    """Use DataFrame.map when available; fall back to applymap for older pandas."""
    if hasattr(df, "map"):
        return df.map(func)
    return df.applymap(func)

def encode_plink_by_frequency(genotype_data: pd.DataFrame, map_df: pd.DataFrame):
    """
    Per-SNP encoding that ranks alleles by dataset-wide frequency and builds two-digit codes.
    Returns: (encoded_df [SNPs x samples + Chromosome/Position], allele_rank_df).
    """
    sample_ids = genotype_data.index.to_list()
    num_snps = len(map_df)
    if genotype_data.shape[1] != 2 * num_snps:
        raise ValueError("Expected two columns per SNP in genotype_data.")

    encoded = np.full((num_snps, len(sample_ids)), "-1", dtype='<U2')
    rank_rows = []

    for snp_idx in range(num_snps):
        pair = genotype_data.iloc[:, 2*snp_idx:2*snp_idx+2]
        pair_clean = _dataframe_elementwise_map(pair, _normalize_ped_allele)
        a1 = pair_clean.iloc[:, 0]
        a2 = pair_clean.iloc[:, 1]
        called_pairs = int(((a1 != "") & (a2 != "")).sum())

        values = pair_clean.to_numpy(dtype=object).reshape(-1)
        valid = values[values != ""]
        counts = Counter(valid)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rank_map = {allele: idx for idx, (allele, _) in enumerate(ranked[:4])}

        mapped1 = a1.map(rank_map).to_numpy(dtype=float)
        mapped2 = a2.map(rank_map).to_numpy(dtype=float)
        present = (a1 != "").to_numpy() & (a2 != "").to_numpy()
        valid_mask = present & (~np.isnan(mapped1)) & (~np.isnan(mapped2))
        if valid_mask.any():
            digits1 = mapped1[valid_mask].astype(np.int32).astype(str)
            digits2 = mapped2[valid_mask].astype(np.int32).astype(str)
            encoded[snp_idx, valid_mask] = np.char.add(digits1, digits2)

        snp_row = map_df.iloc[snp_idx]
        rank_row = {
            "SNP": snp_row["SNP"],
            "Chromosome": snp_row["Chromosome"],
            "Position": int(snp_row["Position"]),
            "called_pairs": called_pairs,
            "alleles_seen": ",".join([allele for allele, _ in ranked])
        }
        for idx, label in enumerate(['major', 'minor1', 'minor2', 'minor3']):
            if idx < len(ranked):
                allele, cnt = ranked[idx]
            else:
                allele, cnt = None, 0
            rank_row[f"{label}_allele"] = allele
            rank_row[f"{label}_count"] = cnt
        rank_rows.append(rank_row)

    encoded_df = pd.DataFrame(encoded, index=map_df['SNP'], columns=sample_ids)
    encoded_df['Chromosome'] = map_df['Chromosome'].astype(str).values
    encoded_df['Position'] = map_df['Position'].astype(int).values
    encoded_df = encoded_df[sample_ids + ['Chromosome', 'Position']]
    allele_rank_df = pd.DataFrame(rank_rows)
    return encoded_df, allele_rank_df


def log_allele_combination_diversity(encoded_df: pd.DataFrame, tag: str = "allele_combination"):
    """
    Log top genotype codes and heterozygosity info from 2-char encoded tables.
    """
    ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        logging.info(f"[{tag}] no genotype columns to summarize.")
        return

    values = encoded_df[sample_cols].values.ravel()
    counts = pd.Series(values.astype(str)).value_counts()
    logging.info(f"[{tag}] top genotype codes:\n{counts.head(10)}")

    def het_rate_row(row):
        het = 0
        called = 0
        for val in row:
            s = str(val).strip()
            if len(s) != 2 or s in {'-1', '99'}:
                continue
            if s[0] not in '0123' or s[1] not in '0123':
                continue
            called += 1
            if s[0] != s[1]:
                het += 1
        return het / called if called else np.nan

    hr = encoded_df[sample_cols].apply(het_rate_row, axis=1)
    mean = float(np.nanmean(hr))
    frac = float((hr.fillna(0) > 0).mean())
    logging.info(f"[{tag}] mean per-SNP het rate={mean:.4f}, SNPs with het={frac:.4f}")

def series_codes_to_dosage_norm(sample_series: np.ndarray) -> np.ndarray:
    """
    Fast per-sample conversion to normalized dosage in [0,1], preserving
    code_to_dosage_norm semantics.
    """
    arr = np.asarray(sample_series)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    if np.issubdtype(arr.dtype, np.number):
        vals = arr.astype(np.float32, copy=False)
        valid = np.isfinite(vals) & (vals >= 0.0) & (vals <= float(PLOIDY))
        if valid.any():
            out[valid] = np.rint(vals[valid]) / float(PLOIDY)
        return out

    # Fast path for common two-character encodings ('00','01','11','-1','99').
    # Guard with exact string length to avoid truncation artifacts from longer strings.
    arr_str = arr.astype(str, copy=False)
    len2 = np.char.str_len(arr_str) == 2
    s = np.ascontiguousarray(arr_str.astype('S2', copy=False))
    b = s.view(np.uint8).reshape(n, 2)
    b0, b1 = b[:, 0], b[:, 1]
    is_empty = arr_str == ''
    is_missing = ((((b0 == 45) & (b1 == 49)) | ((b0 == 57) & (b1 == 57))) & len2) | is_empty
    is_digit_pair = (b0 >= 48) & (b0 <= 51) & (b1 >= 48) & (b1 <= 51)
    valid_pair = is_digit_pair & len2 & (~is_missing)
    if valid_pair.any():
        alt = (b0 != 48).astype(np.float32) + (b1 != 48).astype(np.float32)
        out[valid_pair] = alt[valid_pair] / float(PLOIDY)

    unresolved = ~(valid_pair | is_missing)
    if unresolved.any():
        arr_obj = arr.astype(object, copy=False)
        for i in np.flatnonzero(unresolved):
            out[i] = code_to_dosage_norm(arr_obj[i])

    return out

def build_major_allele_table(encoded_df: pd.DataFrame) -> pd.Series:
    geno = encoded_df.drop(columns=['Chromosome','Position','Haplotype_Block','MAF'], errors='ignore')
    def majority(row):
        counts = {'0':0,'1':0,'2':0,'3':0}
        for v in row.values:
            s = str(v)
            if s in ('-1','99') or len(s) != 2: 
                continue
            a, b = s[0], s[1]
            if a in counts: counts[a] += 1
            if b in counts: counts[b] += 1
        return max(counts, key=counts.get)  # digit of the dataset-major allele
    return geno.apply(majority, axis=1).rename('major_digit')

def major_minor_allele(encoded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized major/minor remapping.

    Input cells should be 2-char strings like '01','23','-1','99'.
    For each SNP (row), find the dataset-major allele digit among {0,1,2,3}
    across all samples and map:
        major -> '0'; others -> '1','2','3' in descending frequency.
    Preserves '-1'; any other non-2-digit entry becomes '99'.
    """
    logging.info("Starting major/minor allele encoding (vectorized bytes)...")
    df = encoded_df.copy()

    # Identify genotype columns once
    ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
    sample_cols = [c for c in df.columns if c not in ann]
    if not sample_cols:
        logging.info("No sample genotype columns detected; returning input unchanged.")
        return df

    # 2-char fixed-width byte strings (fast to slice/view)
    # NOTE: 'S2' will truncate longer strings; our genotypes are 2-char by design.
    A = df[sample_cols].astype('S2').to_numpy(copy=False)
    A = np.ascontiguousarray(A)                               # ensure contiguity for .view
    n_rows, n_cols = A.shape

    # View each 'S2' element as its two ASCII bytes
    B  = A.view(np.uint8).reshape(n_rows, n_cols, 2)
    b0 = B[..., 0]                                            # first char per genotype
    b1 = B[..., 1]                                            # second char per genotype

    # Masks
    # valid genotype: both chars in {'0','1','2','3'} -> ASCII 48..51
    valid  = (b0 >= 48) & (b0 <= 51) & (b1 >= 48) & (b1 <= 51)
    is_neg1 = (b0 == 45) & (b1 == 49)                         # '-1'

    # Count allele digits across samples (counts per row for '0','1','2','3')
    c0 = ((b0 == 48) & valid).sum(axis=1) + ((b1 == 48) & valid).sum(axis=1)
    c1 = ((b0 == 49) & valid).sum(axis=1) + ((b1 == 49) & valid).sum(axis=1)
    c2 = ((b0 == 50) & valid).sum(axis=1) + ((b1 == 50) & valid).sum(axis=1)
    c3 = ((b0 == 51) & valid).sum(axis=1) + ((b1 == 51) & valid).sum(axis=1)
    counts = np.stack([c0, c1, c2, c3], axis=1)               # shape (n_rows, 4)

    # Row-wise ranking of digits by descending frequency.
    # args: 0..3 correspond to allele-rank digits '0','1','2','3'
    order = np.argsort(-counts, axis=1, kind='stable')

    # Build inverse map per row: inv[row, digit_index] -> rank (0 for major)
    inv = np.empty_like(order)
    inv[np.arange(n_rows)[:, None], order] = np.arange(4)[None, :]

    # Map ASCII '0'..'3' to indices 0..3, then to ranks 0..3
    idx0 = np.clip(b0 - 48, 0, 3)
    idx1 = np.clip(b1 - 48, 0, 3)
    new0 = inv[np.arange(n_rows)[:, None], idx0]
    new1 = inv[np.arange(n_rows)[:, None], idx1]

    # Build output ASCII codes for the remapped digits
    # default everything to '99'
    out0 = np.full((n_rows, n_cols), 57, dtype=np.uint8)      # '9'
    out1 = np.full((n_rows, n_cols), 57, dtype=np.uint8)      # '9'
    # valid -> '0'+rank
    out0[valid] = 48 + new0[valid]                             # '0'..'3'
    out1[valid] = 48 + new1[valid]
    # preserve '-1'
    out0[is_neg1] = 45                                         # '-'
    out1[is_neg1] = 49                                         # '1'

    # Convert ASCII codes back to 2-char strings
    s0  = np.char.mod('%c', out0)
    s1  = np.char.mod('%c', out1)
    out = np.char.add(s0, s1)                                  # dtype '<U2'

    # Single block assignment back into the DataFrame
    df.loc[:, sample_cols] = out
    logging.info("Major/minor allele encoding complete (vectorized bytes).")
    return df


def build_allele_rank_summary(encoded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame that ranks the major and minor alleles per SNP.
    Works on datasets encoded as two-character allele codes (e.g., '12','11','-1').
    """
    ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        return pd.DataFrame(columns=[
            'SNP', 'Chromosome', 'Position', 'called_pairs',
            'major_digit', 'major_count',
            'minor1_digit', 'minor1_count',
            'minor2_digit', 'minor2_count',
            'minor3_digit', 'minor3_count'
        ])

    digits = ['0', '1', '2', '3']
    valid_digits = set(digits)
    summary_rows = []
    for snp, row in encoded_df.iterrows():
        counts = {d: 0 for d in digits}
        called_pairs = 0
        for val in row[sample_cols]:
            s = str(val).strip()
            if len(s) != 2 or s in {'-1', '99'}:
                continue
            if s[0] not in valid_digits or s[1] not in valid_digits:
                continue
            counts[s[0]] += 1
            counts[s[1]] += 1
            called_pairs += 1

        sorted_digits = sorted(digits, key=lambda d: (-counts[d], d))
        ranked = [d for d in sorted_digits if counts[d] > 0]
        while len(ranked) < 4:
            ranked.append('')

        row_data = {
            'SNP': snp,
            'Chromosome': row.get('Chromosome'),
            'Position': row.get('Position'),
            'called_pairs': called_pairs
        }
        labels = ['major', 'minor1', 'minor2', 'minor3']
        for i, label in enumerate(labels):
            digit = ranked[i]
            row_data[f'{label}_digit'] = digit
            row_data[f'{label}_count'] = counts[digit] if digit else 0
        summary_rows.append(row_data)

    return pd.DataFrame(summary_rows)

def compute_maf_from_allele_codes(encoded_df: pd.DataFrame) -> pd.Series:
    """
    Compute a multiallelic MAF proxy from two-character allele-rank codes.

    For each SNP, count allele-rank digits 0..3 across all called genotypes and
    return the second-highest observed allele frequency. Monomorphic SNPs map to 0.
    """
    ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        return pd.Series([], index=encoded_df.index, dtype=float)

    A = encoded_df[sample_cols].astype('S2').to_numpy(copy=False)
    A = np.ascontiguousarray(A)
    n_rows, n_cols = A.shape
    B = A.view(np.uint8).reshape(n_rows, n_cols, 2)
    b0, b1 = B[..., 0], B[..., 1]

    valid = (b0 >= 48) & (b0 <= 51) & (b1 >= 48) & (b1 <= 51)
    counts = np.stack([
        ((b0 == 48) & valid).sum(axis=1) + ((b1 == 48) & valid).sum(axis=1),
        ((b0 == 49) & valid).sum(axis=1) + ((b1 == 49) & valid).sum(axis=1),
        ((b0 == 50) & valid).sum(axis=1) + ((b1 == 50) & valid).sum(axis=1),
        ((b0 == 51) & valid).sum(axis=1) + ((b1 == 51) & valid).sum(axis=1),
    ], axis=1).astype(np.float64)

    denom = valid.sum(axis=1).astype(np.float64) * 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        freqs = counts / denom[:, None]
    freqs_sorted = np.sort(freqs, axis=1)[:, ::-1]
    num_pos = (counts > 0).sum(axis=1)
    maf = np.where(num_pos >= 2, freqs_sorted[:, 1], 0.0)
    maf = np.where(denom == 0, np.nan, maf)
    return pd.Series(maf, index=encoded_df.index, dtype=float)

def compute_maf_from_dosage(dosage_df: pd.DataFrame) -> pd.Series:
    """
    dosage_df: SNP x samples (+ Chromosome, Position columns).
    MAF = min(mean(DS/2), 1-mean(DS/2)) ignoring NaNs.
    """
    ann = {'Chromosome','Position','Haplotype_Block','MAF','MA'}
    sample_cols = [c for c in dosage_df.columns if c not in ann]
    if not sample_cols:
        return pd.Series([], index=dosage_df.index, dtype=float)
    A = dosage_df[sample_cols].astype(float).to_numpy()
    with np.errstate(invalid='ignore'):
        p = np.nanmean(A / float(PLOIDY), axis=1)
    maf = np.minimum(p, 1.0 - p)
    return pd.Series(maf, index=dosage_df.index, dtype=float)


def encoded_to_dosage_df(encoded_df: pd.DataFrame) -> pd.DataFrame:
    ann = {'Chromosome','Position','Haplotype_Block','MAF','MA'}
    sample_cols = [c for c in encoded_df.columns if c not in ann]
    if not sample_cols:
        return pd.DataFrame(columns=['Chromosome','Position'])
    dosage = encoded_df[sample_cols].copy()
    dosage = _dataframe_elementwise_map(dosage, _parse_to_dosage_012).astype(np.float32)
    dosage['Chromosome'] = encoded_df['Chromosome'].values
    dosage['Position'] = encoded_df['Position'].values
    return dosage


def maf_to_mix_weight(maf, gamma=0.7):
    """
    Map MAF in [0, 0.5] to a mixing weight t in [0,1].
    t=0 => pure white (very rare); t=1 => original color (common).
    gamma<1 accentuates differences at low MAF.
    """
    if maf is None or (isinstance(maf, float) and (np.isnan(maf) or np.isinf(maf))):
        return 1.0
    t = np.clip(maf / 0.5, 0.0, 1.0)
    return float(t ** gamma)

def filter_snps_by_maf_in_batches(encoded_df, maf_threshold=0, batch_size=1000):
    """Filters SNP rows in batches based on a MAF threshold."""
    batches = []
    for start in range(0, encoded_df.shape[0], batch_size):
        batch = encoded_df.iloc[start:start+batch_size]
        ann = {'Chromosome', 'Position', 'Haplotype_Block', 'MAF', 'MA'}
        sample_cols = [c for c in batch.columns if c not in ann]
        if sample_cols and all(pd.api.types.is_numeric_dtype(batch[c]) for c in sample_cols):
            maf_series = compute_maf_from_dosage(batch)
        else:
            maf_series = compute_maf_from_allele_codes(batch)
        keep = maf_series.index[(maf_series.notna()) & (maf_series >= maf_threshold)]
        batches.append(batch.loc[batch.index.intersection(keep)])
    return pd.concat(batches, axis=0)

def tri_allelic_rate(encoded_df: pd.DataFrame) -> float:
    drop_cols = [c for c in ['Chromosome','Position','Haplotype_Block','MAF'] if c in encoded_df.columns]
    geno = encoded_df.drop(columns=drop_cols, errors='ignore')
    if geno.shape[1] == 0:
        return float('nan')

    # If this looks like numeric dosage data, we cannot recover alleles.
    # Instead, report the fraction of SNPs with any dosage outside [0, PLOIDY]
    # as a proxy for "unexpected/multi-allelic" loci.
    dos = geno.apply(pd.to_numeric, errors='coerce')
    num_finite = np.isfinite(dos.values).sum()
    if num_finite > 0.5 * dos.size:
        out_of_range = ((dos < 0) | (dos > float(PLOIDY))).any(axis=1)
        rate = float(out_of_range.mean())
        logging.info(f"tri_allelic_rate (dosage mode): fraction with out-of-range dosage = {rate:.4f}")
        return rate

    # Legacy path for string-coded alleles (2-char codes like '12','34', etc.)
    def count_alleles(row):
        seen = set()
        for v in row.values:
            s = str(v)
            if s in ('-1','99') or len(s) != 2: 
                continue
            seen.update([s[0], s[1]])
        return len(seen)
    k = geno.apply(count_alleles, axis=1)
    return float((k >= 3).mean())

def instability_score(encoded_df: pd.DataFrame) -> pd.Series:
    drop_cols = [c for c in ['Chromosome','Position','Haplotype_Block','MAF'] if c in encoded_df.columns]
    geno = encoded_df.drop(columns=drop_cols, errors='ignore')

    def gap(row):
        counts = {d: 0 for d in '0123'}
        for v in row.values:
            s = str(v)
            if s in ('-1', '99') or len(s) != 2:
                continue
            if s[0] in counts: counts[s[0]] += 1
            if s[1] in counts: counts[s[1]] += 1
        tot = sum(counts.values())
        if tot == 0:
            return np.nan
        top = sorted((counts[d] for d in '0123'), reverse=True)
        return (top[0] - top[1]) / tot

    return geno.apply(gap, axis=1)

def load_haplotype_blocks(file_path):
    """
    Load haplotype block information from a PLINK output file.
    
    The expected input file should have a header with the following columns:
        CHR, BP1, BP2, KB, NSNPS, SNPS
    """
    haplotype_blocks = pd.read_csv(file_path, sep=r"\s+")
    haplotype_blocks['CHR'] = haplotype_blocks['CHR'].astype(str).map(_normalize_chr_label_to_digits)
    required_columns = ['CHR', 'BP1', 'BP2', 'KB', 'NSNPS', 'SNPS']
    missing_cols = [col for col in required_columns if col not in haplotype_blocks.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in haplotype block file: {missing_cols}")
    return haplotype_blocks

def add_haplotype_block_info(encoded_genotypes, haplotype_blocks):
    """
    Add haplotype block information to the encoded_genotypes DataFrame.
    
    The encoded_genotypes DataFrame is expected to have at least the following columns:
        - 'Chromosome' : indicating the chromosome for each SNP
        - 'Position'   : the genomic coordinate of the SNP
        
    For each haplotype block (defined by [BP1, BP2] in the haplotype_blocks file), 
    this function assigns a unique block id to SNPs whose positions fall within the block on the matching chromosome.
    SNPs that do not fall into any block will be assigned a value of 0.
    """
    # Make a copy to avoid modifying the original dataframe
    encoded_genotypes = encoded_genotypes.copy()
    encoded_genotypes['Haplotype_Block'] = 0  # Default: no block
    
    # Ensure that the chromosome fields in both dataframes have the same type.
    # If haplotype_blocks['CHR'] is numeric, convert encoded_genotypes['Chromosome'] to numeric,
    # otherwise cast both to string.
    if pd.api.types.is_numeric_dtype(haplotype_blocks['CHR']):
        encoded_genotypes['Chromosome'] = pd.to_numeric(encoded_genotypes['Chromosome'], errors='coerce')
    else:
        encoded_genotypes['Chromosome'] = encoded_genotypes['Chromosome'].astype(str).map(_normalize_chr_label_to_digits)
        haplotype_blocks['CHR'] = haplotype_blocks['CHR'].astype(str).map(_normalize_chr_label_to_digits)
    
    # Iterate over each haplotype block and assign a unique block ID (starting from 1)
    for block_id, row in enumerate(haplotype_blocks.itertuples(index=False), start=1):
        # Create a mask: SNPs on the same chromosome and with positions between BP1 and BP2 (inclusive)
        mask = (encoded_genotypes['Chromosome'] == row.CHR) & \
               (encoded_genotypes['Position'] >= row.BP1) & \
               (encoded_genotypes['Position'] <= row.BP2)
        encoded_genotypes.loc[mask, 'Haplotype_Block'] = block_id

    return encoded_genotypes

def _parse_to_dosage_012(x) -> float:
    """
    Return 0.0/1.0/2.0 for:
      - numeric dosages 0/1/2 (or 0.0..2.0),
      - GT strings '0/0','0|1','1/1','0/2','1/2' (any-alt count),
      - legacy two-char codes '00','01','11','12',... (non-zero digits = alt).
    Missing -> np.nan.
    """
    if x is None: return np.nan
    # numeric
    if isinstance(x, (int, np.integer, float, np.floating)):
        if np.isnan(x): return np.nan
        if 0.0 <= float(x) <= 2.0:
            return float(int(round(float(x))))  # 0/1/2
        return np.nan
    s = str(x).strip()
    if s in {"-1","99","",".", "./.", ".|.", "nan","NaN","None"}:
        return np.nan
    if re.fullmatch(r'[012]', s):
        return float(int(s))
    m = re.match(r'^([0-9]+)[/|]([0-9]+)$', s)
    if m:
        a, b = m.groups()
        if a == '.' or b == '.':
            return np.nan
        return float((int(a) != 0) + (int(b) != 0))  # any-alt dosage
    if re.fullmatch(r'\d{2}', s) and all(ch in '0123' for ch in s):
        return float((s[0] != '0') + (s[1] != '0'))
    try:
        v = float(s)
        if 0.0 <= v <= 2.0:
            return float(int(round(v)))
    except Exception:
        pass
    return np.nan

def code_to_dosage_norm(code: str) -> float:
    """
    Normalize to [0,1] by dividing dosage(0/1/2) by diploid PLOIDY.
    """
    d = _parse_to_dosage_012(code)
    return d / float(PLOIDY) if not (isinstance(d, float) and np.isnan(d)) else np.nan

def load_vcf_as_dosage(vcf_path: str, use_DS_if_present: bool = True,
                       samples: list[str] | None = None, region: str | None = None):
    """
    Returns DataFrame with shape [SNPs x (samples + Chromosome + Position)].
    Each sample cell is a dosage 0/1/2 (float if DS used), NaN if missing.
    Requires cyvcf2 (preferred) or pysam as a fallback.
    """
    try:
        from cyvcf2 import VCF
        v = VCF(vcf_path)
        if samples:
            v.set_samples(samples)
        sample_names = v.samples
        rows = []
        chroms, poss, idx = [], [], []
        iterator = v(region) if region else v
        for var in iterator:
            chroms.append(str(var.CHROM))
            poss.append(int(var.POS))
            idx.append(var.ID if var.ID else f"{var.CHROM}:{var.POS}")
            if use_DS_if_present and ('DS' in (var.FORMAT or ())):
                ds = var.format('DS')  # (n_samples,)
                # ds may be None for some records; fall back to GT if so
                if ds is None:
                    gts = var.genotypes  # [[a,b,phased], ...]
                    dos = [(1 if a>0 else 0) + (1 if b>0 else 0) if a>=0 and b>=0 else np.nan
                           for a,b,*_ in gts]
                else:
                    dos = ds.astype(np.float32).ravel()
            else:
                gts = var.genotypes
                dos = [(1 if a>0 else 0) + (1 if b>0 else 0) if a>=0 and b>=0 else np.nan
                       for a,b,*_ in gts]
            rows.append(np.asarray(dos, dtype=np.float32))
        if not rows:
            raise ValueError(f"No variants read from {vcf_path}")
        M = np.vstack(rows)
        df = pd.DataFrame(M, columns=sample_names, index=idx)
        df['Chromosome'] = chroms
        df['Position']   = poss
        return df
    except ImportError:
        try:
            import pysam
        except ImportError as e:
            raise ImportError("Install cyvcf2 (preferred) or pysam to read VCF") from e
        # --- minimal pysam fallback ---
        tb = pysam.TabixFile(vcf_path) if vcf_path.endswith((".gz",".bgz")) else None
        v = pysam.VariantFile(vcf_path)
        sample_names = list(v.header.samples)
        rows, chroms, poss, idx = [], [], [], []
        it = v.fetch(region=region) if region else v.fetch()
        for rec in it:
            chroms.append(str(rec.chrom)); poss.append(int(rec.pos))
            idx.append(rec.id if rec.id else f"{rec.chrom}:{rec.pos}")
            if use_DS_if_present and 'DS' in rec.format.keys():
                ds = [rec.samples[s].get('DS', np.nan) for s in sample_names]
                ds = [float(x) if x is not None else np.nan for x in ds]
                rows.append(np.asarray(ds, dtype=np.float32))
            else:
                g = [rec.samples[s].get('GT', None) for s in sample_names]
                dos = []
                for gt in g:
                    if gt is None or any(a is None for a in gt):
                        dos.append(np.nan); continue
                    # any-alt dosage
                    dos.append(float((gt[0] != 0) + (gt[1] != 0)))
                rows.append(np.asarray(dos, dtype=np.float32))
        M = np.vstack(rows)
        df = pd.DataFrame(M, columns=sample_names, index=idx)
        df['Chromosome'] = chroms
        df['Position']   = poss
        return df

# def code_to_dosage_norm(code: str) -> float:
#     """
#     Your major/minor mapping makes the major allele '0'.
#     Dosage = count of non-zero digits in the 2-char code; normalize to [0,1] by /2.
#     '00'->0.0, '01'/'10'->0.5, '11'/'12'/'21'/'22'->1.0, missing -> np.nan.
#     """
#     s = str(code)
#     if s in ('-1','99') or len(s) != 2:
#         return np.nan
#     return ((s[0] != '0') + (s[1] != '0')) / PLOIDY

def dosage_from_major(code: str, maj_digit: str, ploidy: int = PLOIDY) -> float:
    s = str(code)
    if s in ('-1','99') or len(s) != 2: 
        return np.nan
    # count how many of the two alleles are != major
    non_major = (s[0] != maj_digit) + (s[1] != maj_digit)
    return non_major / float(ploidy)

def build_ld_score_row(major_minor_encoded_T: pd.DataFrame,
                       haplo_row: np.ndarray,
                       sample_cols: list,
                       window:int=5) -> np.ndarray:
    """
    For each SNP, compute local in-block LD score = mean r^2 with up to +/-window nearest
    SNPs within the same block (across all samples). Missing values ignored pairwise.
    Returns array in [0,1] (np.float32). If block_id==0 or insufficient neighbors -> 0.
    """
    # Build dosage matrix [n_snps x n_samples]
    D = np.empty((major_minor_encoded_T.shape[0], len(sample_cols)), dtype=np.float32)
    for j, sname in enumerate(sample_cols):
        D[:, j] = series_codes_to_dosage_norm(major_minor_encoded_T[sname].values)
        #         # snp order must match major_table.index
        # dosage_norm = np.array([dosage_from_major(canonical_pair(x), maj_digit) 
        #                         for x, maj_digit in zip(sample_series, major_table.values)],
        #                     dtype=np.float32)


    n = D.shape[0]
    out = np.zeros(n, dtype=np.float32)
    # precompute indices per block
    block_to_idxs = defaultdict(list)
    for i, b in enumerate(haplo_row):
        if b > 0: block_to_idxs[int(b)].append(i)

    def r2_pair(a, b):
        # pairwise r^2 with NaNs ignored
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 3: return np.nan
        aa, bb = a[m], b[m]
        va = aa.var()
        vb = bb.var()
        if va == 0 or vb == 0: return 0.0
        r = np.corrcoef(aa, bb)[0,1]
        return float(max(0.0, min(1.0, r*r)))

    for b, idxs in block_to_idxs.items():
        if len(idxs) == 0: continue
        for k, i in enumerate(idxs):
            left  = max(0, k - window)
            right = min(len(idxs)-1, k + window)
            nbrs = [idxs[u] for u in range(left, right+1) if idxs[u] != i]
            if not nbrs:
                out[i] = 0.0
                continue
            vals = [r2_pair(D[i,:], D[j,:]) for j in nbrs]
            vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
            out[i] = float(np.mean(vals)) if vals else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def build_blocklen_row_kb(haplotype_blocks: pd.DataFrame, haplo_row: np.ndarray) -> np.ndarray:
    """
    Returns array len(haplo_row) with length of the block (in kb) per SNP; 0 if block_id==0.
    """
    # block_id starts at 1 in your assigner
    lens = haplotype_blocks['KB'].to_numpy(dtype=np.float32)  # index 0..(nblocks-1)
    out = np.zeros_like(haplo_row, dtype=np.float32)
    mask = (haplo_row > 0)
    out[mask] = lens[haplo_row[mask] - 1]  # block_id -> idx
    return out

def build_adaptive_scale_for_chr(chr_len:int,
                                 chr_loci_bp: np.ndarray,
                                 desired_width_px:int,
                                 bins:int=200,
                                 alpha:float=0.8,
                                 min_fold:float=0.5,
                                 max_fold:float=2.0):
    """
    Piece-wise linear map: divide [0, chr_len] into 'bins' segments; compute SNP density in each,
    allocate pixels to segments ~ density^alpha, clipped to [min_fold, max_fold], then renormalize
    so sum(px_segments) == desired_width_px. Returns (bp_edges, px_edges), both length bins+1.
    """
    if chr_len <= 0:  # safety
        return np.array([0, 1], dtype=np.int64), np.array([0, desired_width_px], dtype=np.int64)

    bp_edges = np.linspace(0, chr_len, bins+1, dtype=np.int64)
    # density per segment
    counts, _ = np.histogram(chr_loci_bp, bins=bp_edges)
    seg_bp = np.diff(bp_edges).astype(np.float64)
    # avoid zeros
    dens = (counts / np.maximum(seg_bp, 1.0))  # SNPs per bp
    med = np.median(dens[dens > 0]) if np.any(dens > 0) else 1.0
    weight = np.ones_like(dens) if med == 0 else np.clip((dens/med)**alpha, min_fold, max_fold)
    # convert to pixels
    px_raw = weight * seg_bp
    px_segments = (px_raw / px_raw.sum()) * desired_width_px
    # integer edges
    px_edges = np.concatenate([[0], np.cumsum(px_segments)]).astype(np.int64)
    px_edges[-1] = desired_width_px  # exact
    return bp_edges, px_edges

def normalize_desired_width(desired: int, tile_width: int) -> int:
    return ((int(desired) + tile_width - 1) // tile_width) * tile_width

def interp_bp_to_px(bp_edges: np.ndarray, px_edges: np.ndarray, locus_bp: int) -> int:
    """
    Map a bp coordinate to pixel using the piece-wise linear map.
    Always return a pixel in [0, px_edges[-1]-1].  (Never return the sentinel right edge.)
    """
    # Clamp locus to [bp_edges[0], bp_edges[-1]-1] so we never hit the final sentinel bin
    bp_last = int(bp_edges[-1])
    if locus_bp >= bp_last:
        locus_bp = bp_last - 1
    elif locus_bp < int(bp_edges[0]):
        locus_bp = int(bp_edges[0])

    s = np.searchsorted(bp_edges, locus_bp, side='right') - 1
    s = max(0, min(s, len(px_edges) - 2))
    bp0, bp1 = int(bp_edges[s]), int(bp_edges[s + 1])
    px0, px1 = int(px_edges[s]), int(px_edges[s + 1])

    if bp1 == bp0:
        x = px0
    else:
        frac = (locus_bp - bp0) / float(bp1 - bp0)
        x = int(px0 + frac * (px1 - px0))

    # Never return the right-edge sentinel
    return max(0, min(x, int(px_edges[-1]) - 1))


def rasterize_track_per_chr(track_by_chr, adapt_maps, total_rows, strip_h, tile_w, num_tiles,
                            dtype=np.float32, fill=0.0):
    H = total_rows * strip_h
    rasters = [np.full((H, tile_w), fill_value=fill, dtype=dtype) for _ in range(num_tiles)]
    chr_to_row = {chrom: idx for idx, chrom in enumerate(sorted(adapt_maps.keys(), key=_chr_sort_key))}
    for chrom, intervals in track_by_chr.items():
        if chrom not in adapt_maps: continue
        row = chr_to_row[chrom]; y0, y1 = row*strip_h, (row+1)*strip_h
        be, pe = adapt_maps[chrom]
        for w0, w1, val in intervals:
            x1 = interp_bp_to_px(be, pe, int(w0))
            x2 = min(interp_bp_to_px(be, pe, int(w1 - 1)) + 1, int(pe[-1]))
            for t in range(num_tiles):
                tx0, tx1 = t*tile_w, (t+1)*tile_w - 1
                s, e = max(x1, tx0), min(x2-1, tx1)
                if s > e: continue
                rs, re = s-tx0, e-tx0
                rasters[t][y0:y1, rs:re+1] = val
    return rasters

def group_blocks_by_label(haplotype_blocks: pd.DataFrame, id_normalizer):
    """
    Returns dict: label -> list[(bp1, bp2, block_id, kb)] where label is like 'A01','C03', etc.
    block_id numbering matches add_haplotype_block_info (enumerate order).
    """
    blocks_by_label = defaultdict(list)
    for block_id, r in enumerate(haplotype_blocks.itertuples(index=False), start=1):
        lab = str(id_normalizer(str(r.CHR)))             # map numeric CHR -> 'Axx'/'Cxx' (or '1','2' in diploids)
        kb  = float(getattr(r, 'KB', (int(r.BP2) - int(r.BP1)) / 1000.0))
        blocks_by_label[lab].append((int(r.BP1), int(r.BP2), int(block_id), kb))
    return blocks_by_label

def group_homology_by_label(homology_df: pd.DataFrame, id_normalizer):
    """
    Returns dict: label -> list[(bp1, bp2, hom_group_id, hom_pair_id)] for homology spans.
    """
    if homology_df is None or homology_df.empty:
        return {}
    required = {"Chromosome", "BP1", "BP2", "hom_group_id", "hom_pair_id"}
    missing = required - set(homology_df.columns)
    if missing:
        logging.warning(f"homology_df missing columns {missing}; skipping homology raster.")
        return {}
    hom_by_label = defaultdict(list)
    for r in homology_df.itertuples(index=False):
        chrom = str(getattr(r, "Chromosome"))
        lab = str(id_normalizer(chrom)) if id_normalizer else chrom
        bp1 = int(getattr(r, "BP1"))
        bp2 = int(getattr(r, "BP2"))
        gid = int(float(getattr(r, "hom_group_id")))
        pid = int(float(getattr(r, "hom_pair_id")))
        hom_by_label[lab].append((bp1, bp2, gid, pid))
    return hom_by_label
    
# Lexicographic sort that places `A01..A10, C01..C09` before numeric labels.  
def _chr_sort_key(c):
    s = str(c).strip()
    if not s:
        return (3, "", 0, "")

    tokens = [s]
    tokens.extend(tok for tok in re.split(r"[._:/\\-]+", s) if tok)
    for tok in tokens:
        m = re.fullmatch(r"(?i)(?:chr|chromosome)?[_-]*([A-Za-z])[_-]*0*([0-9]+)", tok)
        if m:
            return (0, m.group(1).upper(), int(m.group(2)), s.lower())

    direct = _extract_chromosome_id(s)
    if direct is not None:
        return (1, "", int(direct), s.lower())

    return (2, s.lower(), 0, s.lower())

# mix_toward_white(base_rgb_uint8, t)` and `darken_by_weight(base_rgb_uint8, t)

def mix_toward_white(base_rgb_uint8, t):
    """
    t in [0,1] from maf_to_mix_weight(): t=1 keeps the color, t=0 pushes to white.
    """
    base = base_rgb_uint8.astype(np.float32)
    new  = base * t + 255.0 * (1.0 - t)
    return np.clip(new, 0, 255).astype(np.uint8)

def darken_by_weight(base_rgb_uint8, t):
    return np.clip(base_rgb_uint8.astype(np.float32) * (0.5 + 0.5*t), 0, 255).astype(np.uint8)

# deterministic pseudo-random colors for haplotype blocks.
def get_color_for_block(block_id):
    """
    Return an RGB color for a given haplotype block id.
    Block id 0 is reserved for "no haplotype block" and returns white.
    For non-zero block ids, we generate a pseudo-random but deterministic color.
    """
    if block_id == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    # Use simple arithmetic with prime multipliers to mix the block id into RGB components.
    r = (block_id * 37) % 256
    g = (block_id * 73) % 256
    b = (block_id * 109) % 256
    return np.array([r, g, b], dtype=np.uint8)

def get_or_build_color_lookup(max_block_id: int, out_dir: str, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "block_color_lookup.npy")
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=False)
        # arr shape (N,3) uint8 with arr[0] reserved for white
        if arr.shape[0] >= (max_block_id + 1):
            return {i: arr[i] for i in range(arr.shape[0])}
    # build new
    lookup = generate_unique_color_lookup(max_block_id + 1, seed=seed)
    # pack into array
    N = max_block_id + 1
    arr = np.zeros((N+1, 3), dtype=np.uint8)
    for i in range(N+1):
        arr[i] = lookup.get(i, np.array([200,200,200], dtype=np.uint8))
    np.save(path, arr)
    return {i: arr[i] for i in range(arr.shape[0])}

def generate_unique_color_lookup(n, seed=42):
    """
    Generate a lookup dictionary that maps block ids 1..n to unique RGB colors.
    Block id 0 is reserved for white.
    """
    random.seed(seed)
    unique_colors = set()
    lookup = {}
    while len(unique_colors) < n:
        # Generate a random color as a tuple of three integers in [0, 255]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color not in unique_colors:
            unique_colors.add(color)
    # Create a sorted list to have deterministic ordering
    unique_colors = sorted(list(unique_colors))
    # Map block ids 1..n to these colors
    for i, color in enumerate(unique_colors, start=1):
        lookup[i] = np.array(color, dtype=np.uint8)
    # Ensure block id 0 is white
    lookup[0] = np.array([255, 255, 255], dtype=np.uint8)
    return lookup

color_lookup = generate_unique_color_lookup(2700)
# print("Color for block 1:", color_lookup[1])
# print("Color for block 0 (no block):", color_lookup[0])

def get_color_for_block_lookup(block_id, color_lookup):
    return color_lookup.get(block_id, np.array([200, 200, 200], dtype=np.uint8))


# 1D strip for a single sample (downsample if >65,535 SNPs)
def save_sample_as_image(sample_data, sample_name, folder, height=1):
    """
    Save a 1D color-coded image from sample_data.
    sample_data should be a 1D array of encoded allele values.
    """
    image_data = np.array(sample_data)
    total_snps = len(image_data)
    max_width = 65535
    width = min(total_snps, max_width)
    
    if total_snps > max_width:
        indices = np.linspace(0, total_snps - 1, max_width).astype(int)
        image_data = image_data[indices]
        downsampled_snps = total_snps - max_width
        logging.info(f"{downsampled_snps} SNPs downsampled for sample {sample_name} (from {total_snps} to {max_width}).")
    else:
        logging.info(f"No downsampling required for sample {sample_name} ({total_snps} SNPs).")

    color_matrix = np.array([color_map.get(canonical_pair(allele), [200, 200, 200])
                         for allele in image_data])
    color_matrix = color_matrix.reshape(1, -1, 3)
    
    plt.figure(figsize=(width/1000, height))
    plt.imshow(color_matrix, aspect='auto')
    plt.axis('off')
    image_path = os.path.join(folder, f'{sample_name}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    logging.info(f"1D image saved for {sample_name} at {image_path}")

# Per-chromosome variants:

def save_sample_as_image_chr(sample_data, sample_name, folder, chr_id_row, locus_row, dpi=100):
    """
    Generate and save images per chromosome without SNP overlaps.
    """
    for chr_name, chr_len in chr_info.items():
        logging.info(f"Processing chr {chr_name} for {sample_name}")

        chr_snps = [(locus, allele) for locus, allele, chr_id in zip(locus_row, sample_data, chr_id_row) if str(chr_id) == str(chr_name)]

        if chr_snps:
            out_folder = os.path.join(folder, sample_name, f'chr_{chr_name}')
            save_sample_as_image_for_chr_helper(chr_snps, sample_name, out_folder, chr_name, chr_len, dpi=dpi)
        else:
            logging.info(f"No SNPs found for chr {chr_name} in {sample_name}")

def save_sample_as_image_for_chr_helper(chr_snps, sample_name, output_folder, chr_name, chr_len, dpi=100):
    """
    Helper: Save a single chromosome image with no SNP overlap.
    """
    # Ensure each SNP is represented distinctly by setting pixel width to SNP count
    num_snps = len(chr_snps)
    fig_width_px = num_snps
    fig_width_in = fig_width_px / dpi

    strip_height = 0.5  # in inches

    fig, ax = plt.subplots(figsize=(fig_width_in, strip_height), dpi=dpi)
    ax.add_patch(plt.Rectangle((0, 0), fig_width_px, 1, color='white', ec='black', lw=1))

    # Rescale loci to pixels explicitly
    loci = np.array([locus for locus, _ in chr_snps])
    loci_scaled = (loci / chr_len) * fig_width_px

    for locus_px, (_, allele_code) in zip(loci_scaled, chr_snps):
        color = np.array(color_map.get(canonical_pair(allele_code), [200, 200, 200])) / 255
        ax.plot([locus_px, locus_px], [0, 1], color=color, lw=1)

    ax.set_xlim(0, fig_width_px)
    ax.set_ylim(0, 1)
    ax.axis('off')

    os.makedirs(output_folder, exist_ok=True)
    image_path = os.path.join(output_folder, f'{sample_name}_chr_{chr_name}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

    logging.info(f"Chromosome {chr_name} image saved for {sample_name} at {image_path}")

def build_adaptive_maps_for_all(local_chr_info: dict, labels: np.ndarray, locus_row: np.ndarray,
                                desired_max_width_px: int,
                                adaptive_bins=200, adaptive_alpha=0.8, min_fold=0.5, max_fold=2.0):
    """
    One-stop builder used by ALL writers so bp->px mapping is consistent.
    Returns dict chrom -> (bp_edges, px_edges) such that the longest chromosome maps to desired_max_width_px.
    """

    sorted_chr = sorted(local_chr_info.keys(), key=_chr_sort_key)
    base_scale = desired_max_width_px / float(max(local_chr_info.values()))
    adapt_maps = {}
    for chrom in sorted_chr:
        chr_len = int(local_chr_info[chrom])
        target_w = int(chr_len * base_scale)
        loci_this = locus_row[labels == chrom]
        adapt_maps[chrom] = build_adaptive_scale_for_chr(
            chr_len, loci_this, target_w,
            bins=adaptive_bins, alpha=adaptive_alpha, min_fold=min_fold, max_fold=max_fold
        )
    return adapt_maps

def _rolling_mean_1d(x: np.ndarray, win: int) -> np.ndarray:
    """
    Length-preserving rolling mean with odd window, edge-padded.
    Returns an array the same length as x.
    """
    if win <= 1:
        return x
    win = int(win) | 1  # force odd
    k = win // 2
    xp = np.pad(np.asarray(x, dtype=float), (k, k), mode='edge')
    c  = np.cumsum(np.r_[0.0, xp])
    out = (c[win:] - c[:-win]) / float(win)  # length == len(x)
    return out.astype(np.asarray(x).dtype, copy=False)


def build_snp_density_rasters(adapt_maps: dict, labels: np.ndarray, locus_row: np.ndarray,
                              total_chrs: int, strip_height: int,
                              tile_width: int, desired_max_width_px: int,
                              smooth_px: int = 101):
    """
    Per chromosome, count SNPs per pixel column using adapt_maps; smooth; normalize to [0,1].
    Returns: list of per-tile 2D float32 arrays (full_h x tile_width).
    """
    desired_max_width_px = normalize_desired_width(desired_max_width_px, tile_width)

    sorted_chr = list(adapt_maps.keys())
    chr_to_row = {chrom: idx for idx, chrom in enumerate(sorted_chr)}
    full_h = total_chrs * strip_height
    num_tiles = math.ceil(desired_max_width_px / tile_width)
    # global pixel grid 0..W-1
    density = np.zeros((total_chrs, desired_max_width_px), dtype=np.float32)
    for chrom in sorted_chr:
        be, pe = adapt_maps[chrom]
        row = chr_to_row[chrom]
        loci = locus_row[labels == chrom]
        xs = np.array([interp_bp_to_px(be, pe, int(bp)) for bp in loci], dtype=np.int32)
        # in build_snp_density_rasters, after xs := mapped pixels
        xs = xs[(xs >= 0) & (xs < desired_max_width_px)]

        if xs.size:
            np.add.at(density[row], xs, 1.0)

    # smooth & normalize per row
    for r in range(total_chrs):
        density[r] = _rolling_mean_1d(density[r], smooth_px)
        mx = density[r].max()
        if mx > 0:
            density[r] /= mx

    # slice into tiles and broadcast across strip height
    rasters = []
    for t in range(num_tiles):
        x0, x1 = t*tile_width, min((t+1)*tile_width, desired_max_width_px)
        tile = np.zeros((full_h, tile_width), dtype=np.float32)
        for r in range(total_chrs):
            row_band = density[r, x0:x1]
            y0, y1 = r*strip_height, (r+1)*strip_height
            # broadcast
            tile[y0:y1, :x1-x0] = row_band[np.newaxis, :]
        rasters.append(tile)
    return rasters

def save_all_chromosomes_as_image(sample_data, sample_name, folder, chr_id_row, locus_row, dpi=100, max_channels=3):
    """
    Save a legacy combined chromosome PNG as a true RGB image.
    Overlapping SNPs in the same row/pixel are averaged up to `max_channels`
    observations; additional collisions are dropped with a summary warning.
    """
    local_chr_info, id_norm = build_chr_info_for_mode()
    row_labels = sorted(local_chr_info.keys(), key=_chr_sort_key)
    row_index = {lab: idx for idx, lab in enumerate(row_labels)}

    strip_height = 10
    total_chrs = len(row_labels)
    desired_max_width_px = 5000
    fig_width_px = max(1, int(desired_max_width_px))
    max_channels = max(1, int(max_channels))

    longest_chr_len = max(local_chr_info.values())
    scale_factor = fig_width_px / float(longest_chr_len)

    labels = np.array([id_norm(c) for c in chr_id_row], dtype=object)
    rgb_sum = np.zeros((total_chrs * strip_height, fig_width_px, 3), dtype=np.float32)
    pixel_counts = np.zeros((total_chrs, fig_width_px), dtype=np.uint16)
    overflow_count = 0

    for lab, locus_bp, allele in zip(labels, locus_row, sample_data):
        row_idx = row_index.get(str(lab))
        if row_idx is None:
            continue
        locus_px = min(max(int(int(locus_bp) * scale_factor), 0), fig_width_px - 1)
        count = int(pixel_counts[row_idx, locus_px])
        if count >= max_channels:
            overflow_count += 1
            continue
        y0, y1 = row_idx * strip_height, (row_idx + 1) * strip_height
        color = np.array(color_map.get(canonical_pair(str(allele)), [200, 200, 200]), dtype=np.float32)
        rgb_sum[y0:y1, locus_px, :] += color
        pixel_counts[row_idx, locus_px] = count + 1

    image = np.full((total_chrs * strip_height, fig_width_px, 3), 255, dtype=np.uint8)
    for row_idx in range(total_chrs):
        y0, y1 = row_idx * strip_height, (row_idx + 1) * strip_height
        counts = pixel_counts[row_idx].astype(np.float32)
        valid = counts > 0
        if not np.any(valid):
            continue
        denom = counts[valid][None, :, None]
        avg = np.clip(rgb_sum[y0:y1, valid, :] / denom, 0.0, 255.0).astype(np.uint8)
        image[y0:y1, valid, :] = avg

    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, f"{sample_name}_chromosomes.png")
    plt.imsave(image_path, image)
    if overflow_count > 0:
        logging.warning(
            f"{overflow_count} SNPs exceeded max_channels={max_channels} in save_all_chromosomes_as_image "
            f"for {sample_name} and were omitted."
        )
    logging.info(f"Saved combined chromosome image for {sample_name} at {image_path}")
    
# Writes **RGB tiles** (PNG + NPZ sidecar) with N-way collision stacking per column and **per-tile event logs** `(row, x, locus_bp, allele)` to enable auditing.

def sanity_checks_after_render(
    sample: str, snp_npz_dir: str, hap_npz_dir: str,
    expected_snps: int, use_subgenomes: bool,
    overflow_warn: int = 10_000
):
    # Event completeness (you already have audit_sample_events)
    ok_snps = audit_sample_events(sample, snp_npz_dir, expected_snps)
    ok_haps = audit_sample_events(sample, hap_npz_dir, expected_snps)
    if not (ok_snps and ok_haps):
        logging.warning(f"[SANITY] {sample}: event completeness failed (snp={ok_snps}, hap={ok_haps}).")

    # Overflow budget
    for root in [snp_npz_dir, hap_npz_dir]:
        paths = sorted(glob.glob(os.path.join(root, f"{sample}_tile_*.npz")))
        total_overflow = 0
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                if "overflow" in z:
                    total_overflow += int(np.asarray(z["overflow"]).sum())
        if total_overflow > overflow_warn:
            logging.warning(f"[SANITY] {sample}@{os.path.basename(root)} overflow={total_overflow} > {overflow_warn}")

    # Stray chromosomes check is best done up-front (after label normalization):
    # done implicitly when row_index lookup fails (we skip), but you can also count skipped loci if you want.

    # Coordinate parity: since we now write scale tables into both SNP and hap NPZs, they are consistent by construction.


def save_all_chromosomes_as_tiled_images(
    sample_data, sample_name, folder, chr_id_row, locus_row,
    dpi=100, tile_width=16384,
    initial_and_desired_max_width_px=DESIRED_MAX_WIDTH_PX,
    strip_height=10, chr_info_override=None, id_normalizer=normalize_chr_id,
    max_width_cap_px=DESIRED_MAX_WIDTH_PX, collision_depth=COLLISION_DEPTH, skip_empty_tiles=False,
    write_index_csv=True, save_npz=True, maf_row=None,  # maf_row kept for parity but unused here
    adapt_maps: dict | None = None
):
    """
    RGB tiles for SNPs using the SAME bp->px mapping as feature tiles (if adapt_maps provided).
    Otherwise falls back to linear scaling (legacy).
    """
    global chr_info, color_map
    desired_w = normalize_desired_width(initial_and_desired_max_width_px, tile_width)
    local_chr = chr_info_override if chr_info_override is not None else chr_info
    row_labels = sorted((str(k) for k in local_chr.keys()), key=_chr_sort_key)
    row_index = {lab: i for i, lab in enumerate(row_labels)}
    n_rows = len(row_labels)
    full_h = n_rows * strip_height
    # desired_w = int(initial_and_desired_max_width_px)
    tiles = (desired_w + tile_width - 1) // tile_width

    # normalize labels
    chr_labels = np.array([id_normalizer(c) for c in chr_id_row], dtype=object)

    # mapping
    if adapt_maps is None:
        longest_bp = int(max(local_chr.values()))
        scale = desired_w / float(longest_bp)
        def _map_x(label, bp):
            return min(max(int(bp * scale), 0), desired_w - 1)
        scale_payload = {}
    else:
        def _map_x(label, bp):
            be, pe = adapt_maps[str(label)]
            return min(max(int(interp_bp_to_px(be, pe, int(bp))), 0), desired_w - 1)
        # serialize maps once per tile
        scale_payload = {}
        for chrom in row_labels:
            be, pe = adapt_maps[chrom]
            scale_payload[f"scale_bp_edges__{chrom}"] = be.astype(np.int64)
            scale_payload[f"scale_px_edges__{chrom}"] = pe.astype(np.int64)

    # allocate
    # Adaptive per-tile collision depth: count max hits per pixel
    depths = [1 for _ in range(tiles)]
    depth_counter = defaultdict(int)
    for lab, pos_bp in zip(chr_labels, locus_row):
        row = row_index.get(str(lab))
        if row is None:
            continue
        x_pix = _map_x(lab, pos_bp)
        t = x_pix // tile_width
        x_in = x_pix % tile_width
        depth_counter[(t, row, x_in)] += 1
    for t in range(tiles):
        max_hit = max((cnt for (tt, _, _), cnt in depth_counter.items() if tt == t), default=1)
        depths[t] = max(1, min(int(max_hit), collision_depth))

    tile_layers = [[np.full((full_h, tile_width, 3), 255, dtype=np.uint8) for _ in range(depths[i])]
                   for i in range(tiles)]
    pixel_depths = [np.zeros((n_rows, tile_width), dtype=np.uint16) for _ in range(tiles)]
    overflow     = [np.zeros((n_rows, tile_width), dtype=np.uint16) for _ in range(tiles)]
    events       = [[] for _ in range(tiles)]

    for lab, pos_bp, code in zip(chr_labels, locus_row, map(str, sample_data)):
        row = row_index.get(str(lab))
        if row is None: 
            continue
        x_pix = _map_x(lab, pos_bp)
        t = x_pix // tile_width
        x_in = x_pix % tile_width
        y0, y1 = row*strip_height, (row+1)*strip_height
        events[t].append((row, x_in, int(pos_bp), str(code)))
        base = np.array(color_map.get(canonical_pair(str(code)), [200, 200, 200]), dtype=np.uint8)
        d = int(pixel_depths[t][row, x_in])
        if d < depths[t]:
            tile_layers[t][d][y0:y1, x_in, :] = base
            pixel_depths[t][row, x_in] = d + 1
        else:
            overflow[t][row, x_in] = min(65535, overflow[t][row, x_in] + 1)

    os.makedirs(folder, exist_ok=True)
    index_rows = []
    for i in range(tiles):
        comp = np.full((full_h, tile_width, 3), 255, dtype=np.uint8)
        has_content = False
        for d in range(depths[i]):
            L = tile_layers[i][d]
            mask = (L != 255).any(axis=2)
            if mask.any():
                comp[mask] = L[mask]
                has_content = True
        png_path = os.path.join(folder, f"{sample_name}_tile_{i}.png")
        plt.imsave(png_path, comp, dpi=dpi)

        if save_npz:
            ev = np.array(events[i],
                          dtype=[('row', np.int32), ('x', np.int32),
                                 ('locus_bp', np.int64), ('allele', 'U4')])
            meta = np.array([full_h, tile_width, strip_height, depths[i], desired_w], dtype=np.int64)
            extras = dict(overflow=overflow[i], **scale_payload)
            npz_path = os.path.join(folder, f"{sample_name}_tile_{i}.npz")
            write_and_verify_npz(npz_path,
                                 layers=np.concatenate(tile_layers[i], axis=2),
                                 events=ev, meta=meta, **extras)

        x0, x1 = i*tile_width, min((i+1)*tile_width, desired_w) - 1
        index_rows.append({"sample": sample_name, "tile_idx": i, "x_start_px": x0,
                           "x_end_px": x1, "has_content": int(has_content or bool(events[i]))})

    if write_index_csv:
        pd.DataFrame(index_rows).to_csv(os.path.join(folder, f"{sample_name}_tile_index.csv"), index=False)
    return desired_w

# Writes **float32 feature tiles** for one sample:
#   - Ch1: dosage_norm `[0,1]`
#   - Ch2: quality (`maf`/`callrate`/`missing`)
#   - optional: density, TE flag, subgenome channels
#   plus overflow maps, collision depth, and **adaptive scale tables** per chromosome to enable exact bp<->px reconstruction.
def save_tiled_features_snp(
    sample_codes: np.ndarray, sample_name: str, folder: str,
    chr_id_row: np.ndarray, locus_row: np.ndarray,
    maf_row: np.ndarray,                      # can be None if quality_channel != 'maf'
    subgenome_labels: dict,                   # kept for backward compatibility (unused here)
    callrate_row: np.ndarray | None = None,   # dataset-wide per-SNP call-rate (preferred for quality_channel='callrate')
    dpi=100, tile_width=16384, initial_desired_max_width_px=DESIRED_MAX_WIDTH_PX,
    strip_height=10, chr_info_override=None, id_normalizer=normalize_chr_id,
    max_width_cap_px=DESIRED_MAX_WIDTH_PX, collision_depth=COLLISION_DEPTH, skip_empty_tiles=False,
    write_index_csv=True, save_npz=True,
    include_sg_channel=True,                  # now means "include N-hot SG channels if config provided"
    adaptive_bins=300, adaptive_alpha=0.6, min_fold=0.5, max_fold=2.0,
    quality_channel: str = SNP_QUALITY_CHANNEL,  # 'maf'|'callrate'|'missing'
    include_density_channel: bool = INCLUDE_DENSITY_CHANNEL,
    density_smooth_px: int = 101,
    sg_config: dict[str,list[str]] = SUBGENOME_CHRS,
    te_is_te: np.ndarray | None = None
):
    global chr_info
    local_chr_info = chr_info_override if chr_info_override is not None else chr_info
    sorted_chr = sorted(local_chr_info.keys(), key=_chr_sort_key)
    chr_to_row = {chrom: idx for idx, chrom in enumerate(sorted_chr)}
    total_chrs = len(local_chr_info)
    desired_max_width_px = normalize_desired_width(initial_desired_max_width_px, tile_width)
    # desired_max_width_px = int(initial_desired_max_width_px)

    # normalize labels once
    labels = np.array([id_normalizer(c) for c in chr_id_row], dtype=object)

    # --- features derived per SNP
    dosage_norm = series_codes_to_dosage_norm(sample_codes).astype(np.float32)  # 0..1

    # Quality channel
    qc = None
    if quality_channel == 'maf':
        if maf_row is None:
            raise ValueError("quality_channel='maf' requires maf_row.")
        qc = np.clip((maf_row * 2.0).astype(np.float32), 0.0, 1.0)
    elif quality_channel == 'callrate':
        if callrate_row is not None and len(callrate_row) == len(sample_codes):
            qc = np.asarray(callrate_row, dtype=np.float32)
        else:
            logging.warning("callrate_row missing or length mismatch; falling back to per-sample call proxy.")
    elif quality_channel == 'missing':
        if callrate_row is not None and len(callrate_row) == len(sample_codes):
            qc = 1.0 - np.asarray(callrate_row, dtype=np.float32)
        else:
            logging.warning("callrate_row missing or length mismatch; falling back to per-sample missing proxy.")

    if qc is None:
        # Minimal leak-safe default derived from this sample only.
        valid = np.array([
            (isinstance(x, str) and len(x) == 2 and x[0] in '0123' and x[1] in '0123')
            for x in sample_codes
        ], dtype=np.float32)
        qc = valid if quality_channel != 'missing' else 1.0 - valid
        qc = qc.astype(np.float32)

    # Build unified adaptive maps and optional density rasters
    adapt_maps = build_adaptive_maps_for_all(local_chr_info, labels, locus_row,
                                             desired_max_width_px,
                                             adaptive_bins=adaptive_bins, adaptive_alpha=adaptive_alpha,
                                             min_fold=min_fold, max_fold=max_fold)

    num_tiles = math.ceil(desired_max_width_px / tile_width)
    density_rasters = None
    if include_density_channel:
        density_rasters = build_snp_density_rasters(adapt_maps, labels, locus_row,
                                                    total_chrs, strip_height,
                                                    tile_width, desired_max_width_px,
                                                    smooth_px=density_smooth_px)

    # SG channels (N-hot per row)
    sg_names, row_to_onehot = build_sg_encoder(sorted_chr, sg_config if include_sg_channel else {})

    include_te_channel = te_is_te is not None and len(te_is_te) == len(locus_row)

    # Feature channel list
    feat_names = ['dosage_norm', f'quality_{quality_channel}']
    if include_density_channel:
        feat_names.append('snp_density')
    if include_te_channel:
        feat_names.append('is_te')
    feat_names.extend([f'sg_{n}' for n in sg_names])
    F = len(feat_names)

    # allocate
    full_height = total_chrs * strip_height
    # Adaptive depth per tile: count max hits per pixel
    depths = [1 for _ in range(num_tiles)]
    depth_counter = defaultdict(int)
    for lab, bp in zip(labels, locus_row):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None:
            continue
        be, pe = adapt_maps[str(lab)]
        x_in_row = interp_bp_to_px(be, pe, int(bp))
        x_pixel = min(max(int(x_in_row), 0), desired_max_width_px - 1)
        t_idx = x_pixel // tile_width
        x_in = x_pixel % tile_width
        depth_counter[(t_idx, row_idx, x_in)] += 1
    for t in range(num_tiles):
        max_hit = max((cnt for (tt, _, _), cnt in depth_counter.items() if tt == t), default=1)
        depths[t] = max(1, min(int(max_hit), collision_depth))

    tile_layers = [
        [np.zeros((full_height, tile_width, F), dtype=np.float32) for _ in range(depths[i])]
        for i in range(num_tiles)
    ]
    pixel_depths = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    overflow     = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    grouped      = [defaultdict(list) for _ in depths]
    events       = [[] for _ in depths]

    # queue events deterministically
    for idx, (lab, bp, dos, qv) in enumerate(zip(labels, locus_row, dosage_norm, qc)):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None: 
            continue
        be, pe = adapt_maps[str(lab)]
        x_in_row = interp_bp_to_px(be, pe, int(bp))
        x_pixel  = min(max(int(x_in_row), 0), desired_max_width_px - 1)
        t_idx    = x_pixel // tile_width
        x_in     = x_pixel % tile_width

        sg_vec = row_to_onehot(str(lab))
        feat = [dos if not np.isnan(dos) else 0.0, qv if not np.isnan(qv) else 0.0]
        if include_density_channel:
            feat.append(0.0)  # placeholder, we will fill from raster after grouping
        if include_te_channel:
            feat.append(float(te_is_te[idx]))
        if sg_vec.size:
            feat.extend(sg_vec.tolist())

        grouped[t_idx][(row_idx, x_in)].append((bp, feat))
        events[t_idx].append((row_idx, x_in, int(bp), float(dos)))

    # draw (and then overlay density)
    for t in range(num_tiles):
        for (row_idx, x), lst in grouped[t].items():
            lst.sort(key=lambda z: z[0])  # by position
            y0, y1 = row_idx*strip_height, (row_idx+1)*strip_height
            depth_used = 0
            for _, feat in lst:
                if depth_used >= depths[t]:
                    overflow[t][row_idx, x] += 1
                else:
                    vec = np.asarray(feat, dtype=np.float32)
                    tile_layers[t][depth_used][y0:y1, x, :] = vec
                    pixel_depths[t][row_idx, x] = depth_used + 1
                    depth_used += 1

        # fill density channel across the tile after the SNPs are placed
        if include_density_channel and density_rasters is not None:
            dens_idx = feat_names.index('snp_density')
            for d in range(depths[t]):
                tile_layers[t][d][:, :, dens_idx] = density_rasters[t]

    # write tiles
    os.makedirs(folder, exist_ok=True)
    index_rows = []
    for i in range(num_tiles):
        layers_stacked = np.concatenate(tile_layers[i], axis=2)
        # scale payload
        scale_payload = {}
        for chrom in sorted_chr:
            be, pe = adapt_maps[chrom]
            scale_payload[f"scale_bp_edges__{chrom}"] = be.astype(np.int64)
            scale_payload[f"scale_px_edges__{chrom}"] = pe.astype(np.int64)

        ev = np.array(events[i], dtype=[('row', np.int32), ('x', np.int32),
                                        ('locus_bp', np.int64), ('dosage_norm', np.float32)])
        if save_npz:
            meta = np.array([full_height, tile_width, strip_height, collision_depth,
                             desired_max_width_px, F], dtype=np.int64)
            ch_names = np.frombuffer(",".join(feat_names).encode('ascii'), dtype=np.uint8)
            npz_path = os.path.join(folder, f"{sample_name}_feat_tile_{i}.npz")
            write_and_verify_npz(
                npz_path,
                layers=layers_stacked,
                overflow=overflow[i],
                events=ev,
                meta=np.array([full_height, tile_width, strip_height, depths[i],
                               desired_max_width_px, F], dtype=np.int64),
                channel_names_bytes=ch_names,
                **scale_payload
            )

        x_start = i*tile_width
        x_end   = (i+1)*tile_width - 1
        index_rows.append({"sample": sample_name, "tile_idx": i,
                           "x_start_px": x_start, "x_end_px": x_end,
                           "has_content": int(len(events[i]) > 0)})

    if write_index_csv:
        idx_path = os.path.join(folder, f"{sample_name}_snp_feat_tile_index.csv")
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        logging.info(f"[{sample_name}] wrote SNP feature tile index: {idx_path}")

    return desired_max_width_px


def _quality_vector_from_codes(sample_codes: np.ndarray,
                               maf_row: np.ndarray | None,
                               quality_channel: str,
                               callrate_row: np.ndarray | None = None) -> np.ndarray:
    """
    Build a per-SNP quality vector aligned to sample_codes.
    Mirrors the logic used in save_tiled_features_snp, but kept standalone so we
    can reuse it for the hierarchical tensor export.
    """
    if quality_channel == 'maf':
        if maf_row is None:
            raise ValueError("quality_channel='maf' requires maf_row.")
        return np.clip((maf_row * 2.0).astype(np.float32), 0.0, 1.0)

    if quality_channel == 'callrate' and callrate_row is not None and len(callrate_row) == len(sample_codes):
        return np.asarray(callrate_row, dtype=np.float32)
    if quality_channel == 'missing' and callrate_row is not None and len(callrate_row) == len(sample_codes):
        return 1.0 - np.asarray(callrate_row, dtype=np.float32)

    valid = np.array([
        (isinstance(x, str) and len(x) == 2 and x[0] in '0123' and x[1] in '0123')
        for x in sample_codes
    ], dtype=np.float32)
    return (valid if quality_channel != 'missing' else 1.0 - valid).astype(np.float32)



def _build_homology_lookup_tables(homology_df: pd.DataFrame | None, id_normalizer=normalize_chr_id) -> dict:
    """
    Convert homology_df with columns [Chromosome, BP1, BP2, hom_group_id, hom_pair_id]
    into per-chromosome lookup tables for fast interval queries.
    """
    if homology_df is None or homology_df.empty:
        return {}
    required = {"Chromosome", "BP1", "BP2", "hom_group_id", "hom_pair_id"}
    if not required.issubset(set(homology_df.columns)):
        missing = required - set(homology_df.columns)
        logging.warning(f"homology_df missing columns {missing}; skipping homology features.")
        return {}
    lookup = {}
    for chrom, sub in homology_df.groupby("Chromosome"):
        starts = sub["BP1"].to_numpy(dtype=np.int64)
        ends = sub["BP2"].to_numpy(dtype=np.int64)
        groups = sub["hom_group_id"].to_numpy(dtype=np.float32)
        pairs = sub["hom_pair_id"].to_numpy(dtype=np.float32)
        order = np.argsort(ends)
        chrom_norm = str(id_normalizer(str(chrom)))
        lookup[chrom_norm] = (
            starts[order],
            ends[order],
            groups[order],
            pairs[order],
        )
    return lookup


def _lookup_homology(lookup: dict, chrom: str, pos_bp: int) -> tuple[float, float]:
    """
    Returns (hom_group_id, hom_pair_id) for chrom/pos, or (nan, nan) if none.
    """
    data = lookup.get(str(chrom))
    if data is None:
        return (np.nan, np.nan)
    starts, ends, groups, pairs = data
    idx = np.searchsorted(ends, int(pos_bp), side="right")
    if idx >= len(ends) or int(pos_bp) < starts[idx]:
        return (np.nan, np.nan)
    return (float(groups[idx]), float(pairs[idx]))


def rolling_mean_std_1d(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean/std (length-preserving) with edge padding.
    x: 1D float array (caller should clean NaNs if needed)
    win: odd integer window size
    Returns (mean, std) arrays of same length as x.
    """
    if win <= 1:
        m = np.asarray(x, dtype=np.float32)
        return m, np.zeros_like(m, dtype=np.float32)

    win = int(win) | 1
    k = win // 2
    arr = np.asarray(x, dtype=np.float32)

    # edge pad
    xp = np.pad(arr, (k, k), mode="edge")
    c = np.cumsum(np.r_[0.0, xp.astype(np.float64)])
    mean = (c[win:] - c[:-win]) / float(win)
    mean = mean.astype(np.float32)

    # std = sqrt(E[x^2] - (E[x])^2)
    xp2 = xp.astype(np.float64) ** 2
    c2 = np.cumsum(np.r_[0.0, xp2])
    mean2 = (c2[win:] - c2[:-win]) / float(win)
    var = np.maximum(0.0, mean2 - mean.astype(np.float64) ** 2)
    std = np.sqrt(var).astype(np.float32)

    return mean, std



def sinusoidal_position_encoding(position: np.ndarray, d_model: int = POSITION_ENCODING_DIM) -> np.ndarray:
    """
    Standard sinusoidal encoding on absolute positions (bp). Returns [N x d_model].
    NaN positions are zeroed by caller.
    """
    if d_model % 2 != 0:
        raise ValueError("POSITION_ENCODING_DIM must be even.")
    pos = np.asarray(position, dtype=np.float64).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model))
    pe = np.zeros((pos.shape[0], d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe.astype(np.float32)


def _hashed_block_bits(block_ids: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic K-bit hash for block IDs (matches haplo feature tiles). Returns
    array shape (len(block_ids), k) of float32 0/1 bits; block_id<=0 -> all zeros.
    """
    K = int(max(0, k))
    out = np.zeros((len(block_ids), K), dtype=np.float32)
    if K == 0 or len(block_ids) == 0:
        return out
    bid = np.asarray(block_ids, dtype=np.int64)
    x = (np.uint64(bid) ^ np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)
    x ^= (x >> np.uint64(30))
    x *= np.uint64(0x94D049BB133111EB)
    for i in range(K):
        out[:, i] = ((x >> np.uint64(i)) & np.uint64(1)).astype(np.float32)
    out[bid <= 0, :] = 0.0
    return out


def build_tensor_layout_cache(
    chr_id_row: np.ndarray,
    locus_row: np.ndarray,
    row_labels: list[str],
    id_normalizer=normalize_chr_id,
) -> dict:
    """
    Precompute chromosome-row token ordering shared by all samples so tensor export
    does not repeat sorting/index construction per sample.
    """
    label_set = set(str(l) for l in row_labels)
    locus = np.asarray(locus_row, dtype=np.int64)

    idx_lists = defaultdict(list)
    for idx, chrom in enumerate(chr_id_row):
        lab = str(id_normalizer(chrom))
        if lab in label_set:
            idx_lists[lab].append(idx)

    indices_by_label: dict[str, np.ndarray] = {}
    tokens_per_chr = []
    max_tokens = 0
    for lab in row_labels:
        idxs = np.asarray(idx_lists.get(str(lab), []), dtype=np.int64)
        if idxs.size > 1:
            idxs = idxs[np.argsort(locus[idxs], kind='mergesort')]
        indices_by_label[str(lab)] = idxs
        n = int(idxs.size)
        tokens_per_chr.append(n)
        if n > max_tokens:
            max_tokens = n

    n_rows = len(row_labels)
    positions_bp_template = np.full((n_rows, max_tokens), -1, dtype=np.int64)
    mask_template = np.zeros((n_rows, max_tokens), dtype=np.float32)
    for row_idx, lab in enumerate(row_labels):
        idxs = indices_by_label[str(lab)]
        n = int(idxs.size)
        if n == 0:
            continue
        positions_bp_template[row_idx, :n] = locus[idxs]
        mask_template[row_idx, :n] = 1.0

    sg_row_flags = np.array(
        [1.0 if str(lab).startswith('A') else 0.0 for lab in row_labels],
        dtype=np.float32,
    )

    return {
        "row_labels": [str(l) for l in row_labels],
        "indices_by_label": indices_by_label,
        "tokens_per_chr": tokens_per_chr,
        "max_tokens": max_tokens,
        "positions_bp_template": positions_bp_template,
        "mask_template": mask_template,
        "sg_row_flags": sg_row_flags,
    }


def _lookup_homology_vectorized(
    lookup: dict,
    chrom: str,
    positions_bp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized homology lookup for one chromosome; unmatched positions are -1.
    """
    out_gid = np.full(positions_bp.shape, -1.0, dtype=np.float32)
    out_pid = np.full(positions_bp.shape, -1.0, dtype=np.float32)
    if positions_bp.size == 0:
        return out_gid, out_pid

    data = lookup.get(str(chrom))
    if data is None:
        return out_gid, out_pid

    starts, ends, groups, pairs = data
    pos = np.asarray(positions_bp, dtype=np.int64)
    idx = np.searchsorted(ends, pos, side="right")
    valid = idx < len(ends)
    if not np.any(valid):
        return out_gid, out_pid

    valid_rows = np.flatnonzero(valid)
    cand = idx[valid]
    inside = pos[valid] >= starts[cand]
    if np.any(inside):
        target = valid_rows[inside]
        src = cand[inside]
        out_gid[target] = groups[src].astype(np.float32, copy=False)
        out_pid[target] = pairs[src].astype(np.float32, copy=False)

    return out_gid, out_pid


def save_hierarchical_tensor(
    sample_codes: np.ndarray,
    sample_name: str,
    out_dir: str,
    chr_id_row: np.ndarray,
    locus_row: np.ndarray,
    maf_row: np.ndarray | None = None,
    callrate_row: np.ndarray | None = None,
    haplo_row: np.ndarray | None = None,
    blocklen_row_kb: np.ndarray | None = None,
    ld_row: np.ndarray | None = None,
    max_block_id: int | None = None,
    chr_info_override: dict | None = None,
    id_normalizer=normalize_chr_id,
    quality_channel: str = SNP_QUALITY_CHANNEL,
    include_sg_channel: bool = INCLUDE_SG_CHANNEL,
    homology_df: pd.DataFrame | None = None,
    hom_has: np.ndarray | None = None,
    hom_size_norm: np.ndarray | None = None,
    hom_gid: np.ndarray | None = None,
    hom_gid_bits: np.ndarray | None = None,
    hom_anchor_density: np.ndarray | None = None,
    hom_hash_k: int = HOM_HASH_K,
    te_is_te: np.ndarray | None = None,
    te_dist_bp: np.ndarray | None = None,
    gene_is_genic: np.ndarray | None = None,
    gene_is_promoter: np.ndarray | None = None,
    gene_dist_bp: np.ndarray | None = None,
    block_gene_count_norm: np.ndarray | None = None,
    block_mean_maf_norm: np.ndarray | None = None,
    block_snp_density_norm: np.ndarray | None = None,
    layout_cache: dict | None = None,
    pos_enc_cache: np.ndarray | None = None,
    quality_vector_override: np.ndarray | None = None,
    attn_mask_cache: dict | None = None,
    block_attn_mask_cache: dict | None = None,
    boundary_flag_cache: dict | None = None
):
    """
    Write a collision-free hierarchical tensor (Y=chromosome strip, X=SNP token order,
    Z=feature channels) to NPZ. Channels: dosage, quality, optional homoeolog
    presence/group-size/hash tracks, TE/gene distances, TE flag, gene context,
    block summaries, haplotype context, and subgenome flag.
    A mask array preserves where tokens exist.
    """
    global chr_info
    local_chr_info = chr_info_override if chr_info_override is not None else chr_info
    row_labels = sorted(local_chr_info.keys(), key=_chr_sort_key)

    if layout_cache is None or layout_cache.get("row_labels") != [str(l) for l in row_labels]:
        layout_cache = build_tensor_layout_cache(
            chr_id_row=chr_id_row,
            locus_row=locus_row,
            row_labels=[str(l) for l in row_labels],
            id_normalizer=id_normalizer,
        )
    indices_by_label = layout_cache["indices_by_label"]
    max_tokens = int(layout_cache["max_tokens"])
    if max_tokens == 0:
        logging.warning(f"[{sample_name}] No SNP tokens found; skipping tensor export.")
        return None

    dosage_norm = series_codes_to_dosage_norm(sample_codes).astype(np.float32, copy=False)
    is_called = np.isfinite(dosage_norm).astype(np.float32)
    dosage_norm = np.nan_to_num(dosage_norm, nan=0.0, posinf=0.0, neginf=0.0)
    if quality_vector_override is not None and len(quality_vector_override) == len(sample_codes):
        qc = np.asarray(quality_vector_override, dtype=np.float32)
    else:
        qc = _quality_vector_from_codes(sample_codes, maf_row, quality_channel, callrate_row)
    qc = np.nan_to_num(np.asarray(qc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if (
        pos_enc_cache is not None
        and pos_enc_cache.ndim == 2
        and pos_enc_cache.shape[0] == len(locus_row)
        and pos_enc_cache.shape[1] == POSITION_ENCODING_DIM
    ):
        pos_enc = np.asarray(pos_enc_cache, dtype=np.float32)
    else:
        pos_bp = np.asarray(locus_row, dtype=np.float32)
        pos_enc = sinusoidal_position_encoding(pos_bp, d_model=POSITION_ENCODING_DIM)
        pos_enc[np.isnan(pos_bp)] = 0.0

    hom_has_arr = hom_size_arr = hom_bits_arr = hom_anchor_density_arr = hom_gid_arr = None
    include_homology_channels = False
    requested_hom_hash_k = int(max(0, hom_hash_k))
    effective_hom_hash_k = 0
    if any(arr is not None for arr in (hom_has, hom_size_norm, hom_gid_bits, hom_anchor_density)):
        if hom_has is None or hom_size_norm is None or hom_gid_bits is None:
            raise ValueError("Homology channels require hom_has, hom_size_norm, and hom_gid_bits together.")
        hom_has_arr = np.asarray(hom_has, dtype=np.float32)
        hom_size_arr = np.asarray(hom_size_norm, dtype=np.float32)
        if hom_gid is not None:
            hom_gid_arr = np.asarray(hom_gid, dtype=np.float32)
            if hom_gid_arr.shape[0] != len(locus_row):
                raise ValueError("hom_gid must align to locus_row length.")
        hom_bits_arr = np.asarray(hom_gid_bits, dtype=np.float32)
        if hom_has_arr.shape[0] != len(locus_row) or hom_size_arr.shape[0] != len(locus_row):
            raise ValueError("Homology arrays must align to locus_row length.")
        if hom_bits_arr.ndim != 2 or hom_bits_arr.shape[0] != len(locus_row):
            raise ValueError("hom_gid_bits must have shape [n_snps, K].")
        effective_hom_hash_k = requested_hom_hash_k
        if effective_hom_hash_k <= 0:
            effective_hom_hash_k = int(hom_bits_arr.shape[1])
        if hom_bits_arr.shape[1] < effective_hom_hash_k:
            raise ValueError("hom_gid_bits does not contain enough hash columns for hom_hash_k.")
        include_homology_channels = True
        if hom_anchor_density is not None:
            hom_anchor_density_arr = np.asarray(hom_anchor_density, dtype=np.float32)
            if hom_anchor_density_arr.shape[0] != len(locus_row):
                raise ValueError("hom_anchor_density must align to locus_row length.")

    # Optional haplotype block context
    has_hap_channels = (
        haplo_row is not None and blocklen_row_kb is not None and ld_row is not None and max_block_id is not None
    )
    hap_block_norm = hap_block_len_norm = hap_ld = hap_block_raw = None
    if has_hap_channels:
        haplo_arr = np.asarray(haplo_row, dtype=np.int64)
        blocklen_arr = np.asarray(blocklen_row_kb, dtype=np.float32)
        ld_arr = np.nan_to_num(np.asarray(ld_row, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        hap_block_norm = np.zeros(len(locus_row), dtype=np.float32)
        hap_block_len_norm = np.zeros(len(locus_row), dtype=np.float32)
        hap_ld = np.zeros(len(locus_row), dtype=np.float32)
        # Persist raw block id per SNP; 0 marks "no block".
        hap_block_raw = np.where(haplo_arr > 0, haplo_arr, 0).astype(np.float32)
        max_bid = float(max(1, int(max_block_id)))
        max_kb = float(max(1.0, float(np.nanmax(blocklen_arr)) if len(blocklen_arr) else 1.0))
        hap_block_norm[:] = np.where(haplo_arr > 0, haplo_arr / max_bid, 0.0)
        hap_block_len_norm[:] = np.where(blocklen_arr > 0, blocklen_arr / max_kb, 0.0)
        hap_ld[:] = np.clip(ld_arr, 0.0, 1.0)
    else:
        haplo_arr = None

    feat_names = ['dosage_norm', 'is_called']
    # stable within-chromosome identity
    if INCLUDE_TOKEN_RANK_NORM:
        feat_names.append('token_rank_norm')
    # local genotype context
    if INCLUDE_LOCAL_DOSAGE_CONTEXT:
        feat_names.append('dosage_local_mean')
        feat_names.append('dosage_local_std')
    # positional encoding + quality
    feat_names.extend([f'pos_enc_{i}' for i in range(POSITION_ENCODING_DIM)])
    feat_names.append(f'quality_{quality_channel}')
    include_te_channel = te_is_te is not None and len(te_is_te) == len(locus_row)
    include_te_dist = te_dist_bp is not None and len(te_dist_bp) == len(locus_row)
    include_genic = gene_is_genic is not None and len(gene_is_genic) == len(locus_row)
    include_promoter = gene_is_promoter is not None and len(gene_is_promoter) == len(locus_row)
    include_gene_dist = gene_dist_bp is not None and len(gene_dist_bp) == len(locus_row)
    include_block_gene = block_gene_count_norm is not None and len(block_gene_count_norm) == len(locus_row)
    include_block_maf = block_mean_maf_norm is not None and len(block_mean_maf_norm) == len(locus_row)
    include_block_density = block_snp_density_norm is not None and len(block_snp_density_norm) == len(locus_row)
    te_hotspot_flag = None
    if include_genic and include_promoter:
        genic_flag = (np.asarray(gene_is_genic) > 0).astype(np.float32)
        prom_flag = (np.asarray(gene_is_promoter) > 0).astype(np.float32)
        # Strict hotspot flag: only genic or promoter loci count as hotspots to avoid over-selection.
        te_hotspot_flag = np.clip(genic_flag + prom_flag, 0, 1).astype(np.float32)
    include_te_hotspot = te_hotspot_flag is not None
    if include_homology_channels:
        feat_names.append('hom_has')
        feat_names.append('hom_group_size_norm')
        if hom_gid is not None:
            feat_names.append('hom_gid_raw')
        if hom_anchor_density_arr is not None:
            feat_names.append('homeolog_anchor_density')
        feat_names.extend([f'hom_gid_hash_{i}' for i in range(effective_hom_hash_k)])
    if include_te_dist:
        feat_names.append('te_dist_bp')
    if include_gene_dist:
        feat_names.append('gene_dist_bp')
    if include_te_channel:
        feat_names.append('is_te')
    if include_te_hotspot:
        feat_names.append('te_hotspot_flag')
    if include_genic:
        feat_names.append('is_genic')
    if include_promoter:
        feat_names.append('is_promoter')
    if include_block_gene:
        feat_names.append('block_gene_count_norm')
    if include_block_maf:
        feat_names.append('block_mean_maf_norm')
    if include_block_density:
        feat_names.append('block_snp_density_norm')
    if has_hap_channels:
        feat_names.extend(['block_id_norm', 'block_id_raw', 'block_len_norm', 'inblock_ld', 'is_block_boundary'])
    if include_sg_channel:
        feat_names.append('sg_flag')
    F = len(feat_names)

    te_channel_names = []
    if include_te_dist:
        te_channel_names.append('te_dist_bp')
    if include_te_channel:
        te_channel_names.append('is_te')
    if include_te_hotspot:
        te_channel_names.append('te_hotspot_flag')
    _log_optional_channel_group_once(
        "hierarchical_snp_tensor",
        "TE-derived",
        te_channel_names,
        "TE annotation, TE distances, or TE hotspot inputs were unavailable or misaligned to the SNP table.",
    )

    homology_channel_names = []
    if include_homology_channels:
        homology_channel_names.extend(['hom_has', 'hom_group_size_norm'])
        if hom_gid_arr is not None:
            homology_channel_names.append('hom_gid_raw')
        if hom_anchor_density_arr is not None:
            homology_channel_names.append('homeolog_anchor_density')
        homology_channel_names.extend([f'hom_gid_hash_{i}' for i in range(effective_hom_hash_k)])
    _log_optional_channel_group_once(
        "hierarchical_snp_tensor",
        "homology-derived",
        homology_channel_names,
        "Homoeolog pair, gene-annotation, or SNP-to-gene inputs did not produce aligned homology features for this tensor export.",
    )

    hap_channel_names = (
        ['block_id_norm', 'block_id_raw', 'block_len_norm', 'inblock_ld', 'is_block_boundary']
        if has_hap_channels else []
    )
    _log_optional_channel_group_once(
        "hierarchical_snp_tensor",
        "haplotype-block",
        hap_channel_names,
        "Haplotype block arrays were unavailable, so block-context channels were omitted from the SNP tensor.",
    )

    te_is_te_arr = np.asarray(te_is_te, dtype=np.float32) if include_te_channel else None
    te_dist_arr = np.asarray(te_dist_bp, dtype=np.float32) if include_te_dist else None
    gene_is_genic_arr = np.asarray(gene_is_genic, dtype=np.float32) if include_genic else None
    gene_is_promoter_arr = np.asarray(gene_is_promoter, dtype=np.float32) if include_promoter else None
    gene_dist_arr = np.asarray(gene_dist_bp, dtype=np.float32) if include_gene_dist else None
    block_gene_arr = np.asarray(block_gene_count_norm, dtype=np.float32) if include_block_gene else None
    block_maf_arr = np.asarray(block_mean_maf_norm, dtype=np.float32) if include_block_maf else None
    block_density_arr = np.asarray(block_snp_density_norm, dtype=np.float32) if include_block_density else None

    # --- NEW: local dosage context per chromosome (no cross-chrom bleed) ---
    dos_local_mean = np.zeros_like(dosage_norm, dtype=np.float32)
    dos_local_std = np.zeros_like(dosage_norm, dtype=np.float32)
    if INCLUDE_LOCAL_DOSAGE_CONTEXT:
        for lab in row_labels:
            idxs = indices_by_label[str(lab)]
            if idxs.size == 0:
                continue
            dos_chr = np.nan_to_num(dosage_norm[idxs], nan=0.0).astype(np.float32, copy=False)
            m_chr, s_chr = rolling_mean_std_1d(dos_chr, LOCAL_DOSAGE_WINDOW)
            dos_local_mean[idxs] = m_chr
            dos_local_std[idxs] = s_chr

    tensor = np.zeros((len(row_labels), max_tokens, F), dtype=np.float32)
    mask = layout_cache["mask_template"].copy()
    positions_bp = layout_cache["positions_bp_template"].copy()
    tokens_per_chr = np.asarray(layout_cache["tokens_per_chr"], dtype=np.int64)
    sg_row_flags = layout_cache["sg_row_flags"]
    attn_masks = {}
    block_attn_masks = {}

    for row_idx, lab in enumerate(row_labels):
        idxs = indices_by_label[str(lab)]
        n = int(idxs.size)
        if n == 0:
            continue
        cached_attn = attn_mask_cache.get(str(lab)) if attn_mask_cache is not None else None
        if cached_attn is not None and cached_attn.shape == (n, n):
            attn_masks[str(lab)] = cached_attn
        row_tensor = tensor[row_idx, :n, :]
        ch = 0
        row_tensor[:, ch] = dosage_norm[idxs]; ch += 1
        row_tensor[:, ch] = is_called[idxs]; ch += 1
        if INCLUDE_TOKEN_RANK_NORM:
            denom = float(max(1, n - 1))
            row_tensor[:, ch] = np.arange(n, dtype=np.float32) / denom
            ch += 1
        if INCLUDE_LOCAL_DOSAGE_CONTEXT:
            row_tensor[:, ch] = dos_local_mean[idxs]; ch += 1
            row_tensor[:, ch] = dos_local_std[idxs]; ch += 1
        row_tensor[:, ch:ch+POSITION_ENCODING_DIM] = pos_enc[idxs]
        ch += POSITION_ENCODING_DIM
        row_tensor[:, ch] = qc[idxs]; ch += 1
        if include_homology_channels:
            row_tensor[:, ch] = hom_has_arr[idxs]; ch += 1
            row_tensor[:, ch] = hom_size_arr[idxs]; ch += 1
            if hom_gid_arr is not None:
                row_tensor[:, ch] = hom_gid_arr[idxs]; ch += 1
            if hom_anchor_density_arr is not None:
                row_tensor[:, ch] = hom_anchor_density_arr[idxs]; ch += 1
            row_tensor[:, ch:ch+effective_hom_hash_k] = hom_bits_arr[idxs, :effective_hom_hash_k]
            ch += effective_hom_hash_k
        if include_te_dist:
            row_tensor[:, ch] = te_dist_arr[idxs]; ch += 1
        if include_gene_dist:
            row_tensor[:, ch] = gene_dist_arr[idxs]; ch += 1
        if include_te_channel:
            row_tensor[:, ch] = te_is_te_arr[idxs]; ch += 1
        if include_te_hotspot:
            row_tensor[:, ch] = te_hotspot_flag[idxs]; ch += 1
        if include_genic:
            row_tensor[:, ch] = gene_is_genic_arr[idxs]; ch += 1
        if include_promoter:
            row_tensor[:, ch] = gene_is_promoter_arr[idxs]; ch += 1
        if include_block_gene:
            row_tensor[:, ch] = block_gene_arr[idxs]; ch += 1
        if include_block_maf:
            row_tensor[:, ch] = block_maf_arr[idxs]; ch += 1
        if include_block_density:
            row_tensor[:, ch] = block_density_arr[idxs]; ch += 1
        if has_hap_channels:
            bids = haplo_arr[idxs]
            cached_block_attn = block_attn_mask_cache.get(str(lab)) if block_attn_mask_cache is not None else None
            if cached_block_attn is not None and cached_block_attn.shape == (n, n):
                block_attn_masks[str(lab)] = cached_block_attn
            cached_boundary = boundary_flag_cache.get(str(lab)) if boundary_flag_cache is not None else None
            if cached_boundary is not None and len(cached_boundary) == n:
                boundary_flags = cached_boundary
            else:
                left_change = np.ones(n, dtype=bool)
                right_change = np.ones(n, dtype=bool)
                if n > 1:
                    left_change[1:] = bids[1:] != bids[:-1]
                    right_change[:-1] = bids[:-1] != bids[1:]
                boundary_flags = ((bids > 0) & (left_change | right_change)).astype(np.float32)
            row_tensor[:, ch] = hap_block_norm[idxs]; ch += 1
            row_tensor[:, ch] = hap_block_raw[idxs]; ch += 1
            row_tensor[:, ch] = hap_block_len_norm[idxs]; ch += 1
            row_tensor[:, ch] = hap_ld[idxs]; ch += 1
            row_tensor[:, ch] = boundary_flags; ch += 1
        if include_sg_channel:
            row_tensor[:, ch] = sg_row_flags[row_idx]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_name}_tensor.npz")
    feature_names_bytes = np.frombuffer(",".join(feat_names).encode('ascii'), dtype=np.uint8)
    write_and_verify_npz(
        out_path,
        tensor=tensor,
        mask=mask,
        positions_bp=positions_bp,
        row_labels=np.array(row_labels, dtype='<U12'),
        chr_lengths=np.array([int(local_chr_info[lab]) for lab in row_labels], dtype=np.int64),
        tokens_per_chr=tokens_per_chr,
        feature_names_bytes=feature_names_bytes,
        quality_mode=np.array([quality_channel], dtype='<U16'),
        hom_hash_k=np.array([effective_hom_hash_k], dtype=np.int64),
        **({f"attn_mask__{lab}": attn_masks[lab] for lab in attn_masks} if attn_masks else {}),
        **({f"block_attn_mask__{lab}": block_attn_masks[lab] for lab in block_attn_masks} if block_attn_masks else {})
    )
    logging.info(f"[{sample_name}] wrote hierarchical tensor: {out_path}")
    return out_path


def save_haplotype_tensor(
    haplo_row: np.ndarray,
    blocklen_row_kb: np.ndarray,
    ld_row: np.ndarray,
    sample_name: str,
    out_dir: str,
    chr_id_row: np.ndarray,
    locus_row: np.ndarray,
    max_block_id: int,
    chr_info_override: dict | None = None,
    id_normalizer=normalize_chr_id,
    include_sg_channel: bool = INCLUDE_SG_CHANNEL,
    hash_block_id_k: int = HASH_BLOCK_ID_K,
    layout_cache: dict | None = None,
    pos_enc_cache: np.ndarray | None = None
):
    """
    Hierarchical tensor for haplotype annotations (Y=chrom strip, X=SNP token order).
    Channels: block_id_norm/raw, block_len_norm, ld_score, positional encoding, optional
    hashed block bits and subgenome flag. Mask + positions_bp preserve sparsity.
    """
    global chr_info
    local_chr_info = chr_info_override if chr_info_override is not None else chr_info
    row_labels = sorted(local_chr_info.keys(), key=_chr_sort_key)

    if layout_cache is None or layout_cache.get("row_labels") != [str(l) for l in row_labels]:
        layout_cache = build_tensor_layout_cache(
            chr_id_row=chr_id_row,
            locus_row=locus_row,
            row_labels=[str(l) for l in row_labels],
            id_normalizer=id_normalizer,
        )
    indices_by_label = layout_cache["indices_by_label"]
    max_tokens = int(layout_cache["max_tokens"])
    if max_tokens == 0:
        logging.warning(f"[{sample_name}] No haplotype tokens found; skipping tensor export.")
        return None

    max_bid = float(max(1, int(max_block_id or 1)))
    blocklen_arr = np.asarray(blocklen_row_kb, dtype=np.float32)
    max_kb = float(max(1.0, float(np.nanmax(blocklen_arr)) if blocklen_arr.size else 1.0))
    haplo_arr = np.asarray(haplo_row, dtype=np.int64)
    ld_vals = np.full(haplo_arr.shape, np.nan, dtype=np.float32)
    n_ld = min(ld_vals.size, len(ld_row))
    if n_ld > 0:
        ld_vals[:n_ld] = np.asarray(ld_row[:n_ld], dtype=np.float32)
    ld_vals = np.nan_to_num(ld_vals, nan=0.0, posinf=0.0, neginf=0.0)

    hashed_bits = _hashed_block_bits(haplo_arr, hash_block_id_k)
    if (
        pos_enc_cache is not None
        and pos_enc_cache.ndim == 2
        and pos_enc_cache.shape[0] == len(locus_row)
        and pos_enc_cache.shape[1] == POSITION_ENCODING_DIM
    ):
        pos_enc = np.asarray(pos_enc_cache, dtype=np.float32)
    else:
        pos_bp = np.asarray(locus_row, dtype=np.float32)
        pos_enc = sinusoidal_position_encoding(pos_bp, d_model=POSITION_ENCODING_DIM)
        pos_enc[np.isnan(pos_bp)] = 0.0

    feat_names = ['block_id_norm', 'block_id_raw', 'block_len_norm', 'ld_score']
    feat_names.extend([f'pos_enc_{i}' for i in range(POSITION_ENCODING_DIM)])
    if hash_block_id_k > 0:
        feat_names.extend([f'block_hash_{i}' for i in range(hash_block_id_k)])
    if include_sg_channel:
        feat_names.append('sg_flag')
    F = len(feat_names)

    _log_optional_channel_group_once(
        "haplotype_tensor",
        "haplotype-block",
        list(feat_names),
        "Haplotype tensor export was skipped before channel assembly.",
    )

    block_norm_all = np.where(haplo_arr > 0, haplo_arr / max_bid, 0.0).astype(np.float32)
    block_raw_all = haplo_arr.astype(np.float32, copy=False)
    block_len_norm_all = np.where(blocklen_arr > 0, blocklen_arr / max_kb, 0.0).astype(np.float32)

    tensor = np.zeros((len(row_labels), max_tokens, F), dtype=np.float32)
    mask = layout_cache["mask_template"].copy()
    positions_bp = layout_cache["positions_bp_template"].copy()
    tokens_per_chr = np.asarray(layout_cache["tokens_per_chr"], dtype=np.int64)
    sg_row_flags = layout_cache["sg_row_flags"]

    for row_idx, lab in enumerate(row_labels):
        idxs = indices_by_label[str(lab)]
        n = int(idxs.size)
        if n == 0:
            continue
        row_tensor = tensor[row_idx, :n, :]
        ch = 0
        row_tensor[:, ch] = block_norm_all[idxs]; ch += 1
        row_tensor[:, ch] = block_raw_all[idxs]; ch += 1
        row_tensor[:, ch] = block_len_norm_all[idxs]; ch += 1
        row_tensor[:, ch] = ld_vals[idxs]; ch += 1
        row_tensor[:, ch:ch+POSITION_ENCODING_DIM] = pos_enc[idxs]
        ch += POSITION_ENCODING_DIM
        if hash_block_id_k > 0:
            row_tensor[:, ch:ch+hash_block_id_k] = hashed_bits[idxs]
            ch += hash_block_id_k
        if include_sg_channel:
            row_tensor[:, ch] = sg_row_flags[row_idx]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_name}_haplo_tensor.npz")
    feature_names_bytes = np.frombuffer(",".join(feat_names).encode('ascii'), dtype=np.uint8)
    write_and_verify_npz(
        out_path,
        tensor=tensor,
        mask=mask,
        positions_bp=positions_bp,
        row_labels=np.array(row_labels, dtype='<U12'),
        chr_lengths=np.array([int(local_chr_info[lab]) for lab in row_labels], dtype=np.int64),
        tokens_per_chr=tokens_per_chr,
        feature_names_bytes=feature_names_bytes,
        hash_block_id_k=np.array([hash_block_id_k], dtype=np.int64)
    )
    logging.info(f"[{sample_name}] wrote haplotype tensor: {out_path}")
    return out_path


#   RGB-only colored by block ID, with collisions and event sidecars.  
def save_all_chromosomes_as_tiled_images_haplo(
    haplo_row, sample_name, folder, chr_id_row, locus_row,
    dpi=100, tile_width=16384, desired_max_width_px=DESIRED_MAX_WIDTH_PX,
    strip_height=10, chr_info_override=None, id_normalizer=normalize_chr_id,
    collision_depth=COLLISION_DEPTH, skip_empty_tiles=False, write_index_csv=True, save_npz=True,
    adapt_maps: dict | None = None, color_lookup_override: dict | None = None
):
    global chr_info
    desired_max_width_px = normalize_desired_width(desired_max_width_px, tile_width)

    local_chr_info = chr_info_override if chr_info_override is not None else chr_info
    sorted_chr = sorted(local_chr_info.keys(), key=_chr_sort_key)
    chr_to_row = {chrom: idx for idx, chrom in enumerate(sorted_chr)}

    total_chrs = len(local_chr_info)
    full_h = total_chrs * strip_height
    num_tiles = math.ceil(desired_max_width_px / tile_width)

    chr_labels = np.array([id_normalizer(c) for c in chr_id_row], dtype=object)

    # mapping
    if adapt_maps is None:
        base_scale = desired_max_width_px / float(max(local_chr_info.values()))
        def _map_x(label, bp):
            return min(max(int(int(bp) * base_scale), 0), desired_max_width_px - 1)
        scale_payload = {}
    else:
        def _map_x(label, bp):
            be, pe = adapt_maps[str(label)]
            return min(max(int(interp_bp_to_px(be, pe, int(bp))), 0), desired_max_width_px - 1)
        scale_payload = {}
        for chrom in sorted_chr:
            be, pe = adapt_maps[chrom]
            scale_payload[f"scale_bp_edges__{chrom}"] = be.astype(np.int64)
            scale_payload[f"scale_px_edges__{chrom}"] = pe.astype(np.int64)

    # colors
    max_block_id = int(np.nanmax(haplo_row)) if haplo_row.size else 0
    palette = color_lookup_override if color_lookup_override is not None \
              else get_or_build_color_lookup(max_block_id, folder)

    depths = [1 for _ in range(num_tiles)]
    depth_counter = defaultdict(int)
    for lab, locus in zip(chr_labels, locus_row):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None:
            continue
        x_pixel = _map_x(lab, locus)
        t_idx = x_pixel // tile_width
        x_in  = x_pixel % tile_width
        depth_counter[(t_idx, row_idx, x_in)] += 1
    for t in range(num_tiles):
        max_hit = max((cnt for (tt, _, _), cnt in depth_counter.items() if tt == t), default=1)
        depths[t] = max(1, min(int(max_hit), collision_depth))

    tile_layers = [
        [np.full((full_h, tile_width, 3), 255, dtype=np.uint8) for _ in range(depths[i])]
        for i in range(num_tiles)
    ]
    pixel_depths = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    overflow     = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    events       = [[] for _ in depths]

    for lab, locus, block_id in zip(chr_labels, locus_row, haplo_row):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None:
            continue
        x_pixel = _map_x(lab, locus)
        t_idx = x_pixel // tile_width
        x_in  = x_pixel % tile_width
        y0, y1 = row_idx * strip_height, (row_idx + 1) * strip_height
        events[t_idx].append((row_idx, x_in, int(locus), int(block_id)))
        color = palette.get(int(block_id), np.array([200,200,200], dtype=np.uint8))
        d = pixel_depths[t_idx][row_idx, x_in]
        if d < depths[t_idx]:
            tile_layers[t_idx][d][y0:y1, x_in] = color
            pixel_depths[t_idx][row_idx, x_in] = d + 1
        else:
            overflow[t_idx][row_idx, x_in] += 1

    os.makedirs(folder, exist_ok=True)
    index_rows = []
    for i in range(num_tiles):
        comp = np.full((full_h, tile_width, 3), 255, dtype=np.uint8)
        has_content = False
        for d in range(depths[i]):
            L = tile_layers[i][d]
            mask = (L != 255).any(axis=2)
            if mask.any():
                comp[mask] = L[mask]
                has_content = True
        png_path = os.path.join(folder, f"{sample_name}_tile_{i}.png")
        plt.imsave(png_path, comp.astype(np.uint8), dpi=dpi)

        if save_npz:
            ev = np.array(events[i],
                          dtype=[('row', np.int32), ('x', np.int32),
                                 ('locus_bp', np.int64), ('block_id', np.int32)])
            meta = np.array([full_h, tile_width, strip_height, depths[i],
                             desired_max_width_px], dtype=np.int64)
            npz_path = os.path.join(folder, f"{sample_name}_tile_{i}.npz")
            write_and_verify_npz(
                npz_path,
                layers=np.concatenate(tile_layers[i], axis=2),
                overflow=overflow[i], events=ev, meta=meta, **scale_payload
            )

        bp_start = i * tile_width  # pixel space, kept for index parity
        bp_end   = (i + 1) * tile_width - 1
        index_rows.append({"sample": sample_name, "tile_idx": i,
                           "x_start_px": i*tile_width, "x_end_px": (i+1)*tile_width - 1,
                           "bp_start": bp_start, "bp_end": bp_end,
                           "has_content": int(has_content or (len(events[i]) > 0))})

    if write_index_csv:
        idx_path = os.path.join(folder, f"{sample_name}_haplo_tile_index.csv")
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)


def save_tiled_features_haplo(
    haplo_row: np.ndarray,            # per-SNP block_id
    sample_name: str,
    folder: str,
    chr_id_row: np.ndarray,
    locus_row: np.ndarray,
    blocklen_row_kb: np.ndarray,      # per-SNP block length (kb)
    ld_row: np.ndarray,               # per-SNP in-block LD score [0..1]
    max_block_id: int,
    subgenome_labels: dict,           # {'A':1.0, 'C':0.0}
    dpi=100, tile_width=16384, desired_max_width_px=DESIRED_MAX_WIDTH_PX,
    strip_height=10, chr_info_override=None, id_normalizer=normalize_chr_id,
    collision_depth=COLLISION_DEPTH, skip_empty_tiles=False, write_index_csv=True, save_npz=True,
    include_sg_channel=True,
    adaptive_bins=200, adaptive_alpha=0.8, min_fold=0.5, max_fold=2.0,
    haplotype_blocks_df: pd.DataFrame = None,   # <-- NEW,
    homology_df: pd.DataFrame = None,
    include_homology: bool = False,
    hash_block_id_k: int = HASH_BLOCK_ID_K
):
    """
    Create NPZ tiles with haplotype features per pixel (float32 in [0,1]):
      Ch1: block_id_norm (block_id/max_block_id; 0 if no block)
      Ch2: block_len_norm (kb / max_kb)
      Ch3: in-block LD score (0..1)
      (+ optional Ch4: sg_flag)
    Deterministic layering by (position asc, |LD-0.5| desc).
    """
    global chr_info
    desired_max_width_px = normalize_desired_width(desired_max_width_px, tile_width)

    local_chr_info = chr_info_override if chr_info_override is not None else chr_info
    sorted_chr = sorted(local_chr_info.keys(), key=_chr_sort_key)
    chr_to_row = {chrom: idx for idx, chrom in enumerate(sorted_chr)}
    total_chrs = len(local_chr_info)

    # normalize features
    mask_b = (haplo_row > 0).astype(np.float32)  # 1 if in a block, else 0
    feat_names = ['in_block_mask', 'block_len_norm', 'inblock_ld']  # Ch1 now the mask
    if hash_block_id_k and hash_block_id_k > 0:
        feat_names += [f"block_hash_{i}" for i in range(hash_block_id_k)]

    # === NEW: has_block mask + optional hashed block-ID bits ===
    has_block = (haplo_row > 0).astype(np.float32)

    # Optional K-bit hash of block_id (stable, zero for "no block")
    K = int(max(0, hash_block_id_k))
    hid = np.zeros((haplo_row.size, K), dtype=np.float32)
    if K > 0:
        bid = haplo_row.astype(np.int64)
        # SplitMix64-style avalanching; deterministic and vectorized
        x = (np.uint64(bid) ^ np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)
        x ^= (x >> np.uint64(30))
        x *= np.uint64(0x94D049BB133111EB)
        for i in range(K):
            hid[:, i] = ((x >> np.uint64(i)) & np.uint64(1)).astype(np.float32)
        # ensure "no block" stays all-zeros
        hid[bid <= 0, :] = 0.0

    # Keep the other two channels the same
    max_kb = float(max(1.0, blocklen_row_kb.max()))
    block_len_norm = (blocklen_row_kb / max_kb).astype(np.float32)
    ld_norm = np.clip(ld_row.astype(np.float32), 0.0, 1.0)


    # width & scales
    desired = int(desired_max_width_px)
    base_scale = desired / float(max(local_chr_info.values()))
    labels = np.array([id_normalizer(c) for c in chr_id_row], dtype=object)

    use_homology = bool(include_homology and (homology_df is not None))

    adapt_maps = {}
    for chrom in sorted_chr:
        chr_len = int(local_chr_info[chrom])
        target_w = int(chr_len * base_scale)
        loci_this = locus_row[labels == chrom]
        adapt_maps[chrom] = build_adaptive_scale_for_chr(chr_len, loci_this, target_w,
                                                         bins=adaptive_bins, alpha=adaptive_alpha,
                                                         min_fold=min_fold, max_fold=max_fold)

    # --- NEW: per-chromosome block lists (bp1,bp2,block_id,kb) ---
    blocks_by_label = group_blocks_by_label(haplotype_blocks_df, id_normalizer) if haplotype_blocks_df is not None else {}

    num_tiles = math.ceil(desired / tile_width)
    feat_names = ['has_block', 'block_len_norm', 'inblock_ld']
    if K > 0:
        feat_names.extend([f'hid_{i}' for i in range(K)])
    if include_sg_channel:
        feat_names.append('sg_flag')
    F = len(feat_names)

    # Adaptive depth per tile: count max hits per pixel
    depths = [1 for _ in range(num_tiles)]
    depth_counter = defaultdict(int)
    for lab, bp in zip(labels, locus_row):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None:
            continue
        be, pe = adapt_maps[str(lab)]
        x_in_row = interp_bp_to_px(be, pe, int(bp))
        x_pixel  = min(max(int(x_in_row), 0), desired - 1)
        t_idx    = x_pixel // tile_width
        x_in_tile= x_pixel % tile_width
        depth_counter[(t_idx, row_idx, x_in_tile)] += 1
    for t in range(num_tiles):
        max_hit = max((cnt for (tt, _, _), cnt in depth_counter.items() if tt == t), default=1)
        depths[t] = max(1, min(int(max_hit), collision_depth))

    tile_layers = [
        [np.zeros((total_chrs * strip_height, tile_width, F), dtype=np.float32) for _ in range(depths[i])]
        for i in range(num_tiles)
    ]

    full_h = total_chrs * strip_height
    num_tiles = math.ceil(desired / tile_width)

    # NEW rasters per tile (single "composite" each, not per-depth)
    border_rasters      = [np.zeros((full_h, tile_width), dtype=np.uint8)   for _ in range(num_tiles)]
    block_id_rasters    = [np.zeros((full_h, tile_width), dtype=np.uint32)  for _ in range(num_tiles)]
    block_len_kb_rasters= [np.zeros((full_h, tile_width), dtype=np.float32) for _ in range(num_tiles)]


    # Build homoeolog map
    hom_by_label = group_homology_by_label(homology_df, id_normalizer) if use_homology else {}
    hom_group_rasters = hom_pair_rasters = None
    if use_homology:
        hom_group_rasters = [np.zeros((full_h, tile_width), dtype=np.uint32) for _ in range(num_tiles)]
        hom_pair_rasters  = [np.zeros((full_h, tile_width), dtype=np.uint32) for _ in range(num_tiles)]

    sg_row_mask_rasters = None
    if include_sg_channel:
        sg_row_mask_rasters = [np.zeros((full_h, tile_width), dtype=np.float32) for _ in range(num_tiles)]

    pixel_depths = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    overflow     = [np.zeros((total_chrs, tile_width), dtype=np.uint16) for _ in depths]
    grouped = [defaultdict(list) for _ in depths]
    events  = [[] for _ in depths]
    
    for idx, (lab, bp, bid, msk, bln, ldn) in enumerate(
            zip(labels, locus_row, haplo_row, has_block, block_len_norm, ld_norm)):
        row_idx = chr_to_row.get(str(lab))
        if row_idx is None:
            continue
        be, pe = adapt_maps[str(lab)]
        x_in_row = interp_bp_to_px(be, pe, int(bp))
        x_pixel  = min(max(int(x_in_row), 0), desired - 1)
        t_idx    = x_pixel // tile_width
        x_in_tile= x_pixel % tile_width

        feat_list = [float(msk), float(bln), float(ldn)]
        if K > 0:
            # append hashed ID bits for this SNP
            feat_list.extend(hid[idx].tolist())
        if include_sg_channel:
            sg_flag = 1.0 if str(lab).startswith('A') else 0.0
            feat_list.append(sg_flag)

        grouped[t_idx][(row_idx, x_in_tile)].append((bp, abs(ldn - 0.5), feat_list))
        events[t_idx].append((row_idx, x_in_tile, int(bp), int(bid)))


    # --- NEW: rasterize block spans per chromosome into per-tile arrays ---
    for chrom in sorted_chr:
        row_idx = chr_to_row[chrom]
        y0, y1 = row_idx * strip_height, (row_idx + 1) * strip_height
        be, pe = adapt_maps[chrom]                 # bp->px map for this chromosome
        tile_x0s = [i * tile_width for i in range(num_tiles)]
        tile_x1s = [(i+1) * tile_width - 1 for i in range(num_tiles)]

        for bp1, bp2, bid, kb in blocks_by_label.get(chrom, []):
            # Map inclusive genomic coordinates into a half-open pixel interval [x1, x2).
            bp1 = int(bp1)
            bp2 = int(bp2)
            x1 = interp_bp_to_px(be, pe, bp1)
            x2 = interp_bp_to_px(be, pe, max(bp1, bp2 - 1)) + 1
            x2 = min(x2, int(pe[-1]))
            x2 = max(x2, x1 + 1)

            # draw into each tile that intersects the block
            for t_idx, (tx0, tx1) in enumerate(zip(tile_x0s, tile_x1s)):
                start = max(x1, tx0)
                end   = min(x2 - 1, tx1)
                if start > end:
                    continue
                xs = start - tx0
                xe = end   - tx0

                # fill block ID and block length across the whole strip height
                block_id_rasters[t_idx][y0:y1, xs:xe+1]     = bid
                block_len_kb_rasters[t_idx][y0:y1, xs:xe+1] = kb

                # borders (1px vertical lines) if they land inside this tile
                if tx0 <= x1 <= tx1:
                    border_rasters[t_idx][y0:y1, x1 - tx0] = 1
                if tx0 <= (x2 - 1) <= tx1:
                    border_rasters[t_idx][y0:y1, (x2 - 1) - tx0] = 1

    # after filling block_len_kb_rasters[...] inside save_tiled_features_haplo(...)
    max_kb_glob = float(max(1.0, max(arr.max() for arr in block_len_kb_rasters)))
    block_len_norm_rasters = [arr / max_kb_glob for arr in block_len_kb_rasters]

    # Rasterize homoeolog spans
    if use_homology:
        for chrom in sorted_chr:
            row_idx = chr_to_row[chrom]
            y0, y1 = row_idx*strip_height, (row_idx+1)*strip_height
            be, pe = adapt_maps[chrom]
            for bp1, bp2, gid, pid in hom_by_label.get(chrom, []):
                bp1 = int(bp1)
                bp2 = int(bp2)
                x1 = interp_bp_to_px(be, pe, bp1)
                x2 = interp_bp_to_px(be, pe, max(bp1, bp2 - 1)) + 1
                x2 = min(x2, int(pe[-1]))
                x2 = max(x2, x1 + 1)
                for t_idx in range(num_tiles):
                    tx0, tx1 = t_idx*tile_width, (t_idx+1)*tile_width - 1
                    start = max(x1, tx0); end = min(x2-1, tx1)
                    if start > end: continue
                    xs, xe = start - tx0, end - tx0
                    hom_group_rasters[t_idx][y0:y1, xs:xe+1] = gid
                    hom_pair_rasters[t_idx][y0:y1,  xs:xe+1] = pid

    # draw deterministically
    full_h = total_chrs * strip_height
    for t in range(num_tiles):
        for (row_idx, x), lst in grouped[t].items():
            lst.sort(key=lambda z: (z[0], -z[1]))
            y0, y1 = row_idx * strip_height, (row_idx + 1) * strip_height
            depth_used = 0
            for _, __, feat in lst:
                if depth_used >= depths[t]:
                    overflow[t][row_idx, x] += 1
                else:
                    tile_layers[t][depth_used][y0:y1, x, :] = np.asarray(feat, dtype=np.float32)
                    pixel_depths[t][row_idx, x] = depth_used + 1
                    depth_used += 1

    # right after you know num_tiles/adapt_maps in save_tiled_features_haplo(...)
    if include_sg_channel:
        for chrom in sorted_chr:
            row_idx = chr_to_row[chrom]
            y0, y1  = row_idx*strip_height, (row_idx+1)*strip_height
            val     = 1.0 if str(chrom).startswith('A') else 0.0
            for t_idx in range(num_tiles):
                sg_row_mask_rasters[t_idx][y0:y1, :] = val

    # save
    os.makedirs(folder, exist_ok=True)
    index_rows = []
    for i in range(num_tiles):
        layers_stacked = np.concatenate(tile_layers[i], axis=2)
        scale_payload = {}
        for chrom in sorted_chr:
            be, pe = adapt_maps[chrom]
            scale_payload[f"scale_bp_edges__{chrom}"] = be.astype(np.int64)
            scale_payload[f"scale_px_edges__{chrom}"] = pe.astype(np.int64)

        ev = np.array(events[i],
                      dtype=[('row', np.int32), ('x', np.int32),
                             ('locus_bp', np.int64), ('block_id', np.int32)])

        if save_npz:
            npz_path = os.path.join(folder, f"{sample_name}_feat_tile_{i}.npz")
            meta = np.array([full_h, tile_width, strip_height, depths[i],
                             desired, F], dtype=np.int64)
            ch_names = np.frombuffer(",".join(feat_names).encode('ascii'), dtype=np.uint8)
            extras = dict(
                block_id_uint32=block_id_rasters[i],
                block_border=border_rasters[i],
                block_len_kb_raster=block_len_kb_rasters[i],
                block_len_norm_raster=block_len_norm_rasters[i],
                **scale_payload
            )
            if use_homology:
                extras["hom_group_uint32"] = hom_group_rasters[i]
                extras["hom_pair_uint32"]  = hom_pair_rasters[i]
            if include_sg_channel and sg_row_mask_rasters is not None:
                extras["sg_row_mask"] = sg_row_mask_rasters[i]
            
            write_and_verify_npz(
                npz_path,
                layers=layers_stacked,
                overflow=overflow[i],
                events=ev,
                meta=meta,
                channel_names_bytes=ch_names,
                **extras
            )


        x_start = i * tile_width
        x_end   = (i + 1) * tile_width - 1
        index_rows.append({
            "sample": sample_name, "tile_idx": i,
            "x_start_px": x_start, "x_end_px": x_end,
            "has_content": int(len(events[i]) > 0),
        })

    if write_index_csv:
        idx_path = os.path.join(folder, f"{sample_name}_haplo_feat_tile_index.csv")
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        logging.info(f"[{sample_name}] wrote haplotype feature tile index: {idx_path}")

def _prepare_sample_subdir(root_dir: str, sample_name: str) -> str:
    """Ensure a per-sample folder exists under the given root and return its path."""
    path = os.path.join(root_dir, sample_name)
    os.makedirs(path, exist_ok=True)
    return path


def _clear_sample_tiles(sample_dir: str, sample_name: str):
    """
    Remove stale tiles/index files from previous runs so new tiles don't get mixed with old ones.
    """
    try:
        for pattern in (
            f"{sample_name}_tile_*.npz",
            f"{sample_name}_feat_tile_*.npz",
        ):
            for fp in glob.glob(os.path.join(sample_dir, pattern)):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        for fp in glob.glob(os.path.join(sample_dir, f"{sample_name}_*tile_index.csv")):
            try:
                os.remove(fp)
            except OSError:
                pass
    except Exception:
        pass


def render_sample_lossless(
    sample: str,
    sample_series: np.ndarray,
    chr_id_row: np.ndarray,
    locus_row: np.ndarray,
    haplo_row: np.ndarray,
    out_snps_root: str,
    out_hap_root: str,
    tile_width: int = 1024,
    maf_row: np.ndarray = None,
    callrate_row: np.ndarray | None = None,
    blocklen_row_kb: np.ndarray = None,
    ld_row: np.ndarray = None,
    include_sg_channel: bool = True,
    write_feature_tiles: bool = True,
    max_block_id: int = None,
    haplotype_blocks: pd.DataFrame = None,
    homology_df: pd.DataFrame = None,
    te_is_te: np.ndarray | None = None
):
    local_chr_info, id_norm = build_chr_info_for_mode()

    # SG channels are included only if config is provided (generic A/B/C/D)
    include_sg_generic = include_sg_channel and bool(SUBGENOME_CHRS)

    if USE_SUBGENOMES:
        CHR_A = {k: v for k, v in local_chr_info.items() if k in SUBGENOME_CHRS.get('A', [])}
        CHR_C = {k: v for k, v in local_chr_info.items() if k in SUBGENOME_CHRS.get('C', [])}
        snp_dir_A = os.path.join(out_snps_root, "A")
        snp_dir_C = os.path.join(out_snps_root, "C")
        hap_dir_A = os.path.join(out_hap_root,  "A")
        hap_dir_C = os.path.join(out_hap_root,  "C")
        os.makedirs(snp_dir_A, exist_ok=True); os.makedirs(snp_dir_C, exist_ok=True)
        os.makedirs(hap_dir_A, exist_ok=True); os.makedirs(hap_dir_C, exist_ok=True)

        sample_snp_dir_A = _prepare_sample_subdir(snp_dir_A, sample)
        sample_snp_dir_C = _prepare_sample_subdir(snp_dir_C, sample)
        sample_hap_dir_A = _prepare_sample_subdir(hap_dir_A, sample)
        sample_hap_dir_C = _prepare_sample_subdir(hap_dir_C, sample)
        _clear_sample_tiles(sample_snp_dir_A, f"{sample}_A")
        _clear_sample_tiles(sample_snp_dir_C, f"{sample}_C")
        _clear_sample_tiles(sample_hap_dir_A, f"{sample}_A")
        _clear_sample_tiles(sample_hap_dir_C, f"{sample}_C")

        # Build adapt maps once per SG at the same target width (we can reuse the same desired width)
        desired_w = get_existing_W(sample, sample_snp_dir_A, sample_snp_dir_C, tile_width) or DESIRED_MAX_WIDTH_PX
        labels = np.array([id_norm(c) for c in chr_id_row], dtype=object)
        expected_A = int(np.isin(labels, list(CHR_A.keys())).sum())
        expected_C = int(np.isin(labels, list(CHR_C.keys())).sum())

        adapt_A = build_adaptive_maps_for_all(CHR_A, labels, locus_row, desired_w)
        adapt_C = build_adaptive_maps_for_all(CHR_C, labels, locus_row, desired_w)

        # SNP pass (RGB + Features)
        WA = save_all_chromosomes_as_tiled_images(
                sample_series, f"{sample}_A", sample_snp_dir_A,
                chr_id_row, locus_row,
                tile_width=tile_width, chr_info_override=CHR_A, id_normalizer=id_norm,
                adapt_maps=adapt_A
        )
        WC = save_all_chromosomes_as_tiled_images(
                sample_series, f"{sample}_C", sample_snp_dir_C,
                chr_id_row, locus_row,
                tile_width=tile_width, chr_info_override=CHR_C, id_normalizer=id_norm,
                adapt_maps=adapt_C
        )
        W = max(WA, WC)

        if write_feature_tiles:
            save_tiled_features_snp(
                sample_series, f"{sample}_A", sample_snp_dir_A,
                chr_id_row, locus_row, maf_row, SUBGENOME_LABELS, callrate_row,
                tile_width=tile_width, chr_info_override=CHR_A, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,  # N-hot rows
                quality_channel=SNP_QUALITY_CHANNEL, include_density_channel=INCLUDE_DENSITY_CHANNEL,
                te_is_te=te_is_te,
            )
            save_tiled_features_snp(
                sample_series, f"{sample}_C", sample_snp_dir_C,
                chr_id_row, locus_row, maf_row, SUBGENOME_LABELS, callrate_row,
                tile_width=tile_width, chr_info_override=CHR_C, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,
                quality_channel=SNP_QUALITY_CHANNEL, include_density_channel=INCLUDE_DENSITY_CHANNEL,
                te_is_te=te_is_te,
            )

        # Haplotype pass
        palette_A = get_or_build_color_lookup(int(haplotype_blocks.shape[0]), sample_hap_dir_A)
        palette_C = get_or_build_color_lookup(int(haplotype_blocks.shape[0]), sample_hap_dir_C)

        save_all_chromosomes_as_tiled_images_haplo(
            haplo_row, f"{sample}_A", sample_hap_dir_A,
            chr_id_row, locus_row,
            desired_max_width_px=W, tile_width=tile_width,
            chr_info_override=CHR_A, id_normalizer=id_norm,
            adapt_maps=adapt_A, color_lookup_override=palette_A
        )
        save_all_chromosomes_as_tiled_images_haplo(
            haplo_row, f"{sample}_C", sample_hap_dir_C,
            chr_id_row, locus_row,
            desired_max_width_px=W, tile_width=tile_width,
            chr_info_override=CHR_C, id_normalizer=id_norm,
            adapt_maps=adapt_C, color_lookup_override=palette_C
        )
        if write_feature_tiles:
            save_tiled_features_haplo(
                haplo_row, f"{sample}_A", sample_hap_dir_A,
                chr_id_row, locus_row, blocklen_row_kb, ld_row, max_block_id, SUBGENOME_LABELS,
                desired_max_width_px=W, tile_width=tile_width,
                chr_info_override=CHR_A, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,
                haplotype_blocks_df=haplotype_blocks, homology_df=homology_df,
                include_homology=INCLUDE_HOMOLOGY, hash_block_id_k=HASH_BLOCK_ID_K
            )
            save_tiled_features_haplo(
                haplo_row, f"{sample}_C", sample_hap_dir_C,
                chr_id_row, locus_row, blocklen_row_kb, ld_row, max_block_id, SUBGENOME_LABELS,
                desired_max_width_px=W, tile_width=tile_width,
                chr_info_override=CHR_C, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,
                haplotype_blocks_df=haplotype_blocks, homology_df=homology_df,
                include_homology=INCLUDE_HOMOLOGY, hash_block_id_k=HASH_BLOCK_ID_K
            )

        # Sanity checks
        sanity_checks_after_render(f"{sample}_A", sample_snp_dir_A, sample_hap_dir_A, expected_snps=expected_A, use_subgenomes=True)
        sanity_checks_after_render(f"{sample}_C", sample_snp_dir_C, sample_hap_dir_C, expected_snps=expected_C, use_subgenomes=True)
        return W

    else:
        os.makedirs(out_snps_root, exist_ok=True)
        os.makedirs(out_hap_root,  exist_ok=True)

        sample_snp_dir = _prepare_sample_subdir(out_snps_root, sample)
        sample_hap_dir = _prepare_sample_subdir(out_hap_root,  sample)
        _clear_sample_tiles(sample_snp_dir, sample)
        _clear_sample_tiles(sample_hap_dir, sample)

        labels = np.array([id_norm(c) for c in chr_id_row], dtype=object)
        desired_w = get_existing_W(sample, sample_snp_dir, sample_snp_dir, tile_width) or DESIRED_MAX_WIDTH_PX
        adapt_maps = build_adaptive_maps_for_all(local_chr_info, labels, locus_row, desired_w)

        W = save_all_chromosomes_as_tiled_images(
                sample_series, f"{sample}", sample_snp_dir,
                chr_id_row, locus_row,
                tile_width=tile_width, chr_info_override=local_chr_info, id_normalizer=id_norm,
                adapt_maps=adapt_maps
        )
        save_all_chromosomes_as_tiled_images_haplo(
                haplo_row, f"{sample}", sample_hap_dir,
                chr_id_row, locus_row,
                desired_max_width_px=W, tile_width=tile_width,
                chr_info_override=local_chr_info, id_normalizer=id_norm,
                adapt_maps=adapt_maps
        )
        if write_feature_tiles:
            save_tiled_features_snp(
                sample_series, f"{sample}", sample_snp_dir,
                chr_id_row, locus_row, maf_row, SUBGENOME_LABELS, callrate_row,
                tile_width=tile_width, chr_info_override=local_chr_info, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,
                quality_channel=SNP_QUALITY_CHANNEL, include_density_channel=INCLUDE_DENSITY_CHANNEL,
                te_is_te=te_is_te
            )
            save_tiled_features_haplo(
                haplo_row, f"{sample}", sample_hap_dir,
                chr_id_row, locus_row, blocklen_row_kb, ld_row, max_block_id, SUBGENOME_LABELS,
                desired_max_width_px=W, tile_width=tile_width,
                chr_info_override=local_chr_info, id_normalizer=id_norm,
                include_sg_channel=include_sg_generic,
                haplotype_blocks_df=haplotype_blocks, homology_df=homology_df if INCLUDE_HOMOLOGY else None,
                include_homology=INCLUDE_HOMOLOGY, hash_block_id_k=HASH_BLOCK_ID_K
            )
        sanity_checks_after_render(f"{sample}", sample_snp_dir, sample_hap_dir,
                                   expected_snps=len(locus_row), use_subgenomes=False)
        return W


def audit_sample_events(sample, npz_dir, expected_snps):
    """Verify every SNP is present in some tile's sidecar events."""
    paths = sorted(glob.glob(os.path.join(npz_dir, f"{sample}_tile_*.npz")))
    n_events, n_overflow = 0, 0
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            n_events += len(z["events"])
            n_overflow += int(np.asarray(z["overflow"]).sum())
    ok = (n_events == expected_snps)
    logging.info(f"[AUDIT] {sample}: events={n_events} expected={expected_snps} "
                 f"OK={ok} overflow_sum={n_overflow} tiles={len(paths)}")
    return ok


def _is_nonempty_file(path: str) -> bool:
    """Fast resume check: file exists and has non-zero size."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _require(condition: bool, message: str, exc_type=ValueError):
    """Runtime validation that remains active under optimized Python runs."""
    if not condition:
        raise exc_type(message)


def main():
        # --- set paths first ---
    vcf_path      = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean.fixed.vcf"
    ped_file      = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean.fixed.ped"
    map_file      = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean.fixed.map"
    haplotype_block_file =  "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/combined_blocks.det"
    ENCODED_WITH_BLOCKS = "encoded_genotypes_with_haplotype_blocks.csv"

    # --- optional: ignore resume until table is clean ---
    # If you really want resume later, see the "safe resume reader" below.
    # For now, always rebuild cleanly:
    # if os.path.exists(ENCODED_WITH_BLOCKS): os.remove(ENCODED_WITH_BLOCKS)

    if COLOR_MODE == 'allele_combination':
        encoded_df, _, _ = load_plink_data(ped_file, map_file)
        logging.info(f"Loaded PLINK .ped with {encoded_df.shape[0]} SNPs and {encoded_df.shape[1]-2} samples.")
        allele_rank_df = encoded_df.attrs.get('allele_rank')
        if allele_rank_df is None:
            allele_rank_df = build_allele_rank_summary(encoded_df)
        allele_rank_df.to_csv("allele_rank_summary.csv", index=False)
        logging.info("Saved allele rank summary (per-SNP frequency ordering) to allele_rank_summary.csv.")
        log_allele_combination_diversity(encoded_df, tag="PLINK allele codes")
        maf_series = compute_maf_from_allele_codes(encoded_df)
    else:
        encoded_df = load_vcf_as_dosage(vcf_path, use_DS_if_present=True)
        logging.info(f"Loaded VCF with {encoded_df.shape[0]} variants and {encoded_df.shape[1]-2} samples.")
        maf_series = compute_maf_from_dosage(encoded_df)

    # 2) Haplotype blocks
    haplotype_blocks = load_haplotype_blocks(haplotype_block_file)

    encoded_df = add_haplotype_block_info(encoded_df, haplotype_blocks)
    encoded_df['MAF'] = maf_series.reindex(encoded_df.index)

    encoded_df.index.name = 'SNP'
    encoded_df.reset_index().to_csv(ENCODED_WITH_BLOCKS, index=False)

    # Use this canonical variable name from here on
    major_minor_encoded_T = encoded_df
    encoded_df = major_minor_encoded_T

    # Identify sample columns (everything except annotations)
    ANN = {'Chromosome','Position','Haplotype_Block','MAF'}
    sample_cols = [c for c in major_minor_encoded_T.columns if c not in ANN]

    # Arrays for later rendering / features
    # Normalize chromosome IDs before deriving any downstream lookups.
    local_chr_info, id_norm = build_chr_info_for_mode()
    raw_chr_ids = major_minor_encoded_T['Chromosome'].astype(str).to_numpy()
    locus_row   = major_minor_encoded_T['Position'].astype(int).to_numpy()
    haplo_row = (
        major_minor_encoded_T['Haplotype_Block']
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    maf_row     = major_minor_encoded_T['MAF'].astype(float).to_numpy()
    callrate_row = compute_call_rate_per_snp(major_minor_encoded_T).reindex(major_minor_encoded_T.index).to_numpy(np.float32)

    snps_df = pd.DataFrame({
        "snp_id": major_minor_encoded_T.index.astype(str),
        "chr": raw_chr_ids,
        "pos": locus_row
    })

    def normalize_te_for_mode(label: str) -> str | None:
        return normalize_chr_for_mode(label, id_norm)

    def normalize_gene_for_mode(label: str) -> str | None:
        return normalize_chr_for_mode(label, id_norm)

    snp_chr_norm = np.array([id_norm(c) for c in raw_chr_ids], dtype=object)
    chr_len_map = {str(k): int(v) for k, v in local_chr_info.items()}
    N = major_minor_encoded_T.shape[0]
    _require(len(raw_chr_ids) == N, "raw_chr_ids length does not match encoded SNP count.")
    _require(len(locus_row) == N, "locus_row length does not match encoded SNP count.")
    _require(len(haplo_row) == N, "haplo_row length does not match encoded SNP count.")
    _require(len(maf_row) == N, "maf_row length does not match encoded SNP count.")
    _require(np.all(np.isfinite(locus_row)), "Positions contain NaNs")
    _require(np.all(locus_row > 0), "Positions should be 1-based positive")
    _require(np.all((callrate_row >= 0.0) & (callrate_row <= 1.0)), "Callrate out of [0,1]")
    unknown_chr = sorted(set(map(str, snp_chr_norm)) - set(map(str, local_chr_info.keys())))
    _require(len(unknown_chr) == 0, f"Found chromosome labels not in chr_info: {unknown_chr[:10]}")

    te_is_te = None
    if TE_GENE_ANNOTATION_TSV:
        snps_with_te = annotate_snps_with_te(
            snps_df,
            TE_GENE_ANNOTATION_TSV,
            output_path=SNP_TE_ANNOTATION_OUT,
            id_normalizer=normalize_te_for_mode
        )
        te_is_te = snps_with_te["is_TE"].astype(np.float32).to_numpy()
        _require(te_is_te.shape[0] == N, "TE annotation array length does not match SNP count.")
        te_frac = float(te_is_te.mean()) if te_is_te.size else 0.0
        logging.info(f"[TE] SNPs with TE tag: {te_frac:.4f}")
        if te_frac <= 0.0:
            logging.warning("[TE] No SNPs received TE tags. Check TE chromosome/gene normalization.")

    genes_df = None
    snps_with_gene = None
    gene_is_genic = None
    gene_is_promoter = None
    if GENE_GFF_PATH:
        genes_df = load_gene_gff(GENE_GFF_PATH, id_normalizer=normalize_gene_for_mode)
        snps_with_gene = annotate_snps_with_genes(
            snps_df,
            GENE_GFF_PATH,
            promoter_bp=PROMOTER_BP,
            output_path=SNP_GENE_ANNOTATION_OUT,
            id_normalizer=normalize_gene_for_mode,
            genes_df=genes_df
        )
        gene_is_genic = snps_with_gene["is_genic"].astype(np.float32).to_numpy()
        gene_is_promoter = snps_with_gene["is_promoter"].astype(np.float32).to_numpy()
        _require(gene_is_genic.shape[0] == N, "Gene genic-flag array length does not match SNP count.")
        _require(gene_is_promoter.shape[0] == N, "Gene promoter-flag array length does not match SNP count.")
        gene_genic_frac = float(gene_is_genic.mean()) if gene_is_genic.size else 0.0
        gene_prom_frac = float(gene_is_promoter.mean()) if gene_is_promoter.size else 0.0
        logging.info(
            f"[GENE] SNPs genic={gene_genic_frac:.4f} promoter={gene_prom_frac:.4f}"
        )
        if gene_genic_frac <= 0.0 and gene_prom_frac <= 0.0:
            logging.warning("[GENE] No SNPs received gene/promoter tags. Check gene chromosome normalization.")

    hom_gid = hom_has = hom_size_norm = hom_gid_bits = hom_anchor_density = None
    homology_df = None
    gid_to_size: dict[int, int] = {}
    if INCLUDE_HOMOLOGY:
        if not HOMOEOLOG_PAIR_FILE or not os.path.exists(HOMOEOLOG_PAIR_FILE):
            raise FileNotFoundError(
                f"[HOMOLOGY] Set HOMOEOLOG_PAIR_FILE to a valid pair table. Missing: {HOMOEOLOG_PAIR_FILE}"
            )
        if genes_df is None or genes_df.empty or snps_with_gene is None or "gene_id" not in snps_with_gene.columns:
            raise RuntimeError("[HOMOLOGY] Gene annotations are required before building homoeolog channels.")

        pairs = load_homoeolog_pairs(HOMOEOLOG_PAIR_FILE)
        gene_to_gid, gid_to_size = build_homoeolog_groups(pairs)

        pair_gene_norm = set(pairs["gene1"]).union(set(pairs["gene2"]))
        gff_gene_norm = set(genes_df["gene_id_norm"].dropna().astype(str).values)
        hit = len(pair_gene_norm.intersection(gff_gene_norm))
        logging.info(
            f"[HOMOLOGY] genes in pair file: {len(pair_gene_norm)} | "
            f"matched to GFF: {hit} ({hit / max(1, len(pair_gene_norm)):.2%})"
        )
        if hit == 0:
            raise ValueError("[HOMOLOGY] No homoeolog pair genes matched GFF gene IDs after normalization.")

        hom_gid, hom_has, hom_size_norm = map_snp_geneid_to_homology(
            snps_with_gene["gene_id"], gene_to_gid, gid_to_size
        )
        _require(hom_gid.shape[0] == N, "hom_gid length does not match SNP count.")
        _require(hom_has.shape[0] == N, "hom_has length does not match SNP count.")
        _require(hom_size_norm.shape[0] == N, "hom_size_norm length does not match SNP count.")
        hom_gid_bits = _hashed_block_bits(hom_gid.astype(np.int64), HOM_HASH_K)
        homology_df = build_homology_spans_df(
            genes_df=genes_df,
            pairs=pairs,
            gene_to_gid=gene_to_gid,
        )
        hom_anchor_density = compute_homoeolog_anchor_density(
            genes_df=genes_df,
            gene_to_gid=gene_to_gid,
            snp_chr_norm=snp_chr_norm,
            snp_pos=locus_row,
            window_bp=HOMOEOLOG_ANCHOR_WINDOW_BP,
        )
        _require(hom_gid_bits.shape[0] == N, "hom_gid_bits length does not match SNP count.")
        _require(hom_anchor_density.shape[0] == N, "hom_anchor_density length does not match SNP count.")

        logging.info(f"[HOMOLOGY] hom_has mean={float(hom_has.mean()):.4f}")
        if float(hom_has.mean()) < 0.01:
            logging.warning("[HOMOLOGY] Almost no SNPs got homology tags; gene IDs may not match.")
        if hom_gid is not None and np.any(hom_gid > 0):
            sizes = pd.Series(hom_gid[hom_gid > 0]).map(lambda g: gid_to_size.get(int(g), 0))
            logging.info(
                f"[HOMOLOGY] group size: min={int(sizes.min())} "
                f"med={float(sizes.median()):.1f} max={int(sizes.max())}"
            )
        if hom_gid_bits is not None and hom_gid_bits.size:
            bit_means = hom_gid_bits.mean(axis=0)
            logging.info(
                f"[HOMOLOGY] hash bit means: min={float(bit_means.min()):.3f} "
                f"max={float(bit_means.max()):.3f}"
            )
            tagged = hom_gid > 0
            if np.any(tagged):
                sig_unique = int(np.unique(hom_gid_bits[tagged].astype(np.uint8, copy=False), axis=0).shape[0])
                group_unique = int(np.unique(hom_gid[tagged]).size)
                collision_rate = 1.0 - (sig_unique / max(1, group_unique))
                logging.info(
                    f"[HOMOLOGY] hash signatures: unique={sig_unique} "
                    f"groups={group_unique} collision_rate={collision_rate:.4f}"
                )
                if sig_unique < group_unique:
                    logging.warning(
                        f"[HOMOLOGY] Detected {group_unique - sig_unique} hash signature collisions at HOM_HASH_K={HOM_HASH_K}."
                    )
        if hom_anchor_density is not None:
            logging.info(
                f"[HOMOLOGY] anchor density: mean={float(hom_anchor_density.mean()):.4f} "
                f"max={float(hom_anchor_density.max()):.4f}"
            )
            if float(hom_anchor_density.max()) <= 0.0:
                logging.warning("[HOMOLOGY] Homeolog anchor density is all zero.")

    te_dist_bp = None
    if TE_GENE_ANNOTATION_TSV and os.path.exists(TE_GENE_ANNOTATION_TSV):
        try:
            te_df = pd.read_csv(TE_GENE_ANNOTATION_TSV, sep="\t")
            te_lookup = _build_interval_lookup(
                te_df,
                "chr",
                "start",
                "end",
                normalizer=normalize_te_for_mode
            )
            te_dist_bp = compute_distance_to_intervals(snp_chr_norm, locus_row, te_lookup, chr_len_map)
            if te_is_te is not None:
                te_dist_bp = te_dist_bp.copy()
                te_dist_bp[te_is_te > 0.0] = 0.0
        except Exception as e:
            logging.warning(f"TE distance computation skipped: {e}")
            te_dist_bp = None

    gene_dist_bp = None
    if genes_df is not None and not genes_df.empty:
        try:
            promoters = _build_promoters(genes_df, promoter_bp=PROMOTER_BP)
            gene_intervals = pd.concat(
                [
                    genes_df[["chr_norm", "start", "end"]],
                    promoters[["chr_norm", "start", "end"]],
                ],
                ignore_index=True
            )
            gene_lookup = _build_interval_lookup(
                gene_intervals,
                "chr_norm",
                "start",
                "end",
                normalizer=id_norm
            )
            gene_dist_bp = compute_distance_to_intervals(snp_chr_norm, locus_row, gene_lookup, chr_len_map)
            if gene_is_genic is not None or gene_is_promoter is not None:
                mask = np.zeros_like(gene_dist_bp, dtype=bool)
                if gene_is_genic is not None:
                    mask |= gene_is_genic > 0.0
                if gene_is_promoter is not None:
                    mask |= gene_is_promoter > 0.0
                gene_dist_bp = gene_dist_bp.copy()
                gene_dist_bp[mask] = 0.0
        except Exception as e:
            logging.warning(f"Gene distance computation skipped: {e}")
            gene_dist_bp = None

    # Haplo features shared across samples
    blocklen_row_kb = build_blocklen_row_kb(haplotype_blocks, haplo_row)     # per-SNP kb
    ld_row          = build_ld_score_row(major_minor_encoded_T, haplo_row, sample_cols, window=5)
    max_block_id    = int(haplotype_blocks.shape[0])
    local_chr_info, id_norm = build_chr_info_for_mode()
    row_labels = sorted(local_chr_info.keys(), key=_chr_sort_key)
    tensor_layout = build_tensor_layout_cache(
        chr_id_row=raw_chr_ids,
        locus_row=locus_row,
        row_labels=row_labels,
        id_normalizer=id_norm
    )
    boundary_flag_cache = {}
    for lab in row_labels:
        idxs = tensor_layout["indices_by_label"][str(lab)]
        n = int(idxs.size)
        if n == 0:
            continue
        bids = haplo_row[idxs]
        left_change = np.ones(n, dtype=bool)
        right_change = np.ones(n, dtype=bool)
        if n > 1:
            left_change[1:] = bids[1:] != bids[:-1]
            right_change[:-1] = bids[:-1] != bids[1:]
        boundary_flag_cache[str(lab)] = ((bids > 0) & (left_change | right_change)).astype(np.float32)

    pos_bp_cache = np.asarray(locus_row, dtype=np.float32)
    pos_enc_cache = sinusoidal_position_encoding(pos_bp_cache, d_model=POSITION_ENCODING_DIM)
    pos_enc_cache[np.isnan(pos_bp_cache)] = 0.0

    shared_quality_vector = None
    if SNP_QUALITY_CHANNEL == 'maf' and maf_row is not None and len(maf_row) == len(locus_row):
        shared_quality_vector = np.clip((maf_row * 2.0).astype(np.float32), 0.0, 1.0)
    elif SNP_QUALITY_CHANNEL == 'callrate' and callrate_row is not None and len(callrate_row) == len(locus_row):
        shared_quality_vector = np.asarray(callrate_row, dtype=np.float32)
    elif SNP_QUALITY_CHANNEL == 'missing' and callrate_row is not None and len(callrate_row) == len(locus_row):
        shared_quality_vector = 1.0 - np.asarray(callrate_row, dtype=np.float32)

    block_region = compute_block_region_features(
        haplotype_blocks=haplotype_blocks,
        haplo_row=haplo_row,
        maf_row=maf_row,
        genes=genes_df,
        id_normalizer=id_norm
    )
    major_minor_encoded_T.attrs["maf_series"] = maf_series
    if homology_df is not None:
        major_minor_encoded_T.attrs["homology_df"] = homology_df

    out_folder_combined_tensor = os.path.expanduser('images_AF_combined_d4_allele_comb_xyz_te_new_snp_rep_/tensors')
    os.makedirs(out_folder_combined_tensor, exist_ok=True)
    logging.info("Tensor-only export enabled: skipping SNP/haplotype tile generation.")

    for sample in sample_cols:
        try:
            sample_out_dir = os.path.join(out_folder_combined_tensor, sample)
            os.makedirs(sample_out_dir, exist_ok=True)
            snp_tensor_path = os.path.join(sample_out_dir, f"{sample}_tensor.npz")
            hap_tensor_path = os.path.join(sample_out_dir, f"{sample}_haplo_tensor.npz")

            snp_done = _is_nonempty_file(snp_tensor_path)
            hap_done = _is_nonempty_file(hap_tensor_path)

            if snp_done and hap_done:
                logging.info(f"[SKIP] {sample}: tensors already exist.")
                continue

            sample_series = major_minor_encoded_T[sample].values

            if not snp_done:
                save_hierarchical_tensor(
                    sample_codes=sample_series,
                    sample_name=sample,
                    out_dir=sample_out_dir,
                    chr_id_row=raw_chr_ids,
                    locus_row=locus_row,
                    maf_row=maf_row,
                    callrate_row=callrate_row,
                    haplo_row=haplo_row,
                    blocklen_row_kb=blocklen_row_kb,
                    ld_row=ld_row,
                    max_block_id=max_block_id,
                    chr_info_override=local_chr_info,
                    id_normalizer=id_norm,
                    quality_channel=SNP_QUALITY_CHANNEL,
                    include_sg_channel=INCLUDE_SG_CHANNEL,
                    hom_has=hom_has,
                    hom_size_norm=hom_size_norm,
                    hom_gid=hom_gid,
                    hom_gid_bits=hom_gid_bits,
                    hom_anchor_density=hom_anchor_density,
                    hom_hash_k=HOM_HASH_K,
                    te_is_te=te_is_te,
                    te_dist_bp=te_dist_bp,
                    gene_is_genic=gene_is_genic,
                    gene_is_promoter=gene_is_promoter,
                    gene_dist_bp=gene_dist_bp,
                    block_gene_count_norm=block_region.get("block_gene_count_norm"),
                    block_mean_maf_norm=block_region.get("block_mean_maf_norm"),
                    block_snp_density_norm=block_region.get("block_snp_density_norm"),
                    layout_cache=tensor_layout,
                    pos_enc_cache=pos_enc_cache,
                    quality_vector_override=shared_quality_vector,
                    boundary_flag_cache=boundary_flag_cache
                )
            else:
                logging.info(f"[SKIP] {sample}: SNP tensor exists.")

            if not hap_done:
                save_haplotype_tensor(
                    haplo_row=haplo_row,
                    blocklen_row_kb=blocklen_row_kb,
                    ld_row=ld_row,
                    sample_name=sample,
                    out_dir=sample_out_dir,
                    chr_id_row=raw_chr_ids,
                    locus_row=locus_row,
                    max_block_id=max_block_id,
                    chr_info_override=local_chr_info,
                    id_normalizer=id_norm,
                    include_sg_channel=INCLUDE_SG_CHANNEL,
                    hash_block_id_k=HASH_BLOCK_ID_K,
                    layout_cache=tensor_layout,
                    pos_enc_cache=pos_enc_cache
                )
            else:
                logging.info(f"[SKIP] {sample}: haplotype tensor exists.")

            logging.info(f"[DONE] {sample}: tensor export complete.")

        except Exception as e:
            logging.error(f"[FAIL] {sample}: {e}")

    return major_minor_encoded_T, haplotype_blocks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Chromomap tensors.")
    parser.add_argument("--color-mode", choices=sorted(VALID_COLOR_MODES),
                        help="Override the chromomap coloring (dosage or allele_combination).")
    parser.add_argument(
        "--diagnostics-plots",
        action="store_true",
        help="Run optional exploratory plotting diagnostics after tensor export."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.color_mode:
        set_color_mode(args.color_mode)

    encoded_df, haplotype_blocks = main()
    if not args.diagnostics_plots:
        raise SystemExit(0)
