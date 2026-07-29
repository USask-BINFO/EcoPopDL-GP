# Ensure this is the point
import os
import glob
import math
import json
import copy
import inspect
import logging
from collections import defaultdict
from typing import Any, List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas_plink import read_plink
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from gxe_transformer_temporal import (
    preprocess_environmental_data,
    GxE_Transformer_Tensor,
    DualBranchGxE,
    HOTSPOT_FOCUS_BIAS,
    token_dropout,
    TemporalEnvironmentalEncoder,
    TemporalConvEncoder,
    TemporalPyramidEncoder,
    WideEnvMLPEncoder,
    LowRankBilinear,
)


def _build_env_lookup(meta: pd.DataFrame, metadata_key_col: str) -> pd.DataFrame:
    """
    Build a deduplicated mapping from metadata keys to location/year codes.
    """
    try:
        env_map = meta.set_index(metadata_key_col)[["Location_Code", "Year_Code"]]
        env_map = env_map.groupby(level=0, sort=False).first()
    except Exception:
        env_map = pd.DataFrame(columns=["Location_Code", "Year_Code"])
    return env_map


def _build_env_index_map(
    sample_ids: List[str],
    sample_key_to_sid: Dict[str, str],
    metadata_indexed: pd.DataFrame,
    env_data_dict: Optional[Dict[str, np.ndarray]]
) -> Tuple[Dict[str, int], List[str]]:
    """
    Map dataset keys to rows in env_data_dict['temporal'] via env_data_dict['key_to_idx'].
    Returns (key -> env_idx, missing_keys list).
    """
    if not env_data_dict or "key_to_idx" not in env_data_dict:
        return {}, list(sample_ids)
    key_to_idx = env_data_dict.get("key_to_idx", {}) or {}
    idx_map: Dict[str, int] = {}
    missing: List[str] = []
    for key in sample_ids:
        sid = sample_key_to_sid.get(key, key)
        idx = key_to_idx.get(str(key))
        if idx is None:
            idx = key_to_idx.get(str(sid))
        if idx is None and str(key) in metadata_indexed.index:
            try:
                row = metadata_indexed.loc[str(key)]
                env_key = row.get("EnvKey", None)
                if env_key is None or (isinstance(env_key, float) and np.isnan(env_key)):
                    sid_col = row.get("SampleID_str", row.get("SampleID", sid))
                    loc_code = row.get("Location_Code", None)
                    year_code = row.get("Year_Code", None)
                    if loc_code is not None and year_code is not None:
                        env_key = f"{sid_col}|{int(loc_code)}|{int(year_code)}"
                if env_key is not None:
                    idx = key_to_idx.get(str(env_key))
            except Exception:
                idx = None
        if idx is None:
            missing.append(str(key))
        else:
            idx_map[str(key)] = int(idx)
    return idx_map, missing


def _build_env_pair_map(
    location_ids: np.ndarray,
    year_ids: np.ndarray,
    n_locations: int,
    n_years: int
) -> Tuple[Dict[Tuple[int, int], int], np.ndarray, np.ndarray]:
    """
    Build a compact env_id map from observed (location, year) pairs.
    Returns (pair_to_id, env_pair_ids, lut).
    """
    pair_to_id: Dict[Tuple[int, int], int] = {}
    pair_ids: List[int] = []
    for loc_id, yr_id in zip(location_ids, year_ids):
        key = (int(loc_id), int(yr_id))
        if key not in pair_to_id:
            pair_to_id[key] = len(pair_to_id)
        pair_ids.append(pair_to_id[key])
    env_pair_ids = np.array(pair_ids, dtype=np.int64)
    lut = np.full((max(1, n_locations), max(1, n_years)), -1, dtype=np.int64)
    for (loc_id, yr_id), env_id in pair_to_id.items():
        if 0 <= loc_id < lut.shape[0] and 0 <= yr_id < lut.shape[1]:
            lut[loc_id, yr_id] = env_id
    return pair_to_id, env_pair_ids, lut


def _resolve_env_id(loc_id: int, year_id: int) -> int:
    """
    Resolve a compact env_id for a (location, year) pair using observed pairs.
    """
    if ENV_PAIR_LUT is not None:
        if 0 <= loc_id < ENV_PAIR_LUT.shape[0] and 0 <= year_id < ENV_PAIR_LUT.shape[1]:
            env_id = int(ENV_PAIR_LUT[loc_id, year_id])
            if env_id >= 0:
                return env_id
    if ENV_PAIR_TO_ID:
        return int(ENV_PAIR_TO_ID.get((loc_id, year_id), 0))
    return int((loc_id * N_YEARS + year_id) % max(1, NUM_ENVIRONMENTS))


def _to_int_scalar(x, default=0):
    """
    Safely coerce scalars (or pandas/NumPy wrappers) to Python int with a default fallback.
    """
    import pandas as pd
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[0]
        if hasattr(x, "item"):
            x = x.item()
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _canonicalize_location_name(value: Any) -> str:
    """
    Normalize known spelling variants so metadata/env join keys stay aligned.
    """
    alias_map = {
        "pathancheru": "Pathancheru",
        "patancheru": "Pathancheru",
        "patancheruvu": "Pathancheru",
    }
    text = str(value).strip()
    key = "".join(ch for ch in text.lower() if ch.isalnum())
    return alias_map.get(key, text)


def _normalize_location_column(df: pd.DataFrame, frame_name: str) -> None:
    """
    Apply canonical location naming in-place and log replacement counts.
    """
    if "Location" not in df.columns:
        return
    before = df["Location"].astype(str).str.strip()
    after = before.map(_canonicalize_location_name)
    replaced = int((before != after).sum())
    if replaced > 0:
        logging.info("Normalized %d location aliases in %s.", replaced, frame_name)
    df["Location"] = after


def _safe_param_count(model: nn.Module) -> int:
    """
    Count parameters while tolerating uninitialized Lazy modules (e.g., dosage branch).
    """
    total = 0
    for p in model.parameters():
        try:
            total += p.numel()
        except ValueError:
            continue
    return total


def _load_dosage_pca_assets():
    """
    Load optional PCA components and scaler stats for the dosage branch from disk.
    """
    comps = None
    mean = None
    std = None
    if DOSAGE_PCA_COMPONENTS_PATH:
        try:
            comps = np.load(DOSAGE_PCA_COMPONENTS_PATH, allow_pickle=False)
            logging.info("Loaded dosage PCA components: %s shape=%s", DOSAGE_PCA_COMPONENTS_PATH, getattr(comps, "shape", None))
        except Exception as e:
            logging.warning("Failed to load dosage PCA components from %s: %s", DOSAGE_PCA_COMPONENTS_PATH, e)
    if DOSAGE_PCA_MEAN_PATH:
        try:
            mean = np.load(DOSAGE_PCA_MEAN_PATH, allow_pickle=False)
            logging.info("Loaded dosage PCA mean: %s shape=%s", DOSAGE_PCA_MEAN_PATH, getattr(mean, "shape", None))
        except Exception as e:
            logging.warning("Failed to load dosage PCA mean from %s: %s", DOSAGE_PCA_MEAN_PATH, e)
    if DOSAGE_PCA_STD_PATH:
        try:
            std = np.load(DOSAGE_PCA_STD_PATH, allow_pickle=False)
            logging.info("Loaded dosage PCA std: %s shape=%s", DOSAGE_PCA_STD_PATH, getattr(std, "shape", None))
        except Exception as e:
            logging.warning("Failed to load dosage PCA std from %s: %s", DOSAGE_PCA_STD_PATH, e)
    return comps, mean, std


def _save_dosage_pca_assets(comps, mean, std):
    """
    Save PCA components/mean/std for reuse by the dosage branch (matches PLINK preprocessing).
    Only saves when corresponding DOSAGE_PCA_*_PATH values are set.
    """
    try:
        if comps is not None and DOSAGE_PCA_COMPONENTS_PATH:
            np.save(DOSAGE_PCA_COMPONENTS_PATH, comps)
            logging.info("Saved dosage PCA components to %s (shape=%s)", DOSAGE_PCA_COMPONENTS_PATH, getattr(comps, "shape", None))
        if mean is not None and DOSAGE_PCA_MEAN_PATH:
            np.save(DOSAGE_PCA_MEAN_PATH, mean)
            logging.info("Saved dosage PCA mean to %s (shape=%s)", DOSAGE_PCA_MEAN_PATH, getattr(mean, "shape", None))
        if std is not None and DOSAGE_PCA_STD_PATH:
            np.save(DOSAGE_PCA_STD_PATH, std)
            logging.info("Saved dosage PCA std to %s (shape=%s)", DOSAGE_PCA_STD_PATH, getattr(std, "shape", None))
    except Exception as e:
        logging.warning("Failed to save dosage PCA assets: %s", e)


def compute_tensor_dosage_pca(dataset, max_samples: int = 4000, n_components: Optional[int] = None):
    """
    Approximate PLINK PCA using the dosage channel from Chromomap tensors.
    Returns (components, mean, std) or (None, None, None) on failure.
    """
    if n_components is None:
        try:
            n_components = int(SIMPLE_GENO_N_PCS)
        except Exception:
            n_components = 200
    if not hasattr(dataset, "dosage_idx") or dataset.dosage_idx is None:
        logging.warning("Tensor PCA skipped: dataset is missing dosage_idx.")
        return None, None, None
    flats: List[np.ndarray] = []
    n_take = min(len(dataset), max_samples)
    for i in range(n_take):
        try:
            genomic_tensor, mask_tensor, *_ = dataset[i]
        except Exception as e:
            logging.warning("Tensor PCA: failed to read sample %d: %s", i, e)
            continue
        if genomic_tensor.dim() != 3:
            continue
        dosage = genomic_tensor[..., dataset.dosage_idx]
        if mask_tensor is not None:
            dosage = dosage.masked_fill(mask_tensor, 0.0)
        flats.append(dosage.reshape(-1).float().cpu().numpy())
    if len(flats) < 2:
        logging.warning("Tensor PCA skipped: not enough samples with dosage channel (got=%d).", len(flats))
        return None, None, None
    X = np.stack(flats, axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)
    n_comp = min(X_std.shape[1], max(1, min(n_components, X_std.shape[0])))
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(X_std)
    comps = pca.components_.astype(np.float32)
    mean_arr = scaler.mean_.astype(np.float32)
    std_arr = scaler.scale_.astype(np.float32)
    logging.info(
        "Tensor-based dosage PCA: samples=%d, flat_dim=%d, n_components=%d",
        X.shape[0],
        X.shape[1],
        comps.shape[0]
    )
    return comps, mean_arr, std_arr


def _build_dosage_matrix_from_tensors(
    sample_ids: List[str],
    sample_key_to_sid: Dict[str, str],
    tensor_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Load dosage channel from Chromomap tensors and return (matrix, ids) ordered by sample_ids.
    Falls back to channel 0 when feature names are missing.
    """
    mats: List[np.ndarray] = []
    ids: List[str] = []
    dosage_idx: Optional[int] = None
    for key in sample_ids:
        sid = sample_key_to_sid.get(key, key)
        path = os.path.join(tensor_dir, sid, f"{sid}_tensor.npz")
        if not os.path.exists(path):
            logging.warning("Tensor NPZ not found for %s: %s", sid, path)
            continue
        try:
            z = np.load(path, allow_pickle=False)
            tensor = z["tensor"].astype(np.float32)
            mask = z["mask"].astype(np.float32)
            pad_mask = mask <= 0.0
            if dosage_idx is None:
                if "feature_names_bytes" in z:
                    try:
                        names_str = bytes(z["feature_names_bytes"].tolist()).decode("ascii")
                        names = names_str.split(",") if names_str else []
                        if "dosage" in names:
                            dosage_idx = names.index("dosage")
                    except Exception:
                        dosage_idx = None
                if dosage_idx is None:
                    dosage_idx = 0
            if tensor.shape[-1] <= dosage_idx:
                logging.warning("Dosage index %d out of bounds for %s (feat_dim=%d)", dosage_idx, sid, tensor.shape[-1])
                continue
            dosage = tensor[..., dosage_idx]
            dosage = np.nan_to_num(dosage, nan=0.0)
            dosage = np.where(pad_mask, 0.0, dosage)
            mats.append(dosage.reshape(-1))
            ids.append(str(sid))
        except Exception as e:
            logging.warning("Failed to load tensor for %s: %s", sid, e)
            continue
    if not mats:
        raise RuntimeError("No dosage tensors could be loaded; cannot build dosage feature source.")
    return np.stack(mats, axis=0), ids


def diagnose_cv_leakage(meta, metadata_key_col, sample_key_to_sid, all_ids, CV_FOLDS, SEED):
    """
    Print diagnostics verifying that CV folds do not leak genotypes.
    """
    print("\n" + "=" * 80)
    print("CV LEAKAGE DIAGNOSTIC")
    print("=" * 80)

    _ = meta
    _ = metadata_key_col
    id_to_geno = {str(k): str(sample_key_to_sid.get(k, k)) for k in all_ids}
    genotypes = [id_to_geno[str(k)] for k in all_ids]
    unique_genos = sorted(set(genotypes))
    n_splits = min(CV_FOLDS, len(unique_genos))
    if n_splits < 2:
        raise RuntimeError("Not enough unique genotypes for grouped CV.")
    print(f"\nTotal CV samples: {len(all_ids)}")
    print(f"Unique genotypes: {len(unique_genos)}")
    print(f"CV folds: {n_splits} (GroupKFold by genotype)")

    splitter = GroupKFold(n_splits=n_splits)
    fold_iter = list(splitter.split(all_ids, groups=genotypes))

    print(f"\nAnalyzing {len(fold_iter)} CV folds...\n")
    total_leakage = 0
    for fold_num, (tr_idx, te_idx) in enumerate(fold_iter, start=1):
        tr_ids = [all_ids[i] for i in tr_idx]
        te_ids = [all_ids[i] for i in te_idx]
        tr_genos = set(id_to_geno[str(k)] for k in tr_ids)
        te_genos = set(id_to_geno[str(k)] for k in te_ids)
        leaked_genos = tr_genos.intersection(te_genos)
        leakage_pct = 100 * len(leaked_genos) / len(te_genos) if te_genos else 0
        leaked_samples = set(map(str, tr_ids)).intersection(set(map(str, te_ids)))
        print(f"FOLD {fold_num}:")
        print(f"  Train: {len(tr_ids)} samples, {len(tr_genos)} unique genotypes")
        print(f"  Test:  {len(te_ids)} samples, {len(te_genos)} unique genotypes")
        if leaked_genos:
            total_leakage += len(leaked_genos)
            print(f"  Ã¢Å¡ Ã¯Â¸Â  LEAKAGE: {len(leaked_genos)} genotypes ({leakage_pct:.1f}%) appear in both train and test")
            print(f"      Example leaked genotypes: {list(leaked_genos)[:3]}")
        if leaked_samples:
            print(f"  Ã¢Å¡ Ã¯Â¸Â  SAMPLE OVERLAP: {len(leaked_samples)} samples appear in both train and test")
            print(f"      Example overlapping samples: {list(leaked_samples)[:3]}")
        if not leaked_genos and not leaked_samples:
            print(" âœ“ NO LEAKAGE (test uses unseen genotypes)")
        print()

    print("=" * 80)
    if total_leakage > 0:
        print(f"Ã¢ÂÅ’ PROBLEM FOUND: {total_leakage} total genotype leakages across folds")
        print(
            "  Ã¢â‚¬Â¢ Validation few points may overstate generalization because genotypes reoccur."
            "   Use genotype-based CV with strict grouping."
        )
        return True
    else:
        print("âœ“â€œ No genotype leakage detected. CV split looks clean.")
        print("If validation >> test performance, inspect environment/test distribution mismatch.")
        return False


def diagnose_and_fix_cv_split(
    meta,
    metadata_key_col,
    sample_key_to_sid,
    all_ids,
    CV_FOLDS,
    SEED,
    MULTI_ENV
):
    """
    Diagnose genotype leakage and return a genotype-based split.
    """
    print("\n" + "=" * 80)
    print("CV SPLIT DIAGNOSTIC & FIX")
    print("=" * 80)

    id_to_geno = {str(k): str(sample_key_to_sid.get(k, k)) for k in all_ids}
    print(f"\nData summary: {len(all_ids)} rows, {len(set(id_to_geno.values()))} unique genotypes")

    print("\nCorrected CV: group by genotype only")
    unique_genos_list = sorted(set(id_to_geno.values()))
    n_splits = min(CV_FOLDS, len(unique_genos_list))
    if n_splits < 2:
        raise RuntimeError("Not enough genotypes for CV")
    geno_to_group = {geno: i for i, geno in enumerate(unique_genos_list)}
    corrected_splitter = GroupKFold(n_splits=n_splits)
    groups_array = np.array([geno_to_group[id_to_geno[str(k)]] for k in all_ids])
    corrected_splits = list(corrected_splitter.split(all_ids, groups=groups_array))

    train_ids_list = []
    val_ids_list = []
    seen_test = set()
    seen_val = set()
    for fold_num, (trval_idx, te_idx) in enumerate(corrected_splits, start=1):
        trval_ids = [all_ids[i] for i in trval_idx]
        te_ids = [all_ids[i] for i in te_idx]
        trval_genos = sorted(set(id_to_geno[str(k)] for k in trval_ids))
        n_train = int(0.8 * len(trval_genos))
        train_genos = set(trval_genos[:n_train])
        tr_ids = [k for k in trval_ids if id_to_geno[str(k)] in train_genos]
        val_ids = [k for k in trval_ids if id_to_geno[str(k)] not in train_genos]
        tr_set = set(map(str, tr_ids))
        val_set = set(map(str, val_ids))
        te_set = set(map(str, te_ids))
        overlap = (tr_set & val_set) | (tr_set & te_set) | (val_set & te_set)
        if overlap:
            raise RuntimeError(f"Fold {fold_num}: overlap detected among train/val/test: {list(overlap)[:3]}")
        cross_test = te_set & seen_test
        if cross_test:
            raise RuntimeError(f"Fold {fold_num}: test samples repeat across folds: {list(cross_test)[:3]}")
        cross_val = val_set & seen_val
        if cross_val:
            logging.warning(f"Fold {fold_num}: validation samples repeat across folds: {list(cross_val)[:3]}")
        seen_test.update(te_set)
        seen_val.update(val_set)
        train_ids_list.append(tr_ids)
        val_ids_list.append(val_ids)
        print(f"  Fold {fold_num}: {len(tr_ids)} train / {len(val_ids)} val / {len(te_ids)} test (all genotypes disjoint)")

    print("\nâœ“â€œ Genotype-based CV ready (no leakage). Validation RÂ² will drop but match test RÂ².")
    return train_ids_list, val_ids_list, corrected_splits


def use_corrected_cv_in_main(
    meta,
    metadata_key_col,
    sample_key_to_sid,
    all_ids,
    CV_FOLDS,
    SEED,
    MULTI_ENV,
    logging_module
):
    return diagnose_and_fix_cv_split(
        meta,
        metadata_key_col,
        sample_key_to_sid,
        all_ids,
        CV_FOLDS,
        SEED,
        MULTI_ENV
    )


def run_genotype_cv(
    all_ids: List[str],
    sample_key_to_sid: Dict[str, str],
    cv_folds: int,
    seed: int
) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    GroupKFold on genotype (SampleID) so no genotype appears in both train and test.
    Returns list of (train_ids, val_ids, test_ids) for each fold.
    """
    genotypes = [sample_key_to_sid.get(k, k) for k in all_ids]
    unique_genos = sorted(set(genotypes))
    n_splits = min(cv_folds, len(unique_genos))
    if n_splits < 2:
        raise RuntimeError("Not enough unique genotypes for grouped CV.")
    splitter = GroupKFold(n_splits=n_splits)
    fold_triplets: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(all_ids, groups=genotypes), start=1):
        te_ids = [all_ids[i] for i in te_idx]
        trval_ids = [all_ids[i] for i in tr_idx]
        trval_genos = sorted(set(sample_key_to_sid.get(k, k) for k in trval_ids))
        if len(trval_genos) < 2:
            raise RuntimeError("Not enough genotypes to split train/val within fold.")
        import random
        rng = random.Random(seed + fold_idx)
        rng.shuffle(trval_genos)
        n_val = max(1, int(0.2 * len(trval_genos)))
        val_genos = set(trval_genos[:n_val])
        train_genos = set(trval_genos[n_val:])
        tr_ids = [k for k in trval_ids if sample_key_to_sid.get(k, k) in train_genos]
        va_ids = [k for k in trval_ids if sample_key_to_sid.get(k, k) in val_genos]
        fold_triplets.append((tr_ids, va_ids, te_ids))
    logging.info(f"Using genotype-based GroupKFold with {n_splits} folds over {len(unique_genos)} genotypes.")
    return fold_triplets


def run_within_genotype_env_holdout_cv(
    meta: pd.DataFrame,
    split_key_col: str,
    sampleid_col: str,
    n_folds: int,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 20,
    min_test: int = 1,
    min_train: int = 1,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Within-genotype CV: for each genotype (SampleID), hold out a fraction of its environment
    rows (SplitKeys) for test (and optionally val). Ensures each genotype appears in train.
    """
    meta2 = meta.copy()
    meta2[split_key_col] = meta2[split_key_col].astype(str)
    meta2[sampleid_col] = meta2[sampleid_col].astype(str)
    grouped = meta2.groupby(sampleid_col)[split_key_col].apply(list).to_dict()
    genos = sorted(grouped.keys())
    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for fold in range(1, n_folds + 1):
        rng = np.random.default_rng(seed + fold)
        tr: List[str] = []
        va: List[str] = []
        te: List[str] = []
        for g in genos:
            keys = list(grouped[g])
            if len(keys) < (min_train + min_test):
                tr.extend(keys)
                continue
            rng.shuffle(keys)
            n_test = max(min_test, int(math.ceil(test_frac * len(keys))))
            n_test = min(n_test, len(keys) - min_train)
            remaining = len(keys) - n_test
            n_val = max(0, int(math.ceil(val_frac * len(keys))))
            n_val = min(n_val, max(0, remaining - min_train))
            te_keys = keys[:n_test]
            va_keys = keys[n_test:n_test + n_val]
            tr_keys = keys[n_test + n_val:]
            if len(tr_keys) < min_train:
                tr_keys = tr_keys + va_keys
                va_keys = []
                if len(tr_keys) < min_train:
                    tr_keys = tr_keys + te_keys
                    te_keys = []
            tr.extend(tr_keys)
            va.extend(va_keys)
            te.extend(te_keys)
        # enforce uniqueness/disjointness in case of upstream duplicates
        tr_set = set(map(str, tr))
        va_set = set(map(str, va))
        te_set = set(map(str, te))
        # priority: test -> val -> train
        va_set -= te_set
        tr_set -= (va_set | te_set)
        folds.append((list(tr_set), list(va_set), list(te_set)))
    return folds


def check_within_genotype_split(
    meta: pd.DataFrame,
    fold_triplets: List[Tuple[List[str], List[str], List[str]]],
    split_key_col: str = "SplitKey",
    sampleid_col: str = "SampleID_str"
) -> None:
    """
    Sanity checks for within-genotype env holdout:
    - train/val/test disjoint within fold
    - each genotype retains train and test rows when possible
    """
    meta2 = meta.copy()
    meta2[split_key_col] = meta2[split_key_col].astype(str)
    meta2[sampleid_col] = meta2[sampleid_col].astype(str)
    key_to_sid = meta2.set_index(split_key_col)[sampleid_col].to_dict()
    for fold_idx, (tr_ids, va_ids, te_ids) in enumerate(fold_triplets, start=1):
        tr_set = set(map(str, tr_ids))
        va_set = set(map(str, va_ids))
        te_set = set(map(str, te_ids))
        overlap = (tr_set & va_set) | (tr_set & te_set) | (va_set & te_set)
        if overlap:
            raise RuntimeError(
                f"[within-genotype] Fold {fold_idx}: overlap detected among train/val/test: {list(overlap)[:3]}"
            )
        # per-genotype coverage check
        geno_to_keys = defaultdict(lambda: {"tr": 0, "te": 0})
        for k in tr_set:
            sid = key_to_sid.get(k, "")
            geno_to_keys[sid]["tr"] += 1
        for k in te_set:
            sid = key_to_sid.get(k, "")
            geno_to_keys[sid]["te"] += 1
        bad = [sid for sid, counts in geno_to_keys.items() if counts["tr"] == 0 or counts["te"] == 0]
        if bad:
            logging.warning(
                "[within-genotype] Fold %d: %d genotypes missing train/test coverage (examples: %s)",
                fold_idx,
                len(bad),
                bad[:3],
            )


def check_cv_fold_overlap(
    fold_triplets: List[Tuple[List[str], List[str], List[str]]],
    context: str = "cv"
) -> None:
    """
    Ensure no sample overlaps within fold (train/val/test) and no test reuse across folds.
    """
    seen_test: Set[str] = set()
    seen_val: Set[str] = set()
    for fold_idx, (tr_ids, va_ids, te_ids) in enumerate(fold_triplets, start=1):
        tr_set = set(map(str, tr_ids))
        val_set = set(map(str, va_ids))
        te_set = set(map(str, te_ids))
        overlap = (tr_set & val_set) | (tr_set & te_set) | (val_set & te_set)
        if overlap:
            raise RuntimeError(
                f"[{context}] Fold {fold_idx}: overlap detected among train/val/test: {list(overlap)[:3]}"
            )
        cross_test = te_set & seen_test
        if cross_test:
            raise RuntimeError(
                f"[{context}] Fold {fold_idx}: test samples repeat across folds: {list(cross_test)[:3]}"
            )
        cross_val = val_set & seen_val
        if cross_val:
            logging.warning(
                f"[{context}] Fold {fold_idx}: validation samples repeat across folds: {list(cross_val)[:3]}"
            )
        seen_test.update(te_set)
        seen_val.update(val_set)
# ---------------------------
# CONFIG
# ---------------------------
METADATA_FILE = os.environ.get("ECOPOP_METADATA_FILE", "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Phenotype/Phenotype_files/oil_db_mean.txt")
ENVIRONMENT_FILE = os.environ.get("ECOPOP_ENVIRONMENT_FILE", "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Environment/D4/d4_env_matrix.csv")

TARGET_COL = os.environ.get("ECOPOP_TARGET_COL", "OIL_DB")

# ---- Opt-in multi-trait auxiliary learning (hard parameter sharing) ----
# When ECOPOP_AUX_TARGETS is unset, AUX_TARGETS == [] and USE_AUX is False, so the
# engine is byte-identical to single-trait training: no aux heads are built, the
# aux-loss guard in train_epoch_regularized is skipped, and n_aux_targets=0 is passed
# to every model. Only TARGET_COL is ever evaluated/reported; aux traits are an extra
# training signal and are NEVER fed as model input.
AUX_TARGETS = [c.strip() for c in os.environ.get("ECOPOP_AUX_TARGETS", "").split(",") if c.strip()]
AUX_PHENO_PATH = os.environ.get("ECOPOP_AUX_PHENO", METADATA_FILE)
AUX_LOSS_WEIGHT = float(os.environ.get("ECOPOP_AUX_WEIGHT", "0.2"))
USE_AUX = len(AUX_TARGETS) > 0

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
PRETRAIN_GENOMIC_SIMCLR = False   # self-supervised genomic contrastive pretrain before supervision
SIMCLR_EPOCHS = 15
SIMCLR_LR = 3e-4
SIMCLR_TEMP = 0.1
SIMCLR_TOKEN_DROP = 0.2
SIMCLR_FEATURE_NOISE = 0.01
SIMCLR_USE_VICREG = False  # use VICReg loss (better for small batches); if False, use SimCLR

# Residual-focus two-stage training (main effects -> GxE residuals)
USE_RESIDUAL_FOCUS_ARCH = False
RESIDUAL_GATE_INIT = 0.01
MAIN_HEAD_DROPOUT = 0.15
INTERACTION_HEAD_DROPOUT = 0.35
DISTANCE_LOG1P = True  # set False if te_dist/gene_dist already log-scaled
USE_ENV_ANOMALIES = True  # use env deltas from mean in GxE interaction path
STAGE2_LEARNING_RATE = 5e-5
MAIN_WEIGHT_DECAY = float(os.environ.get("ECOPOP_WEIGHT_DECAY", "0.02"))  # [EXP] per-dataset regularization tuning
GXE_WEIGHT_DECAY = 0.1    # Stage 2: stronger regularization for GxE residuals

GXE_DROPOUT = float(os.environ.get("ECOPOP_GXE_DROPOUT", "0.4"))  # [EXP] per-dataset regularization tuning

SEED = int(os.environ.get("ECOPOP_SEED", "20"))
USE_HABE = os.environ.get("ECOPOP_USE_HABE", "1") == "1"  # [EXP] -HABE ablation
USE_POPULATION_EMBEDDING = os.environ.get("ECOPOP_USE_POP", "1") == "1"  # [EXP] -Population ablation
ABLATE_WEATHER = os.environ.get("ECOPOP_ABLATE_WEATHER", "0") == "1"  # [EXP] -Weather ablation
ENV_WINDOW_FRACTION = float(os.environ.get("ECOPOP_ENV_WINDOW_FRAC", "1.0"))  # [EXP] early-selection partial season
ENV_BLOCKED_MODES = [m.strip() for m in os.environ.get("ECOPOP_ENV_BLOCKED_MODES", "year,location,loc_year").split(",") if m.strip()]
def _ecopop_env_hook(_e):
    """[EXP] apply weather ablation / partial-season truncation to an env time-series tensor."""
    try:
        if ABLATE_WEATHER:
            return torch.zeros_like(_e)
        if ENV_WINDOW_FRACTION < 1.0 and hasattr(_e, "dim") and _e.dim() >= 1 and _e.shape[0] > 1:
            _k = max(1, int(round(ENV_WINDOW_FRACTION * _e.shape[0])))
            if _k < _e.shape[0]:
                _e = _e.clone(); _e[_k:] = 0.0
    except Exception:
        pass
    return _e
STANDARDIZE_TARGET = True     # standardize target using train-set mean/std for stability
USE_ENV_ZSCORE = os.environ.get("ECOPOP_USE_ENV_ZSCORE", "0") == "1"
LR_WARMUP_EPOCHS = 5          # epochs used to linearly warm up the LR
LR_MIN_LR_FACTOR = 0.1         # cosine decay will anneal towards base_lr * LR_MIN_LR_FACTOR
USE_SNAPSHOT_ENSEMBLE = False  # snapshot ensembling with cyclical LR
SNAPSHOT_CYCLES = 3
SNAPSHOT_DIR = "snapshots"
LOSS_FUNCTION = "huber"       # choose between "mse", "huber", "quantile"
HUBER_DELTA = 1.0
QUANTILE_ALPHA = 0.5
SNAPSHOT_CYCLE_LENGTH = 0  # disable snapshot ensembling; keep only the best model

MIXUP_ALPHA = 0.0              # alpha hyperparameter for genomic+env mixup augmentation (0=off)
USE_CHR_POOLING = True        # optional per-chromosome window pooling for tensor inputs (preserves chromosome axis)
CHR_POOL_WINDOW = 1000          # window size (tokens) when USE_CHR_POOLING is enabled
USE_TOKEN_DROPOUT = False     # set True to apply token-dropout augmentation on genomic tensors during training
TOKEN_DROPOUT_P = 0.10
TOKEN_DROPOUT_KEEP_FIRST = False
USE_CHANNEL_DROPOUT = False     # set True to apply structured feature/channel dropout during training
CHANNEL_GROUP_DROP_P = 0.20
CHANNEL_DROP_P = 0.0
ENV_MOD_DROPOUT_P = 0.0         # probability to drop env modulation (zero env_ts) during training
USE_BLOCK_MASKING = False       # set True to apply CutMix-style contiguous masking per chromosome
BLOCK_MASK_P_APPLY = 0.8
BLOCK_MASK_FRAC_RANGE = (0.05, 0.15)
BLOCK_MASK_NUM_BLOCKS = 1
BLOCK_MASK_MIN_TOKENS = 16
BLOCK_MASK_MIN_KEEP_TOTAL = None  # None -> defaults to max(64, 2% of C*T)
BLOCK_MASK_KEEP_FIRST = False
BLOCK_MASK_CHR_MODE = "proportional"  # "proportional" or "uniform"
# Optional additional downsampling inside ChromoAwareTransformer for very long sequences (tensor model)
CHR_DOWNSAMPLE_STRIDE = 2
CHR_DOWNSAMPLE_KERNEL = None
USE_DUAL_BRANCH_MODEL = True   # enable additive+interaction fusion wrapper (helps across simple/complex traits)
ADDITIVE_BRANCH_HIDDEN = 128
DUAL_GATE_HIDDEN = 32
DUAL_GATE_DROPOUT = 0.1
USE_DOSAGE_BRANCH = os.environ.get("ECOPOP_USE_ADDITIVE", "1") == "1"
DOSAGE_SOURCE = "plink"       # "tensor" (from Chromomap channel) or "plink" (from PLINK SNP matrix)
USE_DOSAGE_PCA = False        # False => use flattened dosage (all SNP positions) without PCA projection in dosage branch
DOSAGE_BRANCH_HIDDEN = 128
DOSAGE_GATE_HIDDEN = 32
DOSAGE_GATE_DROPOUT = 0.3
DOSAGE_PCA_COMPONENTS_PATH = None  # npy path with PCA components [n_components, n_snps] (optional; used when USE_DOSAGE_PCA=True)
DOSAGE_PCA_MEAN_PATH = None       # npy path with per-SNP mean used for PCA/standardization
DOSAGE_PCA_STD_PATH = None        # npy path with per-SNP std
DOSAGE_PCA_CENTER = True
DOSAGE_PCA_SCALE = True
DOSAGE_BLEND_PRIOR = 0.9        # initial bias toward dosage path for learnable gate
_ECOPOP_DW = str(os.environ.get("ECOPOP_DOSAGE_WEIGHT", "0.9")).strip().lower()
# [EXP] fixed weight on the dosage/additive path, OR "learn"/"none" for the self-adjusting blend.
DOSAGE_FIXED_WEIGHT = None if _ECOPOP_DW in ("learn", "none", "auto", "gate", "") else float(_ECOPOP_DW)
# Aliases for quick copy/paste (matches doc names)
DOSAGEPCACOMPONENTSPATH = DOSAGE_PCA_COMPONENTS_PATH
DOSAGEPCAMEANPATH = DOSAGE_PCA_MEAN_PATH
DOSAGEPCASTDPATH = DOSAGE_PCA_STD_PATH
DOSAGEFIXEDWEIGHT = DOSAGE_FIXED_WEIGHT

# Sequence sizes
# Transformer sizes
EMBED_DIM = 128
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
FF_DIM = 512

# Dataloader
NUM_WORKERS = 8
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-3
USE_CV = True         # if True, run K-fold CV instead of single train/val/test
# CV mode selection (use at most one of the two specialized modes)
USE_GENOTYPE_CV = True  # group CV by genotype (SampleID) = untested genotypes across folds
EVAL_TIER_MODE = os.environ.get("ECOPOP_EVAL_MODE", "geno_cv")
WITHIN_GENO_TEST_FRAC = 0.3
WITHIN_GENO_VAL_FRAC = 0.3
WITHIN_GENO_MIN_TEST = 1
WITHIN_GENO_MIN_TRAIN = 1
WITHIN_GENO_SEED_OFFSET = 100
CV_FOLDS = int(os.environ.get("ECOPOP_CV_FOLDS", "4"))
SAVE_FOLD_PREDICTIONS = True  # save per-fold GEBVs when running CV
EXPORT_EMBEDDINGS = True     # if True, export penultimate embeddings on test set
RUN_EMBED_PLOTS = True      # if True, run t-SNE plots on exported embeddings
EMBEDDING_VIEWS = ("fused", "genomic", "pop", "loc", "year")  # which embeddings to plot
TSNE_EMBEDDING_VIEWS = ("genomic", "pop", "loc", "year")  # which embeddings to t-SNE
BOXPLOT_TRIM = False        # if True, trim trait values outside boxplot whiskers
BOXPLOT_WHISKER_K = 5.0     # whisker multiplier (1.5=default Tukey)
BOXPLOT_SAVE_PATH = "trait_boxplot.png"    # set to a path (e.g., "trait_boxplot.png") to save boxplot
# Environment feature engineering
USE_ENV_PCA = False
ENV_PCA_COMPONENTS = 10
USE_ENV_MATRIX_AS_MLP = False  # use baseline wide env matrix via MLP instead of temporal encoder
TRAIT_MODE = "complex"  # "complex" = ChromoMap tensors, "simple" = raw genotype vectors
GENO_SOURCE = "plink"  # "plink" or "vcf"
PLINK_PREFIX = os.environ.get("ECOPOP_PLINK_PREFIX", "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean")
VCF_PATH = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean.map.vcf"
SIMPLE_GENO_PCA = True
SIMPLE_GENO_N_PCS = 300
SIMPLE_GENO_STANDARDIZE = True

# Temporal environmental encoding
USE_TEMPORAL_ENV_ENCODING = True
USE_ENV_RESIDUAL_TRAINING = False  # train gxetensor on env-only residuals (y - y_env)
N_MONTHS = int(os.environ.get("ECOPOP_N_MONTHS", "32"))
ENV_ENGINEERED_FEATURES = True
ENV_STAGE_SUMMARIES = True
ENV_HIDDEN_DIM = 32
ENV_LSTM_LAYERS = 2
ENV_ENCODER_TYPE = "lstm"  # "lstm", "tcn", or "pyramid"
ENV_CONV_CHANNELS = 64
ENV_CONV_LAYERS = 2
ENV_CONV_KERNEL = 3
ENV_PYRAMID_SCALES = [1, 2, 4]
ENV_PYRAMID_LAYERS = 2
ENV_PERTURB_SIGMA = 0.01  # Gaussian noise scale (fraction of per-feature std) applied during training
# Cache preprocessed temporal env tensors to skip recomputation on every run
USE_ENV_CACHE = False
ENV_CACHE_FILE = "env_temporal_cache.npz"
# When True, fail fast if any sample is missing temporal weather data instead of silently
# feeding zeros (which makes the model ignore environment signals).
ENV_LOOKUP_FAIL_ON_MISSING = os.environ.get("ECOPOP_ENV_FAIL_MISSING", "0") == "1"

# Location and year
N_LOCATIONS = 2
N_YEARS = 3
LOCATION_EMBED_DIM = 4
YEAR_EMBED_DIM = 4

# Population structure
N_POPULATIONS = 50
POP_EMBED_DIM = 51
POP_EMBED_WEIGHT_DECAY = 1.0
METADATA_WEIGHT_DECAY = 1.0 # stronger L2 for loc/year + metadata FiLM modules
INTERACTION_REG_LAMBDA = 0.05
LRBI_RANK = 16  # low-rank bilinear interaction rank (0 disables LRBI)
USE_GXE_MOE = False  # population-specific mixture of experts for GxE interaction head
GXE_MOE_NUM_EXPERTS = 3
GXE_MOE_HIDDEN_DIM = 32
GXE_MOE_TEMPERATURE = 1.0
USE_ENV_FILM = True  # FiLM-style env modulation inside the hierarchical genomic encoder
USE_ENV_POOL_BIAS = True  # Env-conditioned attention bias during pooling for gxetensor
USE_META_FILM = False  # FiLM-style metadata modulation (loc/year/pop) on genomic tokens for gxetensor
META_FILM_SCALE = 0.1  # keep metadata FiLM subtle to avoid overfitting sparse groups
USE_BIOLOGICAL_AWARE_EMBEDDING = os.environ.get("ECOPOP_USE_BAE", "1") == "1"
USE_DOSAGE_ANNOT_CHANNELS = False # derive dosage*annotation channels at load time
# Opt-in functional-SNP prioritization for the ADDITIVE/dosage branch (see diploid engine).
_ECOPOP_GENIC = os.environ.get("ECOPOP_ADDITIVE_GENIC_IDS", "").strip()
_GENIC_SNP_IDS: Set[str] = set()
if _ECOPOP_GENIC:
    try:
        _GENIC_SNP_IDS = {ln.strip() for ln in open(_ECOPOP_GENIC) if ln.strip()}
        print(f"[GENIC] additive-branch functional-SNP prioritization: {len(_GENIC_SNP_IDS)} genic SNP IDs loaded.", flush=True)
    except Exception as _e:
        print(f"[GENIC] failed to load ECOPOP_ADDITIVE_GENIC_IDS={_ECOPOP_GENIC} ({_e}); no filter applied.", flush=True)
MAX_SPARSE_TOKENS = 256  # cap HABE sparse SNP tokens to avoid quadratic attention blowups
# Reweight loss toward under-represented environments/populations
USE_ENV_WEIGHTED_LOSS = False
POP_LOSS_BOOST = 0.0
# Number of distinct (location, year) pairs; set after env data is loaded.
NUM_ENVIRONMENTS = 3
ENV_PAIR_TO_ID: Dict[Tuple[int, int], int] = {}
ENV_PAIR_LUT: Optional[np.ndarray] = None
# Runtime cache for optional PLINK-driven dosage branch override in tensor/complex mode.
DOSAGE_OVERRIDE_ENABLED = False
DOSAGE_OVERRIDE_MAP: Dict[str, np.ndarray] = {}
DOSAGE_OVERRIDE_DIM = 0
DOSAGE_OVERRIDE_SAMPLE_TO_SID: Dict[str, str] = {}
_DOSAGE_OVERRIDE_MISSING_WARNED = False
# Adversarial environment invariance (GRL on genomic branch)
USE_ENV_ADVERSARY = False
ENV_ADVERSARY_WEIGHT = 0.05
ENV_ADVERSARY_WARMUP_EPOCHS = 40
ENV_ADVERSARY_MAX_ALPHA = 1.0

# Critical features
CRITICAL_ENV_FEATURES = ['daylength_h', 'tmax_C', 'tmin_C', 'precip_mm', 'gdd', 'vpd_kPa', 'srad_allsky']
ENV_DERIVED_FEATURES = [
    "photo_temp",
    "cum_gdd",
    "cum_ptu",
    "heat_hdd",
    "cold_cdd",
    "drought_vpd",
]
ENV_STAGE_METRICS = [
    "gdd_sum",
    "heat_hdd_sum",
    "vpd_mean",
]
N_ENV_FEATURES_PER_MONTH = (
    len(CRITICAL_ENV_FEATURES)
    + (len(ENV_DERIVED_FEATURES) if ENV_ENGINEERED_FEATURES else 0)
    + (len(ENV_STAGE_METRICS) * 3 if (ENV_ENGINEERED_FEATURES and ENV_STAGE_SUMMARIES) else 0)
)

# Model (gxetensor only)

# Combined hierarchical tensors (Chromomap 2D layout; single NPZ per sample)
USE_GENOMIC_TENSORS = True
TENSOR_DIR = os.environ.get("ECOPOP_TENSOR_DIR", "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/Chromomap/images_AF_combined_d3_allele_comb_xyz_te_new_snp_rep_/tensors")
if not USE_GENOMIC_TENSORS:
    raise RuntimeError("Tile/PNG loaders have been removed. Set USE_GENOMIC_TENSORS=True and provide tensor NPZs.")

# Optional TE sub-type channels (e.g., te_gypsy, te_copia) if present in NPZ
USE_TE_SUBTYPE_FEATURES = False

# ---- Opt-in 3D-genome channels (Project 1): append Hi-C compartment/insulation/boundary ----
# ECOPOP_3D_CHANNELS=/path/to/[C,T,K].npy (sample-independent, aligned to the tensor layout).
# When unset, USE_3D_CHANNELS is False and behavior is byte-identical (no channels appended).
_ECOPOP_3D_PATH = os.environ.get("ECOPOP_3D_CHANNELS", "").strip()
USE_3D_CHANNELS = bool(_ECOPOP_3D_PATH)
_3D_CHANNELS = None
_3D_K = 0
if USE_3D_CHANNELS:
    try:
        _3D_CHANNELS = np.load(_ECOPOP_3D_PATH).astype(np.float32)
        _3D_K = int(_3D_CHANNELS.shape[-1])
        print(f"[3D] loaded {_ECOPOP_3D_PATH} -> {_3D_CHANNELS.shape} ; appending {_3D_K} channels to the genomic tensor.", flush=True)
    except Exception as _e:
        print(f"[3D] FAILED to load ECOPOP_3D_CHANNELS={_ECOPOP_3D_PATH} ({_e}); 3D channels disabled.", flush=True)
        USE_3D_CHANNELS = False; _3D_CHANNELS = None; _3D_K = 0
print(f"[3D] status at import: USE_3D_CHANNELS={USE_3D_CHANNELS} K={_3D_K}", flush=True)
# Multi-environment flag (True = multiple rows per SampleID; False = single env row per SampleID)
MULTI_ENV = True
# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_chromomap.log")]
)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay (epoch-based)."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.max_epochs = max(1, int(max_epochs))
        self.warmup_epochs = max(0, min(int(warmup_epochs), self.max_epochs))
        self.min_lr_ratio = max(0.0, float(min_lr_ratio))
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch_idx: int):
        if epoch_idx < 0:
            epoch_idx = 0
        if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
            warmup_progress = (epoch_idx + 1) / float(self.warmup_epochs)
            lrs = [base * warmup_progress for base in self.base_lrs]
        else:
            cosine_epochs = max(1, self.max_epochs - self.warmup_epochs)
            t = min(max(0, epoch_idx - self.warmup_epochs), cosine_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * t / float(cosine_epochs)))
            lrs = []
            for base in self.base_lrs:
                min_lr = base * self.min_lr_ratio
                lr = min_lr + (base - min_lr) * cosine
                lrs.append(lr)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        return lrs

class QuantileLoss(nn.Module):
    def __init__(self, quantile: float = 0.5, reduction: str = "mean"):
        super().__init__()
        assert 0.0 < quantile < 1.0, "Quantile must be between 0 and 1"
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        errors = targets - preds
        loss = torch.max(self.quantile * errors, (self.quantile - 1.0) * errors)
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class AdaptiveLossFunction(nn.Module):
    """
    Adaptive loss that softly switches between MSE (simple traits) and Huber (robust)
    based on a learned gate.
    """
    def __init__(self, initial_delta: float = 1.0):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(initial_delta, dtype=torch.float32))
        self.use_huber = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # 0=MSE, 1=Huber

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(pred, target, reduction="none")
        huber_loss = F.smooth_l1_loss(pred, target, beta=self.delta.clamp(0.1, 5.0), reduction="none")
        alpha = torch.sigmoid(self.use_huber)
        mixed = (1 - alpha) * mse_loss + alpha * huber_loss
        return mixed.mean()


def build_loss_fn(name: str, reduction: Optional[str] = None) -> nn.Module:
    name = (name or "mse").lower()
    if reduction is None:
        reduction = "none" if (USE_ENV_WEIGHTED_LOSS or POP_LOSS_BOOST > 0) else "mean"
    if name == "mse":
        return nn.MSELoss(reduction=reduction)
    if name == "huber":
        return nn.SmoothL1Loss(beta=HUBER_DELTA, reduction=reduction)
    if name == "quantile":
        return QuantileLoss(quantile=QUANTILE_ALPHA, reduction=reduction)
    if name == "adaptive":
        return AdaptiveLossFunction(initial_delta=HUBER_DELTA)
    raise ValueError(f"Unsupported loss '{name}'")


def create_snapshot_scheduler(optimizer, max_epochs: int, cycles: int):
    """Returns a cosine annealing scheduler that restarts every cycle (for snapshot ensembling)."""
    cycles = max(1, int(cycles))
    cycle_length = max(1, max_epochs // cycles)
    return CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=1)


def build_scheduler(optimizer, max_epochs: int):
    """
    Select scheduler:
      - Snapshot ensembling: cosine restarts
      - Else: warmup + cosine anneal to LR_MIN_LR_FACTOR
    """
    if USE_SNAPSHOT_ENSEMBLE:
        return create_snapshot_scheduler(optimizer, max_epochs, SNAPSHOT_CYCLES)
    return WarmupCosineScheduler(
        optimizer,
        warmup_epochs=LR_WARMUP_EPOCHS,
        max_epochs=max_epochs,
        min_lr_ratio=LR_MIN_LR_FACTOR,
    )


def step_scheduler(scheduler, epoch_idx: int):
    """
    Consistent scheduler stepping (epoch_idx is zero-based).
    """
    if scheduler is None:
        return
    if isinstance(scheduler, WarmupCosineScheduler):
        scheduler.step(epoch_idx)
    else:
        scheduler.step()


def create_gxe_optimizer(
    model,
    lr: float,
    weight_decay: float,
    pop_weight_decay: float,
    metadata_weight_decay: Optional[float] = None
):
    """
    Builds an optimizer that optionally applies stronger L2 penalty to metadata modules.
    """
    if metadata_weight_decay is None:
        metadata_weight_decay = weight_decay
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    pop_params = [p for n, p in named_params if "pop_embedding" in n]
    meta_keys = ("location_embedding", "year_embedding", "meta_to_gamma", "meta_to_beta")
    meta_params = [p for n, p in named_params if any(k in n for k in meta_keys)]
    pop_param_ids = {id(p) for p in pop_params}
    meta_param_ids = {id(p) for p in meta_params if id(p) not in pop_param_ids}
    other_params = [p for _, p in named_params if id(p) not in pop_param_ids and id(p) not in meta_param_ids]
    param_groups = [{"params": other_params, "weight_decay": weight_decay}]
    if meta_param_ids:
        meta_group = [p for p in meta_params if id(p) in meta_param_ids]
        param_groups.append({"params": meta_group, "weight_decay": metadata_weight_decay})
    if pop_param_ids:
        param_groups.append({"params": pop_params, "weight_decay": pop_weight_decay})
    return optim.AdamW(param_groups, lr=lr)


def _build_env_feature_names() -> List[str]:
    names = list(CRITICAL_ENV_FEATURES)
    if ENV_ENGINEERED_FEATURES:
        names.extend(ENV_DERIVED_FEATURES)
        if ENV_STAGE_SUMMARIES:
            stage_names = ["early", "mid", "late"]
            for stage in stage_names:
                for metric in ENV_STAGE_METRICS:
                    names.append(f"{stage}_{metric}")
    return names


def _env_feature_signature() -> str:
    heat_q = 0.90
    cold_q = 0.10
    vpd_q = 0.90
    payload = {
        "critical_features": CRITICAL_ENV_FEATURES,
        "engineered": ENV_ENGINEERED_FEATURES,
        "stage_summaries": ENV_STAGE_SUMMARIES,
        "derived": ENV_DERIVED_FEATURES if ENV_ENGINEERED_FEATURES else [],
        "stage_metrics": ENV_STAGE_METRICS if ENV_STAGE_SUMMARIES else [],
        "stage_names": ["early", "mid", "late"] if ENV_STAGE_SUMMARIES else [],
        "n_steps": N_MONTHS,
        "threshold_method": "quantile",
        "heat_quantile": heat_q,
        "cold_quantile": cold_q,
        "vpd_quantile": vpd_q,
    }
    return json.dumps(payload, sort_keys=True)

def load_checkpoint_safely(model: nn.Module, path: str, device: torch.device, allow_shape_mismatch: bool = True) -> bool:
    """
    Loads a checkpoint while gracefully skipping incompatible tensors (e.g., when
    an older checkpoint used different embed dims). Returns True if any params
    were loaded.
    """
    if not os.path.exists(path):
        logging.warning(f"Checkpoint not found: {path}")
        return False
    try:
        state = torch.load(path, map_location=device)
    except Exception as e:
        logging.warning(f"Failed to read checkpoint {path}: {e}")
        return False

    model_state = model.state_dict()
    if allow_shape_mismatch:
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if not filtered:
            logging.warning(f"Skipped loading {path}: no compatible parameters found (possible embed dim change).")
            return False
        result = model.load_state_dict(filtered, strict=False)
        if skipped:
            logging.warning(f"Skipped {len(skipped)} incompatible keys from {path} (shape mismatch or missing in model).")
        if result.missing_keys or result.unexpected_keys:
            logging.info(f"Loaded checkpoint with missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)} keys.")
        else:
            logging.info(f"Loaded checkpoint from {path} (compatible params only).")
        return True

    # Strict shape matching: if any mismatch, skip entirely
    mismatches = [
        k for k, v in state.items()
        if (k not in model_state) or (model_state[k].shape != v.shape)
    ]
    if mismatches:
        logging.warning(f"Strict load skipped for {path}: {len(mismatches)} keys have incompatible shapes.")
        return False
    model.load_state_dict(state, strict=True)
    logging.info(f"Loaded checkpoint from {path} (strict).")
    return True

def get_regularization_for_model(_: str = "gxetensor") -> Tuple[float, float]:
    """
    Returns (dropout, weight_decay) for the gxetensor model.
    """
    return GXE_DROPOUT, MAIN_WEIGHT_DECAY


# Target scaling helpers
# ---------------------------

TARGET_SCALER = {"mean": None, "std": None}

def set_target_scaler(mean: float, std: float):
    """
    Store active target scaler (mean/std). Std is clamped to avoid divide-by-zero.
    """
    if mean is None or std is None:
        TARGET_SCALER["mean"] = None
        TARGET_SCALER["std"] = None
        return
    std_safe = std if std > 1e-8 else 1.0
    TARGET_SCALER["mean"] = float(mean)
    TARGET_SCALER["std"] = float(std_safe)


def destandardize_targets(arr: np.ndarray) -> np.ndarray:
    if TARGET_SCALER["mean"] is None or TARGET_SCALER["std"] is None:
        return arr
    return arr * TARGET_SCALER["std"] + TARGET_SCALER["mean"]


def _filter_finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        logging.warning(f"Dropping {np.size(mask) - mask.sum()} samples due to non-finite predictions/targets.")
    return y_true[mask], y_pred[mask]

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance correlation coefficient (CCC): more robust than R2 on small samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))
    cov = float(np.cov(y_true, y_pred, ddof=0)[0, 1])  # ddof=0 to match np.var (population) — CCC denominator consistency
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * cov / denom)


def _r2_from_lists(y_true_list: List[float], y_pred_list: List[float]) -> Optional[float]:
    if len(y_pred_list) < 2:
        return None
    y_true = np.asarray(y_true_list, dtype=float)
    y_pred = np.asarray(y_pred_list, dtype=float)
    y_true, y_pred = _filter_finite_pairs(y_true, y_pred)
    if len(y_pred) < 2:
        return None
    return r2_score(y_true, y_pred)


def residual_variance_ceiling(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Approximate trait ceiling from predictions:
      ceiling ~= 1 - Var(residual) / Var(y_true)
    where residual = y_true - y_pred.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true, y_pred = _filter_finite_pairs(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    var_y = float(np.var(y_true))
    if var_y <= 1e-12:
        return float("nan")
    residual = y_true - y_pred
    var_res = float(np.var(residual))
    return float(1.0 - (var_res / var_y))


def _compute_target_scaler(meta_df: pd.DataFrame, sample_keys: List[str], key_col: str) -> Tuple[float, float]:
    """
    Compute mean/std of target column over the provided sample keys (train split).
    """
    if key_col not in meta_df.columns:
        return 0.0, 1.0
    mask = meta_df[key_col].astype(str).isin(sample_keys)
    series = pd.to_numeric(meta_df.loc[mask, TARGET_COL], errors="coerce").dropna()
    if series.empty:
        return 0.0, 1.0
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    std = std if std > 1e-8 else 1.0
    return mean, std


def _build_aux_target_lookup(
    aux_source: Optional[pd.DataFrame],
    train_keys: List[str],
    all_keys: List[str],
    key_col: str,
    aux_cols: List[str],
) -> Dict[str, List[float]]:
    """
    Build {sample_key -> [K standardized aux-trait values]} for opt-in multi-trait MTL.
    Per-trait mean/std are computed on TRAIN keys ONLY (per fold / per split), mirroring
    _compute_target_scaler (ddof=0, std floor 1e-8), so no aux-trait test info leaks.
    Missing trait values are stored as NaN and masked out of the aux loss.
    """
    if aux_source is None or not aux_cols:
        return {}
    src = aux_source.drop_duplicates(subset=[key_col], keep="first")
    key_str = src[key_col].astype(str)
    col_maps: Dict[str, Dict[str, float]] = {}
    for col in aux_cols:
        if col in src.columns:
            vals = pd.to_numeric(src[col], errors="coerce")
        else:
            vals = pd.Series([np.nan] * len(src), index=src.index)
        col_maps[col] = dict(zip(key_str, vals))
    train_set = {str(k) for k in train_keys}
    stats: List[Tuple[float, float]] = []
    for col in aux_cols:
        cmap = col_maps[col]
        tvals = [float(v) for k, v in cmap.items() if k in train_set and pd.notna(v)]
        if not tvals:
            stats.append((0.0, 1.0))
        else:
            arr = np.asarray(tvals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            stats.append((mean, std if std > 1e-8 else 1.0))
    lookup: Dict[str, List[float]] = {}
    for key in dict.fromkeys(str(k) for k in all_keys):
        vec: List[float] = []
        for j, col in enumerate(aux_cols):
            raw = col_maps[col].get(key, np.nan)
            if raw is None or pd.isna(raw):
                vec.append(float("nan"))
            else:
                mean, std = stats[j]
                vec.append((float(raw) - mean) / std)
        lookup[key] = vec
    return lookup


def _build_env_target_stats(
    meta_df: pd.DataFrame,
    sample_keys: List[str],
    key_col: str,
    target_lookup: Optional[Dict[str, float]] = None,
    env_data_dict: Optional[Dict[str, np.ndarray]] = None,
    sample_key_to_sid: Optional[Dict[str, str]] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Compute per-environment (Location x Year) mean/std for targets on train keys.
    """
    key_set = {str(k) for k in sample_keys}
    env_vals: Dict[int, List[float]] = defaultdict(list)
    if env_data_dict is not None and sample_key_to_sid is not None:
        metadata_indexed = meta_df.set_index(key_col)
        idx_map, _ = _build_env_index_map(
            sample_ids=sample_keys,
            sample_key_to_sid=sample_key_to_sid,
            metadata_indexed=metadata_indexed,
            env_data_dict=env_data_dict
        )
        loc_ids = env_data_dict.get("location_ids")
        year_ids = env_data_dict.get("year_ids")
        env_pair_ids = env_data_dict.get("env_pair_ids")
        for key in sample_keys:
            key_str = str(key)
            if key_str not in key_set:
                continue
            env_idx = idx_map.get(key_str)
            if env_idx is None:
                continue
            if env_pair_ids is not None and len(env_pair_ids) > env_idx:
                env_id = int(env_pair_ids[env_idx])
            else:
                loc_id = int(loc_ids[env_idx]) if loc_ids is not None and len(loc_ids) > env_idx else 0
                year_id = int(year_ids[env_idx]) if year_ids is not None and len(year_ids) > env_idx else 0
                env_id = _resolve_env_id(loc_id, year_id)
            if target_lookup is not None:
                val = target_lookup.get(key_str)
            else:
                try:
                    val = meta_df.loc[meta_df[key_col].astype(str) == key_str, TARGET_COL].iloc[0]
                except Exception:
                    val = np.nan
            if pd.isna(val):
                continue
            env_vals[env_id].append(float(val))
    else:
        if "Location_Code" not in meta_df.columns or "Year_Code" not in meta_df.columns:
            return {}
        for _, row in meta_df.iterrows():
            key = str(row.get(key_col, ""))
            if key not in key_set:
                continue
            loc = row.get("Location_Code", None)
            yr = row.get("Year_Code", None)
            if pd.isna(loc) or pd.isna(yr):
                continue
            try:
                loc_id = int(loc)
                year_id = int(yr)
            except Exception:
                continue
            env_id = _resolve_env_id(loc_id, year_id)
            if target_lookup is not None:
                val = target_lookup.get(key)
            else:
                val = row.get(TARGET_COL, np.nan)
            if pd.isna(val):
                continue
            env_vals[env_id].append(float(val))
    env_stats: Dict[int, Tuple[float, float]] = {}
    for env_id, vals in env_vals.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        if std <= 1e-8:
            std = 1.0
        env_stats[int(env_id)] = (mean, std)
    return env_stats


def _apply_env_unscale(
    values: np.ndarray,
    env_ids: List[int],
    env_target_stats: Dict[int, Tuple[float, float]]
) -> np.ndarray:
    if not env_target_stats:
        return values
    if len(values) != len(env_ids):
        return values
    out = values.copy()
    for i, env_id in enumerate(env_ids):
        mean, std = env_target_stats.get(int(env_id), (0.0, 1.0))
        out[i] = out[i] * std + mean
    return out


def build_env_data_dict_fold(env: pd.DataFrame, tr_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Build temporal environment tensors with normalization fit on a fold's train IDs only.
    """
    env_keys = env["EnvKey"].astype(str).tolist()
    train_keys = [str(k) for k in tr_ids]
    has_split_keys = any("|" in k for k in train_keys)
    split_set = set(train_keys) if has_split_keys else set()
    sample_set = {k.split("|", 1)[0] for k in train_keys}
    sample_col = None
    for cand in ("SampleID", "SampleID_str"):
        if cand in env.columns:
            sample_col = cand
            break
    if sample_col is not None:
        env_sample_ids = env[sample_col].astype(str).tolist()
    else:
        env_sample_ids = [k.split("|", 1)[0] for k in env_keys]
    fit_mask = np.array(
        [
            (k in split_set) or (sid in sample_set)
            for k, sid in zip(env_keys, env_sample_ids)
        ],
        dtype=bool
    )
    env_wide_cols = [c for c in env.columns if c.startswith("E_")]
    env_wide = None
    env_wide_mean = None
    env_wide_std = None
    if env_wide_cols:
        env_wide = env[env_wide_cols].to_numpy(dtype=float)
        if env_wide.shape[0] == fit_mask.shape[0]:
            fit_slice = env_wide[fit_mask] if fit_mask.any() else env_wide
        else:
            fit_slice = env_wide
        env_wide_mean = fit_slice.mean(axis=0, keepdims=True)
        env_wide_std = fit_slice.std(axis=0, keepdims=True) + 1e-8
        env_wide = (env_wide - env_wide_mean) / env_wide_std

    (
        env_temporal,
        loc_ids,
        yr_ids,
        loc_map,
        yr_map,
        feat_names,
        env_stats,
    ) = preprocess_environmental_data(
        env,
        critical_features=CRITICAL_ENV_FEATURES,
        strict=ENV_LOOKUP_FAIL_ON_MISSING,
        n_steps=N_MONTHS,
        engineer_features=ENV_ENGINEERED_FEATURES,
        use_stage_summaries=ENV_STAGE_SUMMARIES,
        return_feature_names=True,
        fit_mask=fit_mask,
        return_stats=True,
    )

    return {
        "temporal": env_temporal,
        "location_ids": loc_ids,
        "year_ids": yr_ids,
        "key_to_idx": {str(k): i for i, k in enumerate(env_keys)},
        "env_mean": env_stats[0],
        "env_std": env_stats[1],
        "feature_names": feat_names,
        "env_wide": env_wide,
        "env_wide_mean": env_wide_mean,
        "env_wide_std": env_wide_std,
        "env_wide_cols": env_wide_cols,
    }


def fit_env_main_effects(
    meta_df: pd.DataFrame,
    train_keys: List[str],
    all_keys: List[str],
    key_col: str,
    env_data_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Fit an env-only ridge model on train keys using temporal env vectors.
    Returns predictions for all_keys (missing keys default to 0).
    """
    if env_data_dict is None or "temporal" not in env_data_dict or "key_to_idx" not in env_data_dict:
        return {}
    key_to_idx = env_data_dict.get("key_to_idx", {})
    temporal = env_data_dict.get("temporal", None)
    if temporal is None:
        return {}
    # Build train matrices
    X_train, y_train = [], []
    for k in train_keys:
        idx = key_to_idx.get(str(k))
        if idx is None or idx >= len(temporal):
            continue
        x = temporal[idx].reshape(-1)  # flatten months Ãƒâ€” features
        y_val = pd.to_numeric(meta_df.loc[meta_df[key_col].astype(str) == str(k), TARGET_COL], errors="coerce")
        if y_val.empty:
            continue
        y = float(y_val.iloc[0])
        X_train.append(x)
        y_train.append(y)
    if not X_train:
        logging.warning("Env-only model skipped: no train keys with temporal env data.")
        return {}
    X_train = np.stack(X_train, axis=0)
    y_train = np.array(y_train, dtype=float)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = {}
    for k in all_keys:
        idx = key_to_idx.get(str(k))
        if idx is None or idx >= len(temporal):
            preds[k] = 0.0
            continue
        x = temporal[idx].reshape(1, -1)
        preds[k] = float(model.predict(x)[0])
    logging.info(f"Fitted env-only ridge model on {len(X_train)} samples; generated {len(preds)} predictions.")
    return preds


def log_population_r2(predictions: List[Dict[str, float]], sample_to_pop: Dict[str, int], context: str = ""):
    if not sample_to_pop or not predictions:
        return
    groups: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"pred": [], "true": []})
    for row in predictions:
        pop = sample_to_pop.get(row["SampleID"])
        if pop is None:
            continue
        groups[pop]["pred"].append(row["pred"])
        groups[pop]["true"].append(row["true"])
    if not groups:
        return
    entries = []
    for pop, rec in sorted(groups.items()):
        if len(rec["pred"]) < 2:
            continue
        entries.append(f"pop_{pop}: R2={r2_score(rec['true'], rec['pred']):.4f} (n={len(rec['pred'])})")
    if entries:
        logging.info(f"{context}Cross-population RÂ² -> " + " | ".join(entries))


class GxE_FusionHead(nn.Module):
    """
    Shared fusion + interaction head used by both tensor and raw-genotype models.
    """
    def __init__(
        self,
        embed_dim: int,
        env_embed_dim: int,
        location_embed_dim: int,
        year_embed_dim: int,
        pop_embed_dim: int,
        interaction_dim: int = 64,
        dropout: float = GXE_DROPOUT,
        main_head_dropout: Optional[float] = None,
        interaction_head_dropout: Optional[float] = None,
        low_rank_bilinear_rank: int = 0,
        interaction_reg_lambda: float = 0.0,
        use_gxe_moe: bool = False,
        gxe_moe_num_experts: int = 4,
        gxe_moe_hidden_dim: Optional[int] = None,
        gxe_moe_temperature: float = 1.0,
    ):
        super().__init__()
        self.interaction_dim = int(interaction_dim)
        self.interaction_reg_lambda = float(interaction_reg_lambda)
        self.low_rank_bilinear_rank = int(low_rank_bilinear_rank)
        self.use_gxe_moe = bool(use_gxe_moe)
        self.gxe_moe_temperature = float(gxe_moe_temperature)

        fused_dim = embed_dim + env_embed_dim + location_embed_dim + year_embed_dim + pop_embed_dim
        head_drop = dropout if main_head_dropout is None else main_head_dropout
        self.fuse_dropout = nn.Dropout(head_drop)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(fused_dim, 1)
        )

        if self.low_rank_bilinear_rank > 0:
            self.interaction_bilinear = LowRankBilinear(
                embed_dim, env_embed_dim, self.interaction_dim, rank=self.low_rank_bilinear_rank
            )
        else:
            self.interaction_bilinear = nn.Bilinear(embed_dim, env_embed_dim, self.interaction_dim)
        self.g_proj = nn.Linear(embed_dim, self.interaction_dim)
        self.e_proj = nn.Linear(env_embed_dim, self.interaction_dim)
        inter_hidden = max(64, self.interaction_dim if gxe_moe_hidden_dim is None else gxe_moe_hidden_dim)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, inter_hidden),
            nn.LayerNorm(inter_hidden),
            nn.GELU(),
            nn.Dropout(dropout if interaction_head_dropout is None else interaction_head_dropout),
            nn.Linear(inter_hidden, 1)
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(RESIDUAL_GATE_INIT)))

        if self.use_gxe_moe:
            self.gxe_gate = nn.Sequential(
                nn.Linear(pop_embed_dim, max(4, pop_embed_dim)),
                nn.GELU(),
                nn.Linear(max(4, pop_embed_dim), gxe_moe_num_experts)
            )
            self.gxe_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, inter_hidden),
                        nn.LayerNorm(inter_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout if interaction_head_dropout is None else interaction_head_dropout),
                        nn.Linear(inter_hidden, 1)
                    )
                    for _ in range(gxe_moe_num_experts)
                ]
            )
        else:
            self.gxe_gate = None
            self.gxe_experts = None

    def interaction_reg_penalty(self) -> Optional[torch.Tensor]:
        if self.interaction_reg_lambda <= 0.0:
            return None
        if hasattr(self.interaction_bilinear, "regularization"):
            return self.interaction_reg_lambda * self.interaction_bilinear.regularization()
        weight = getattr(self.interaction_bilinear, "weight", None)
        if weight is None:
            return None
        return self.interaction_reg_lambda * weight.abs().mean()

    def compute_ge_feat(self, g_repr: torch.Tensor, env_repr: torch.Tensor) -> torch.Tensor:
        return self.interaction_bilinear(g_repr, env_repr) + (self.g_proj(g_repr) * self.e_proj(env_repr))

    def forward(
        self,
        g_repr: torch.Tensor,
        env_repr_main: torch.Tensor,
        loc_emb: torch.Tensor,
        year_emb: torch.Tensor,
        pop_emb: torch.Tensor,
        env_repr_gxe: Optional[torch.Tensor] = None,
        stage: int = 0,
        return_components: bool = False
    ):
        env_int = env_repr_gxe if env_repr_gxe is not None else env_repr_main
        fused = torch.cat([g_repr, env_repr_main, loc_emb, year_emb, pop_emb], dim=-1)
        fused = self.fuse_dropout(fused)
        main_out = self.head(fused).squeeze(-1)

        ge_feat = self.compute_ge_feat(g_repr, env_int)
        inter_input = torch.cat([ge_feat, loc_emb, year_emb, pop_emb], dim=-1)
        if self.use_gxe_moe and self.gxe_gate is not None and self.gxe_experts is not None:
            gate_logits = self.gxe_gate(pop_emb)
            if self.gxe_moe_temperature != 1.0:
                gate_logits = gate_logits / max(1e-6, self.gxe_moe_temperature)
            gate = torch.softmax(gate_logits, dim=-1)
            expert_outs = torch.stack(
                [expert(inter_input).squeeze(-1) for expert in self.gxe_experts],
                dim=-1
            )
            gxe_out = (expert_outs * gate).sum(dim=-1)
        else:
            gxe_out = self.interaction_head(inter_input).squeeze(-1)

        gate = torch.clamp(self.residual_gate, min=0.0)
        gxe_out_scaled = gxe_out * gate

        if stage == 1:
            out = main_out
        elif stage == 2:
            out = gxe_out_scaled
        else:
            out = main_out + gxe_out_scaled

        aux: Dict[str, torch.Tensor] = {}
        penalty = self.interaction_reg_penalty()
        if penalty is not None:
            aux["interaction_reg"] = penalty
        aux["main_out"] = main_out
        aux["gxe_out"] = gxe_out_scaled
        aux["gxe_out_raw"] = gxe_out
        aux["residual_gate"] = gate.detach()

        if return_components:
            return out, aux
        if aux:
            return out, aux
        return out


class GxE_RawGenotypeModel(nn.Module):
    """
    Lightweight GxE model for raw genotype vectors (no ChromoMap tensors).
    Reuses env encoder + interaction head structure from the tensor model.
    """
    def __init__(
        self,
        geno_input_dim: int,
        embed_dim: int = EMBED_DIM,
        env_embed_dim: int = 32,
        env_encoder_type: str = "mlp",
        env_hidden_dim: int = ENV_HIDDEN_DIM,
        env_lstm_layers: int = ENV_LSTM_LAYERS,
        env_conv_channels: int = ENV_CONV_CHANNELS,
        env_conv_layers: int = ENV_CONV_LAYERS,
        env_conv_kernel: int = ENV_CONV_KERNEL,
        env_pyramid_scales: Optional[List[int]] = None,
        env_pyramid_layers: Optional[int] = None,
        n_env_features_per_month: int = N_ENV_FEATURES_PER_MONTH,
        n_months: int = N_MONTHS,
        n_locations: int = N_LOCATIONS,
        n_years: int = N_YEARS,
        location_embed_dim: int = LOCATION_EMBED_DIM,
        year_embed_dim: int = YEAR_EMBED_DIM,
        n_populations: int = N_POPULATIONS,
        pop_embed_dim: int = POP_EMBED_DIM,
        dropout: float = GXE_DROPOUT,
        main_head_dropout: Optional[float] = None,
        interaction_head_dropout: Optional[float] = None,
        interaction_dim: int = 64,
        low_rank_bilinear_rank: int = 0,
        interaction_reg_lambda: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.env_embed_dim = env_embed_dim
        self.interaction_dim = interaction_dim
        self.interaction_reg_lambda = float(interaction_reg_lambda)
        self.low_rank_bilinear_rank = int(low_rank_bilinear_rank)

        self.geno_proj = nn.Sequential(
            nn.Linear(geno_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        enc_type = str(env_encoder_type).lower()
        if enc_type == "lstm":
            self.env_encoder = TemporalEnvironmentalEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                hidden_dim=env_hidden_dim,
                num_layers=env_lstm_layers,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6
            )
        elif enc_type == "tcn":
            self.env_encoder = TemporalConvEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                conv_channels=env_conv_channels,
                num_layers=env_conv_layers,
                kernel_size=env_conv_kernel,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6
            )
        elif enc_type == "pyramid":
            pyramid_layers = env_conv_layers if env_pyramid_layers is None else int(env_pyramid_layers)
            self.env_encoder = TemporalPyramidEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                conv_channels=env_conv_channels,
                num_layers=pyramid_layers,
                kernel_size=env_conv_kernel,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6,
                scales=env_pyramid_scales
            )
        else:
            self.env_encoder = WideEnvMLPEncoder(
                n_features_per_month=n_env_features_per_month,
                env_embed_dim=env_embed_dim,
                hidden_dim=max(64, env_embed_dim * 4),
                dropout=dropout * 0.6
            )

        self.location_embedding = nn.Embedding(n_locations, location_embed_dim)
        self.year_embedding = nn.Embedding(n_years, year_embed_dim)
        self.pop_embedding = nn.Embedding(n_populations, pop_embed_dim)

        self.fusion = GxE_FusionHead(
            embed_dim=embed_dim,
            env_embed_dim=env_embed_dim,
            location_embed_dim=location_embed_dim,
            year_embed_dim=year_embed_dim,
            pop_embed_dim=pop_embed_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
            main_head_dropout=main_head_dropout,
            interaction_head_dropout=interaction_head_dropout,
            low_rank_bilinear_rank=low_rank_bilinear_rank,
            interaction_reg_lambda=interaction_reg_lambda,
            use_gxe_moe=False,
            gxe_moe_num_experts=4,
            gxe_moe_hidden_dim=None,
            gxe_moe_temperature=1.0,
        )

    def forward(self, geno_vec, env_ts, loc_ids, year_ids, pop_ids, stage: int = 0, return_components: bool = False):
        g_repr = self.geno_proj(geno_vec)
        env_repr = self.env_encoder(env_ts)
        loc_emb = self.location_embedding(loc_ids)
        year_emb = self.year_embedding(year_ids)
        pop_emb = self.pop_embedding(pop_ids)
        return self.fusion(
            g_repr,
            env_repr,
            loc_emb,
            year_emb,
            pop_emb,
            env_repr_gxe=None,
            stage=stage,
            return_components=return_components
        )

def load_genotypes_plink(prefix: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load genotype dosage matrix from PLINK files. Returns (X [N x P], iid list).
    """
    bim, fam, bed = read_plink(prefix, verbose=False)
    X = bed.compute().T  # [samples, snps]
    ids = fam["iid"].astype(str).tolist()
    if _GENIC_SNP_IDS:  # functional-SNP prioritization (additive/dosage branch only)
        snp_ids = bim["snp"].astype(str).tolist()
        keep = [j for j, s in enumerate(snp_ids) if s in _GENIC_SNP_IDS]
        if 0 < len(keep) < X.shape[1]:
            X = X[:, keep]
            print(f"[GENIC] additive branch restricted to {len(keep)}/{len(snp_ids)} genic SNPs.", flush=True)
    return X.astype(float), ids


def build_geno_transform(
    X: np.ndarray,
    ids: List[str],
    train_ids: List[str],
    use_pca: bool = SIMPLE_GENO_PCA,
    n_pcs: int = SIMPLE_GENO_N_PCS,
    standardize: bool = SIMPLE_GENO_STANDARDIZE,
    return_assets: bool = False
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Fit scaler/PCA on train genotypes and return transformed vectors mapped by SampleID.
    When return_assets is True, also return (components, mean, std) for reuse.
    """
    id_to_idx = {str(iid): i for i, iid in enumerate(ids)}
    tr_idx = [id_to_idx[str(k)] for k in train_ids if str(k) in id_to_idx]
    if not tr_idx:
        raise RuntimeError("No training genotypes found to fit scaler/PCA.")
    X_tr = X[tr_idx]
    scaler = None
    comps = None
    mean_arr = None
    std_arr = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X_tr)
        mean_arr = scaler.mean_.astype(np.float32)
        std_arr = scaler.scale_.astype(np.float32)
    if use_pca:
        n_comp = min(X_tr.shape[1], max(1, min(n_pcs, X_tr.shape[0])))
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr = pca.fit_transform(X_tr)
        comps = pca.components_.astype(np.float32)
        def _tx(arr):
            arr2 = scaler.transform(arr) if scaler is not None else arr
            return pca.transform(arr2)
    else:
        def _tx(arr):
            return scaler.transform(arr) if scaler is not None else arr
    geno_map = {}
    for iid, idx in id_to_idx.items():
        vec = _tx(X[idx:idx + 1]).reshape(-1)
        geno_map[iid] = vec.astype(np.float32)
    dim = len(next(iter(geno_map.values()))) if geno_map else 0
    if return_assets:
        return geno_map, dim, (comps, mean_arr, std_arr)
    return geno_map, dim

class ChromomapTensorDataset(Dataset):
    """
    Simple loader for the hierarchical ChromoMap tensors written by integrated_tile_generation.py.
    Each sample has a single NPZ with fields: tensor [C, T, F], mask [C, T], row_labels, positions_bp.
    """
    def __init__(
        self,
        sample_ids: List[str],
        metadata: pd.DataFrame,
        environment_data: pd.DataFrame,
        env_data_dict: Optional[Dict[str, np.ndarray]],
        target_col: str,
        tensor_dir: str = TENSOR_DIR,
        sample_key_to_sid: Optional[Dict[str, str]] = None,
        metadata_key_col: str = "SampleID",
        standardize_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        residual_targets: Optional[Dict[str, float]] = None,
        env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        self.sample_ids = sample_ids
        self.sample_key_to_sid = sample_key_to_sid or {k: k for k in sample_ids}
        self.meta_index_col = metadata_key_col
        self.metadata = metadata.set_index(metadata_key_col)
        self.environment = environment_data
        self.env_data_dict = env_data_dict or {}
        self.target_col = target_col
        self.tensor_dir = tensor_dir
        self.standardize_target = standardize_target
        self.target_mean = float(target_mean)
        self.target_std = float(target_std if target_std > 1e-8 else 1.0)
        self.residual_targets = residual_targets or {}
        self.env_target_stats = env_target_stats or {}
        self.sample_env_idx, env_missing_keys = _build_env_index_map(
            sample_ids=self.sample_ids,
            sample_key_to_sid=self.sample_key_to_sid,
            metadata_indexed=self.metadata,
            env_data_dict=self.env_data_dict
        )
        self.env_missing_keys = env_missing_keys
        if env_data_dict is not None and env_missing_keys:
            miss_pct = 100.0 * len(env_missing_keys) / max(1, len(self.sample_ids))
            msg = (
                f"Missing temporal environment rows for {len(env_missing_keys)}/{len(self.sample_ids)} "
                f"samples ({miss_pct:.1f}%). Examples: {env_missing_keys[:5]}"
            )
            if ENV_LOOKUP_FAIL_ON_MISSING:
                raise ValueError(msg)
            logging.warning(msg)

        # Infer feature dims from first available tensor
        self.feature_dim = None
        self.num_chromosomes = None
        self.max_block_id_est = 0
        self.feature_names: List[str] = []
        self.block_id_raw_idx: Optional[int] = None
        self.quality_idx: Optional[int] = None
        self.te_hotspot_idx: Optional[int] = None
        self.is_te_idx: Optional[int] = None
        self.is_genic_idx: Optional[int] = None
        self.is_promoter_idx: Optional[int] = None
        self.dosage_idx: Optional[int] = None
        self.token_rank_idx: Optional[int] = None
        self.dosage_local_mean_idx: Optional[int] = None
        self.dosage_local_std_idx: Optional[int] = None
        self.te_dist_idx: Optional[int] = None
        self.gene_dist_idx: Optional[int] = None
        self.block_gene_count_idx: Optional[int] = None
        self.block_snp_density_idx: Optional[int] = None
        self.block_mean_maf_idx: Optional[int] = None
        self.drop_feature_indices: List[int] = []
        self._feature_dim_mismatch_warned = False
        self._feature_name_mismatch_warned = False
        for key in sample_ids:
            sid = self.sample_key_to_sid.get(key, key)
            p = os.path.join(self.tensor_dir, sid, f"{sid}_tensor.npz")
            if not os.path.exists(p):
                continue
            try:
                z = np.load(p, allow_pickle=False)
                t = z["tensor"]
                names: List[str] = []
                drop_idx: List[int] = []
                if "feature_names_bytes" in z:
                    try:
                        names_str = bytes(z["feature_names_bytes"].tolist()).decode("ascii")
                        names = names_str.split(",") if names_str else []
                        for feat_name in ("snp_importance", "saliency_mask"):
                            if feat_name in names:
                                drop_idx.append(names.index(feat_name))
                        if drop_idx:
                            for idx in sorted(drop_idx, reverse=True):
                                if 0 <= idx < len(names):
                                    names.pop(idx)
                        if "block_id_raw" in names:
                            self.block_id_raw_idx = names.index("block_id_raw")
                        if "te_hotspot_flag" in names:
                            self.te_hotspot_idx = names.index("te_hotspot_flag")
                        if "is_te" in names:
                            self.is_te_idx = names.index("is_te")
                        if "is_genic" in names:
                            self.is_genic_idx = names.index("is_genic")
                        if "is_promoter" in names:
                            self.is_promoter_idx = names.index("is_promoter")
                        if "block_gene_count_norm" in names:
                            self.block_gene_count_idx = names.index("block_gene_count_norm")
                        if "block_snp_density_norm" in names:
                            self.block_snp_density_idx = names.index("block_snp_density_norm")
                        if "block_mean_maf_norm" in names:
                            self.block_mean_maf_idx = names.index("block_mean_maf_norm")
                        for i, nm in enumerate(names):
                            if nm.startswith("quality_"):
                                self.quality_idx = i
                                break
                        names_lower = [nm.lower() for nm in names]
                        def _find_idx(candidates: Tuple[str, ...]) -> Optional[int]:
                            for cand in candidates:
                                cand_lower = cand.lower()
                                if cand_lower in names_lower:
                                    return names_lower.index(cand_lower)
                            return None
                        if self.te_hotspot_idx is None:
                            self.te_hotspot_idx = _find_idx(
                                ("te_hotspot_flag", "te_hotspot_mask", "is_te_hotspot", "te_hotspot")
                            )
                        if self.te_hotspot_idx is None and self.is_te_idx is not None:
                            # Fallback: no dedicated hotspot flag; reuse is_te channel.
                            self.te_hotspot_idx = self.is_te_idx
                        if self.is_promoter_idx is None:
                            self.is_promoter_idx = _find_idx(
                                ("is_promoter", "is_gene_promoter", "gene_promoter")
                            )
                        self.dosage_idx = _find_idx(
                            ("dosage", "dosage_norm", "dosage_raw", "dosage_scaled", "dosage_float", "dosage_prior")
                        )
                        if self.token_rank_idx is None and "token_rank_norm" in names:
                            self.token_rank_idx = names.index("token_rank_norm")
                        if self.dosage_local_mean_idx is None and "dosage_local_mean" in names:
                            self.dosage_local_mean_idx = names.index("dosage_local_mean")
                        if self.dosage_local_std_idx is None and "dosage_local_std" in names:
                            self.dosage_local_std_idx = names.index("dosage_local_std")
                        self.te_dist_idx = _find_idx(
                            ("te_dist", "te_distance", "te_dist_bp", "dist_te", "te_dist_norm")
                        )
                        self.gene_dist_idx = _find_idx(
                            ("gene_dist", "gene_distance", "gene_dist_bp", "dist_gene", "genic_dist", "genic_distance", "gene_dist_norm")
                        )
                    except Exception:
                        names = []
                        self.block_id_raw_idx = None
                        self.quality_idx = None
                        self.te_hotspot_idx = None
                        self.is_te_idx = None
                        self.is_genic_idx = None
                        self.is_promoter_idx = None
                        self.dosage_idx = None
                        self.token_rank_idx = None
                        self.dosage_local_mean_idx = None
                        self.dosage_local_std_idx = None
                        self.te_dist_idx = None
                        self.gene_dist_idx = None
                        self.block_gene_count_idx = None
                        self.block_snp_density_idx = None
                        self.block_mean_maf_idx = None
                if drop_idx:
                    for idx in sorted(drop_idx, reverse=True):
                        if t.shape[-1] > idx:
                            t = np.delete(t, idx, axis=-1)
                    self.drop_feature_indices = sorted(drop_idx)
                self.feature_names = names
                self._base_feature_dim = t.shape[-1]
                self.feature_dim = t.shape[-1] + (_3D_K if USE_3D_CHANNELS else 0)  # model sees base + appended 3D channels
                self.num_chromosomes = t.shape[0]
                if self.block_id_raw_idx is not None:
                    self.max_block_id_est = int(np.nanmax(t[..., self.block_id_raw_idx]))
                break
            except Exception:
                continue
        if self.feature_dim is None:
            raise RuntimeError("Could not infer tensor shape from any sample NPZ.")
        self.n_chr = self.num_chromosomes or 0

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_key = self.sample_ids[idx]
        sid = self.sample_key_to_sid.get(sample_key, sample_key)
        path = os.path.join(self.tensor_dir, sid, f"{sid}_tensor.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tensor NPZ not found for {sid}: {path}")
        z = np.load(path, allow_pickle=False)
        tensor = z["tensor"].astype(np.float32)
        if self.drop_feature_indices:
            for idx in sorted(self.drop_feature_indices, reverse=True):
                if tensor.shape[-1] > idx:
                    tensor = np.delete(tensor, idx, axis=-1)
        # Align per-sample feature order to the dataset feature names when available.
        if self.feature_names and "feature_names_bytes" in z:
            try:
                names_str = bytes(z["feature_names_bytes"].tolist()).decode("ascii")
                curr_names = names_str.split(",") if names_str else []
                for feat_name in ("snp_importance", "saliency_mask"):
                    if feat_name in curr_names:
                        curr_names.pop(curr_names.index(feat_name))
                if curr_names and curr_names != self.feature_names:
                    name_to_idx = {name: i for i, name in enumerate(curr_names)}
                    aligned = np.zeros((*tensor.shape[:3], len(self.feature_names)), dtype=tensor.dtype)
                    for j, name in enumerate(self.feature_names):
                        idx = name_to_idx.get(name)
                        if idx is not None and idx < tensor.shape[-1]:
                            aligned[..., j] = tensor[..., idx]
                    tensor = aligned
                    if not self._feature_name_mismatch_warned:
                        logging.warning(
                            "Feature name mismatch detected; aligning tensor channels to the dataset feature list."
                        )
                        self._feature_name_mismatch_warned = True
            except Exception:
                pass
        # Fallback: pad or truncate when feature dims differ across samples (BASE channels only).
        _base_fd = getattr(self, "_base_feature_dim", self.feature_dim)
        if tensor.shape[-1] != _base_fd:
            if tensor.shape[-1] > _base_fd:
                tensor = tensor[..., : _base_fd]
            else:
                pad = _base_fd - tensor.shape[-1]
                tensor = np.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, pad)), mode="constant")
            if not self._feature_dim_mismatch_warned:
                logging.warning(
                    "Feature dim mismatch across tensors; padding/truncating to base_feature_dim=%d.",
                    _base_fd
                )
                self._feature_dim_mismatch_warned = True
        mask = z["mask"].astype(np.float32)
        tensor = np.nan_to_num(tensor, nan=0.0)
        # Append opt-in 3D-genome channels (sample-independent; aligned to tensor [C,T]).
        if USE_3D_CHANNELS and _3D_CHANNELS is not None:
            _ch = _3D_CHANNELS
            if _ch.shape[0] == tensor.shape[0] and _ch.shape[1] == tensor.shape[1]:
                tensor = np.concatenate([tensor, _ch], axis=-1)
            else:
                _C0, _T0 = tensor.shape[0], tensor.shape[1]
                _chp = np.zeros((_C0, _T0, _ch.shape[-1]), dtype=tensor.dtype)
                _cc, _tt = min(_C0, _ch.shape[0]), min(_T0, _ch.shape[1])
                _chp[:_cc, :_tt, :] = _ch[:_cc, :_tt, :]
                tensor = np.concatenate([tensor, _chp], axis=-1)
        pad_mask = mask <= 0.0  # original mask is 1 on token, 0 on pad
        row_labels = z.get("row_labels", np.arange(tensor.shape[0])).astype(str)
        chr_lengths = z.get("chr_lengths", np.ones((tensor.shape[0],), dtype=np.int64))
        positions_bp = z.get("positions_bp", None)

        genomic_tensor = torch.from_numpy(tensor)
        mask_tensor = torch.from_numpy(pad_mask.astype(np.bool_))  # True for padding
        row_label_tensor = torch.arange(genomic_tensor.shape[0], dtype=torch.long)

        # Temporal env lookup (or wide env vector when USE_ENV_MATRIX_AS_MLP is enabled)
        if USE_ENV_MATRIX_AS_MLP:
            env_ts_tensor = torch.zeros((N_ENV_FEATURES_PER_MONTH,), dtype=torch.float32)
        else:
            env_ts_tensor = torch.zeros((N_MONTHS, N_ENV_FEATURES_PER_MONTH), dtype=torch.float32)
        env_wide_arr = self.env_data_dict.get("env_wide", None)
        loc_tensor = torch.tensor(0, dtype=torch.long)
        year_tensor = torch.tensor(0, dtype=torch.long)
        env_idx = self.sample_env_idx.get(str(sample_key))
        found_env = env_idx is not None
        if env_idx is not None:
            env_arr = self.env_data_dict.get("temporal", None)
            if not USE_ENV_MATRIX_AS_MLP and env_arr is not None and len(env_arr) > env_idx:
                env_ts_tensor = torch.from_numpy(env_arr[env_idx]).float()
            if USE_ENV_MATRIX_AS_MLP and env_wide_arr is not None and len(env_wide_arr) > env_idx:
                env_ts_tensor = torch.from_numpy(env_wide_arr[env_idx]).float()
            loc_arr = self.env_data_dict.get("location_ids", None)
            yr_arr = self.env_data_dict.get("year_ids", None)
            if loc_arr is not None and len(loc_arr) > env_idx:
                loc_tensor = torch.tensor(int(loc_arr[env_idx]), dtype=torch.long)
            if yr_arr is not None and len(yr_arr) > env_idx:
                year_tensor = torch.tensor(int(yr_arr[env_idx]), dtype=torch.long)

        # Population
        try:
            meta_row = self.metadata.loc[sample_key]
        except KeyError:
            logging.warning(f"Metadata missing sample_key={sample_key}; defaulting population/target to 0.")
            meta_row = pd.Series({"PopID": 0, "Pop_Code": 0, self.target_col: 0})
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]
        pop_val = meta_row.get("PopID", meta_row.get("Pop_Code", 0))
        pop_tensor = torch.tensor(_to_int_scalar(pop_val, default=0) if USE_POPULATION_EMBEDDING else 0, dtype=torch.long)

        # Target
        target_val = meta_row.get(self.target_col, np.nan)
        if isinstance(target_val, pd.Series):
            target_val = target_val.iloc[0]
        try:
            target = float(target_val) if pd.notna(target_val) else 0.0
        except Exception:
            target = 0.0
        if self.residual_targets and sample_key in self.residual_targets:
            target = float(self.residual_targets[sample_key])
        if self.env_target_stats:
            env_id = None
            if env_idx is not None:
                env_pair_ids = self.env_data_dict.get("env_pair_ids")
                if env_pair_ids is not None and len(env_pair_ids) > env_idx:
                    env_id = int(env_pair_ids[env_idx])
            if env_id is None:
                env_id = _resolve_env_id(int(loc_tensor.item()), int(year_tensor.item()))
            mean, std = self.env_target_stats.get(env_id, (0.0, 1.0))
            std = std if std > 1e-8 else 1.0
            target = (target - mean) / std
        if self.standardize_target:
            target = (target - self.target_mean) / self.target_std
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return (
            genomic_tensor,
            mask_tensor,
            row_label_tensor,
            _ecopop_env_hook(env_ts_tensor),
            loc_tensor,
            year_tensor,
            pop_tensor,
            target_tensor,
            sample_key
        )


class RawGenoEnvDataset(Dataset):
    """
    Dataset for raw genotype vectors (simple branch) with matching environment + metadata.
    """
    def __init__(
        self,
        sample_ids: List[str],
        metadata: pd.DataFrame,
        environment_data: pd.DataFrame,
        env_data_dict: Optional[Dict[str, np.ndarray]],
        geno_map: Dict[str, np.ndarray],
        target_col: str,
        sample_key_to_sid: Optional[Dict[str, str]] = None,
        metadata_key_col: str = "SampleID",
        standardize_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        residual_targets: Optional[Dict[str, float]] = None,
        env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        self.sample_ids = sample_ids
        self.sample_key_to_sid = sample_key_to_sid or {k: k for k in sample_ids}
        self.meta_index_col = metadata_key_col
        self.metadata = metadata.set_index(metadata_key_col)
        self.environment = environment_data
        self.env_data_dict = env_data_dict or {}
        self.geno_map = geno_map
        self.target_col = target_col
        self.standardize_target = standardize_target
        self.target_mean = float(target_mean)
        self.target_std = float(target_std if target_std > 1e-8 else 1.0)
        self.residual_targets = residual_targets or {}
        self.env_target_stats = env_target_stats or {}
        self.sample_env_idx, env_missing_keys = _build_env_index_map(
            sample_ids=self.sample_ids,
            sample_key_to_sid=self.sample_key_to_sid,
            metadata_indexed=self.metadata,
            env_data_dict=self.env_data_dict
        )
        self.env_missing_keys = env_missing_keys
        self.geno_dim = len(next(iter(geno_map.values()))) if geno_map else 0

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_key = self.sample_ids[idx]
        sid = self.sample_key_to_sid.get(sample_key, sample_key)
        geno_vec = self.geno_map.get(str(sid))
        if geno_vec is None:
            raise KeyError(f"No genotype vector for {sid}")
        geno_tensor = torch.from_numpy(np.asarray(geno_vec, dtype=np.float32))

        # Environment lookup (temporal or wide)
        if USE_ENV_MATRIX_AS_MLP:
            env_ts_tensor = torch.zeros((N_ENV_FEATURES_PER_MONTH,), dtype=torch.float32)
        else:
            env_ts_tensor = torch.zeros((N_MONTHS, N_ENV_FEATURES_PER_MONTH), dtype=torch.float32)
        loc_tensor = torch.tensor(0, dtype=torch.long)
        year_tensor = torch.tensor(0, dtype=torch.long)
        env_idx = self.sample_env_idx.get(str(sample_key))
        if env_idx is not None:
            env_arr = self.env_data_dict.get("temporal", None)
            if not USE_ENV_MATRIX_AS_MLP and env_arr is not None and len(env_arr) > env_idx:
                env_ts_tensor = torch.from_numpy(env_arr[env_idx]).float()
            env_wide_arr = self.env_data_dict.get("env_wide", None)
            if USE_ENV_MATRIX_AS_MLP and env_wide_arr is not None and len(env_wide_arr) > env_idx:
                env_ts_tensor = torch.from_numpy(env_wide_arr[env_idx]).float()
            loc_arr = self.env_data_dict.get("location_ids", None)
            yr_arr = self.env_data_dict.get("year_ids", None)
            if loc_arr is not None and len(loc_arr) > env_idx:
                loc_tensor = torch.tensor(int(loc_arr[env_idx]), dtype=torch.long)
            if yr_arr is not None and len(yr_arr) > env_idx:
                year_tensor = torch.tensor(int(yr_arr[env_idx]), dtype=torch.long)

        # Population + target
        try:
            meta_row = self.metadata.loc[sample_key]
        except KeyError:
            logging.warning(f"Metadata missing sample_key={sample_key}; defaulting population/target to 0.")
            meta_row = pd.Series({"PopID": 0, "Pop_Code": 0, self.target_col: 0})
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]
        pop_val = meta_row.get("PopID", meta_row.get("Pop_Code", 0))
        pop_tensor = torch.tensor(_to_int_scalar(pop_val, default=0) if USE_POPULATION_EMBEDDING else 0, dtype=torch.long)

        target_val = meta_row.get(self.target_col, np.nan)
        if isinstance(target_val, pd.Series):
            target_val = target_val.iloc[0]
        try:
            target = float(target_val) if pd.notna(target_val) else 0.0
        except Exception:
            target = 0.0
        if self.residual_targets and sample_key in self.residual_targets:
            target = float(self.residual_targets[sample_key])
        if self.env_target_stats:
            env_id = _resolve_env_id(int(loc_tensor.item()), int(year_tensor.item()))
            mean, std = self.env_target_stats.get(env_id, (0.0, 1.0))
            std = std if std > 1e-8 else 1.0
            target = (target - mean) / std
        if self.standardize_target:
            target = (target - self.target_mean) / self.target_std
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return (
            geno_tensor,
            _ecopop_env_hook(env_ts_tensor),
            loc_tensor,
            year_tensor,
            pop_tensor,
            target_tensor,
            sample_key
        )


def chromomap_collate_fn(batch):
    """
    Collate ChromomapTensorDataset samples into batch tensors.
    """
    genomics, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sids = zip(*batch)
    return (
        torch.stack(genomics, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(row_labels, dim=0),
        torch.stack(env_ts, dim=0),
        torch.stack(loc_ids, dim=0),
        torch.stack(year_ids, dim=0),
        torch.stack(pop_ids, dim=0),
        torch.stack(targets, dim=0),
        list(sids),
    )

def raw_geno_collate_fn(batch):
    geno, env_ts, loc_ids, year_ids, pop_ids, targets, sids = zip(*batch)
    return (
        torch.stack(geno, dim=0),
        torch.stack(env_ts, dim=0),
        torch.stack(loc_ids, dim=0),
        torch.stack(year_ids, dim=0),
        torch.stack(pop_ids, dim=0),
        torch.stack(targets, dim=0),
        list(sids),
    )

def _find_feature_indices(feature_names: List[str], patterns: List[str]) -> List[int]:
    out: Set[int] = set()
    for pat in patterns:
        if pat.endswith("*"):
            pref = pat[:-1]
            out.update(i for i, n in enumerate(feature_names) if n.startswith(pref))
        elif pat.startswith("*"):
            suf = pat[1:]
            out.update(i for i, n in enumerate(feature_names) if n.endswith(suf))
        else:
            out.update(i for i, n in enumerate(feature_names) if n == pat)
    return sorted(out)


def apply_channel_dropout(
    genomic: torch.Tensor,
    pad_mask: torch.Tensor,
    feature_names: List[str],
    p_group_drop: float = CHANNEL_GROUP_DROP_P,
    p_channel_drop: float = CHANNEL_DROP_P,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Structured channel dropout: zeros selected feature channels across all tokens.
    pad_mask ignored (channels zeroed globally). Returns augmented genomic and dropped indices.
    """
    if (p_group_drop <= 0.0 and p_channel_drop <= 0.0) or not feature_names:
        return genomic, []
    device = genomic.device
    F = genomic.shape[-1]
    gen = rng if rng is not None else torch.Generator(device=device)

    groups = {
        "TE_GROUP": ["is_te", "te_dist_bp"],
        "GENE_GROUP": ["is_genic", "is_promoter", "gene_dist_bp"],
        "BLOCK_SUMMARY_GROUP": ["block_gene_count_norm", "block_snp_density_norm", "block_mean_maf_norm"],
        "HAP_BLOCK_GROUP": ["block_id_norm", "block_id_raw", "block_len_norm", "inblock_ld", "is_block_boundary"],
        "HOMOLOGY_GROUP": ["hom_group_id", "hom_pair_id"],
    }
    always_keep = set(_find_feature_indices(feature_names, ["dosage_norm", "quality_", "quality_*", "pos_enc_"]))
    drop_idx: Set[int] = set()

    if p_group_drop > 0:
        for gname, pats in groups.items():
            idxs = [i for i in _find_feature_indices(feature_names, pats) if i not in always_keep]
            if not idxs:
                continue
            if torch.rand((), generator=gen, device=device).item() < p_group_drop:
                drop_idx.update(idxs)

    if p_channel_drop > 0:
        candidates = [i for i in range(F) if i not in always_keep]
        if candidates:
            u = torch.rand(len(candidates), generator=gen, device=device)
            for i, r in zip(candidates, u):
                if r.item() < p_channel_drop:
                    drop_idx.add(i)

    drop_idx = sorted(i for i in drop_idx if 0 <= i < F and i not in always_keep)
    if len(drop_idx) >= F:
        drop_idx = []
    if drop_idx:
        genomic = genomic.clone()
        genomic[..., drop_idx] = 0.0
    return genomic, drop_idx


class GenomicOnlyTensorDataset(Dataset):
    """
    Minimal dataset for SSL pretraining: loads only genomic tensor + pad mask by SampleID.
    Aligns feature dimension and optional feature order to match the supervised dataset.
    """
    def __init__(
        self,
        sample_ids: List[str],
        tensor_dir: str,
        feature_dim: Optional[int] = None,
        drop_feature_indices: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.sample_ids = [str(s) for s in sample_ids]
        self.tensor_dir = tensor_dir
        self.feature_dim = feature_dim
        self.drop_feature_indices = sorted(set(drop_feature_indices or []))
        self.feature_names = feature_names or []
        if self.feature_dim is None:
            # Infer feature dim from the first available tensor after optional drops.
            for sid in self.sample_ids:
                path = os.path.join(self.tensor_dir, sid, f"{sid}_tensor.npz")
                if not os.path.exists(path):
                    continue
                try:
                    z = np.load(path, allow_pickle=False)
                    t = z["tensor"]
                    if self.drop_feature_indices:
                        for idx in sorted(self.drop_feature_indices, reverse=True):
                            if t.shape[-1] > idx:
                                t = np.delete(t, idx, axis=-1)
                    if self.feature_names and "feature_names_bytes" in z:
                        names_str = bytes(z["feature_names_bytes"].tolist()).decode("ascii")
                        curr_names = names_str.split(",") if names_str else []
                        for feat_name in ("snp_importance", "saliency_mask"):
                            if feat_name in curr_names:
                                curr_names.pop(curr_names.index(feat_name))
                        if curr_names and curr_names != self.feature_names:
                            name_to_idx = {n: i for i, n in enumerate(curr_names)}
                            aligned = np.zeros((*t.shape[:3], len(self.feature_names)), dtype=t.dtype)
                            for j, name in enumerate(self.feature_names):
                                idx = name_to_idx.get(name)
                                if idx is not None and idx < t.shape[-1]:
                                    aligned[..., j] = t[..., idx]
                            t = aligned
                    self.feature_dim = t.shape[-1]
                    break
                except Exception:
                    continue
            if self.feature_dim is None:
                raise RuntimeError("Could not infer feature_dim for GenomicOnlyTensorDataset.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        path = os.path.join(self.tensor_dir, sid, f"{sid}_tensor.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tensor NPZ not found for {sid}: {path}")
        z = np.load(path, allow_pickle=False)
        tensor = z["tensor"].astype(np.float32)
        if self.drop_feature_indices:
            for drop_idx in sorted(self.drop_feature_indices, reverse=True):
                if tensor.shape[-1] > drop_idx:
                    tensor = np.delete(tensor, drop_idx, axis=-1)
        if self.feature_names and "feature_names_bytes" in z:
            try:
                names_str = bytes(z["feature_names_bytes"].tolist()).decode("ascii")
                curr_names = names_str.split(",") if names_str else []
                for feat_name in ("snp_importance", "saliency_mask"):
                    if feat_name in curr_names:
                        curr_names.pop(curr_names.index(feat_name))
                if curr_names and curr_names != self.feature_names:
                    name_to_idx = {n: i for i, n in enumerate(curr_names)}
                    aligned = np.zeros((*tensor.shape[:3], len(self.feature_names)), dtype=tensor.dtype)
                    for j, name in enumerate(self.feature_names):
                        idx = name_to_idx.get(name)
                        if idx is not None and idx < tensor.shape[-1]:
                            aligned[..., j] = tensor[..., idx]
                    tensor = aligned
            except Exception:
                pass
        if self.feature_dim is not None and tensor.shape[-1] != self.feature_dim:
            if tensor.shape[-1] > self.feature_dim:
                tensor = tensor[..., : self.feature_dim]
            else:
                pad = self.feature_dim - tensor.shape[-1]
                tensor = np.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, pad)), mode="constant")
        mask = z["mask"].astype(np.float32)
        pad_mask = mask <= 0.0  # True for padding
        tensor = np.nan_to_num(tensor, nan=0.0)
        return torch.from_numpy(tensor), torch.from_numpy(pad_mask.astype(np.bool_))


def collate_genomic_only(batch):
    genomics, masks = zip(*batch)
    return torch.stack(genomics, dim=0), torch.stack(masks, dim=0)


def _augment_genomic_for_ssl(
    genomic: torch.Tensor,
    mask: torch.Tensor,
    token_drop_p: float,
    feature_noise: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Drop a random subset of valid tokens and add light noise for SimCLR-style SSL.
    """
    valid = ~mask
    drop = (torch.rand_like(valid.float()) < token_drop_p) & valid
    aug_mask = mask | drop
    aug_genomic = genomic.masked_fill(aug_mask.unsqueeze(-1), 0.0)
    if feature_noise > 0:
        noise = torch.randn_like(aug_genomic) * feature_noise
        aug_genomic = aug_genomic + noise * (~aug_mask).unsqueeze(-1).float()
    return aug_genomic, aug_mask


def _simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(-1e9)
    pos_idx = (torch.arange(z.size(0), device=z.device) + z1.size(0)) % z.size(0)
    return F.cross_entropy(sim, pos_idx)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    eps: float = 1e-4
) -> torch.Tensor:
    # Invariance
    sim = F.mse_loss(z1, z2)

    def _var(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _cov(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        n, d = z.shape
        cov = (z.T @ z) / max(1, n - 1)
        cov_offdiag = cov - torch.diag(torch.diag(cov))
        return (cov_offdiag ** 2).sum() / d

    var = _var(z1) + _var(z2)
    cov = _cov(z1) + _cov(z2)
    return sim_w * sim + var_w * var + cov_w * cov


def run_genomic_simclr_pretraining(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = SIMCLR_EPOCHS,
    lr: float = SIMCLR_LR,
    temp: float = SIMCLR_TEMP,
    token_drop_p: float = SIMCLR_TOKEN_DROP,
    feature_noise: float = SIMCLR_FEATURE_NOISE
) -> None:
    """
    Contrastive SSL over genomic embeddings using encode_genomic_only on train split.
    """
    if not PRETRAIN_GENOMIC_SIMCLR:
        return
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=MAIN_WEIGHT_DECAY)
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for batch in loader:
            genomic, masks = None, None
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                genomic = batch[0]
                masks = batch[1]
            if genomic is None or masks is None:
                continue
            genomic = genomic.to(device)
            masks = masks.to(device)
            g1, m1 = _augment_genomic_for_ssl(genomic, masks, token_drop_p, feature_noise)
            g2, m2 = _augment_genomic_for_ssl(genomic, masks, token_drop_p, feature_noise)
            z1 = model.encode_genomic_only(g1, m1)
            z2 = model.encode_genomic_only(g2, m2)
            if SIMCLR_USE_VICREG:
                loss = vicreg_loss(z1, z2)
            else:
                loss = _simclr_loss(z1, z2, temp=temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg = total_loss / max(1, batches)
        logging.info("[SimCLR] Epoch %d/%d loss=%.4f", ep, epochs, avg)


def _is_tensor_model(model: nn.Module) -> bool:
    """True when the model consumes hierarchical ChromoMap tensors directly."""
    if isinstance(model, GxE_Transformer_Tensor):
        return True
    if hasattr(model, "base"):
        return isinstance(getattr(model, "base"), GxE_Transformer_Tensor)
    return False


def _supports_gxe_stage(model: nn.Module) -> bool:
    """True when model forward supports staged GxE training."""
    try:
        sig = inspect.signature(model.forward)
        return "gxe_stage" in sig.parameters or "stage" in sig.parameters
    except Exception:
        return False


def _stage_kwargs(model: nn.Module, gxe_stage: int, return_components: bool = False) -> Dict[str, Any]:
    """
    Build forward kwargs for staged GxE models without passing unsupported keys.
    """
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(model.forward)
        if "gxe_stage" in sig.parameters:
            kwargs["gxe_stage"] = gxe_stage
        elif "stage" in sig.parameters:
            kwargs["stage"] = gxe_stage
        if return_components and "return_components" in sig.parameters:
            kwargs["return_components"] = True
    except Exception:
        return {}
    return kwargs


def _forward_maybe_row_labels(model, args, kwargs, row_labels):
    """
    Call model forward, only supplying row_labels when the signature supports it.
    """
    try:
        return model(*args, row_labels=row_labels, **kwargs)
    except TypeError:
        return model(*args, **kwargs)


def _build_dosage_override_batch(sample_keys, device: torch.device) -> Optional[torch.Tensor]:
    """
    Build [B, P] PLINK dosage vectors for a tensor batch when DOSAGE_SOURCE='plink'.
    Returns None when override is disabled/unavailable.
    """
    global _DOSAGE_OVERRIDE_MISSING_WARNED
    if not DOSAGE_OVERRIDE_ENABLED or not DOSAGE_OVERRIDE_MAP:
        return None
    if sample_keys is None:
        return None
    rows: List[np.ndarray] = []
    missing: List[str] = []
    for key in sample_keys:
        key_str = str(key)
        sid = str(DOSAGE_OVERRIDE_SAMPLE_TO_SID.get(key_str, key_str))
        vec = DOSAGE_OVERRIDE_MAP.get(sid)
        if vec is None:
            missing.append(sid)
            if DOSAGE_OVERRIDE_DIM <= 0:
                return None
            vec = np.zeros((DOSAGE_OVERRIDE_DIM,), dtype=np.float32)
        rows.append(np.asarray(vec, dtype=np.float32))
    if not rows:
        return None
    if missing and not _DOSAGE_OVERRIDE_MISSING_WARNED:
        logging.warning(
            "Missing PLINK dosage vectors for %d samples (examples=%s); filling with zeros.",
            len(missing),
            missing[:3]
        )
        _DOSAGE_OVERRIDE_MISSING_WARNED = True
    arr = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(arr).to(device=device)


def _pool_tokens_windowed(seq: torch.Tensor, mask: torch.Tensor, window_size: int):
    """
    Pool a single chromosome strip [B, T, F] into windows, returning pooled tokens and mask.
    Output token dim = 2*F + 1 (mean, std, count).
    """
    B, T, F = seq.shape
    window = max(1, int(window_size))
    num_windows = math.ceil(T / float(window))
    tokens = []
    masks = []
    for w in range(num_windows):
        start = w * window
        end = min((w + 1) * window, T)
        win = seq[:, start:end, :]           # [B, Tw, F]
        win_mask = mask[:, start:end]        # [B, Tw] True=pad
        valid = (~win_mask).float()          # 1 where real token
        counts = valid.sum(dim=1, keepdim=True)  # [B,1]
        counts_safe = torch.clamp(counts, min=1.0)
        mean = (win * valid.unsqueeze(-1)).sum(dim=1) / counts_safe
        diff = (win - mean.unsqueeze(1)) * valid.unsqueeze(-1)
        var = (diff.pow(2).sum(dim=1) / counts_safe).clamp(min=1e-8)
        std = torch.sqrt(var)
        count_feat = counts.squeeze(-1)
        tok = torch.cat([mean, std, count_feat.unsqueeze(-1)], dim=-1)  # [B, 2F+1]
        tokens.append(tok)
        masks.append(count_feat <= 0.0)
    pooled = torch.stack(tokens, dim=1)          # [B, W, 2F+1]
    pooled_mask = torch.stack(masks, dim=1)      # [B, W]
    return pooled, pooled_mask


def pool_within_chromosomes(genomic: torch.Tensor, mask: torch.Tensor, window_size: int = 100):
    """
    Pool SNPs within each chromosome strip while preserving chromosome structure.
    Input: genomic [B, C, T, F], mask [B, C, T] (True=pad)
    Output: pooled genomic [B, C, W, 2F+1], pooled mask [B, C, W] (True=pad)
    """
    B, C, T, F = genomic.shape
    window = max(1, int(window_size))
    pooled_tokens = []
    pooled_masks = []
    for c in range(C):
        seq = genomic[:, c, :, :]
        m = mask[:, c, :]
        pooled, pooled_mask = _pool_tokens_windowed(seq, m, window)
        pooled_tokens.append(pooled)
        pooled_masks.append(pooled_mask)
    pooled_tokens = torch.stack(pooled_tokens, dim=1)  # [B, C, W, 2F+1]
    pooled_masks = torch.stack(pooled_masks, dim=1)    # [B, C, W]
    return pooled_tokens, pooled_masks


def _apply_temporal_mixup(genomic_seq, genomic_mask, env_ts, targets, alpha):
    """
    Apply mixup augmentation to temporal batches (genomic sequence + env timeseries + target).
    """
    if alpha <= 0.0 or genomic_seq.size(0) < 2:
        return genomic_seq, genomic_mask, env_ts, targets

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(genomic_seq.size(0), device=genomic_seq.device)
    lam_tensor = torch.tensor(lam, dtype=genomic_seq.dtype, device=genomic_seq.device)

    genomic_seq = genomic_seq * lam_tensor + genomic_seq[perm] * (1.0 - lam_tensor)
    env_ts = env_ts * lam_tensor + env_ts[perm] * (1.0 - lam_tensor)
    genomic_mask = genomic_mask | genomic_mask[perm]
    targets = targets * lam_tensor + targets[perm] * (1.0 - lam_tensor)
    return genomic_seq, genomic_mask, env_ts, targets


def _apply_temporal_mixup_environment_aware(genomic_seq, genomic_mask, env_ts, targets, alpha, env_ids):
    """
    Apply mixup only within the same environment (identified by env_ids).
    """
    if alpha <= 0.0 or genomic_seq.size(0) < 2:
        return genomic_seq, genomic_mask, env_ts, targets

    unique_envs = torch.unique(env_ids)
    pairs = []
    for env in unique_envs:
        env_mask = (env_ids == env).nonzero(as_tuple=True)[0]
        if env_mask.numel() < 2:
            continue
        perm = torch.randperm(env_mask.numel(), device=env_ids.device)
        pairs.extend([(env_mask[i], env_mask[perm[i]]) for i in range(env_mask.numel())])

    if not pairs:
        return genomic_seq, genomic_mask, env_ts, targets

    lam = np.random.beta(alpha, alpha)
    lam_tensor = torch.tensor(lam, dtype=genomic_seq.dtype, device=genomic_seq.device)

    g_out = genomic_seq.clone()
    env_out = env_ts.clone()
    mask_out = genomic_mask.clone()
    targ_out = targets.clone()

    for src, tgt in pairs:
        g_out[src] = genomic_seq[src] * lam_tensor + genomic_seq[tgt] * (1.0 - lam_tensor)
        env_out[src] = env_ts[src] * lam_tensor + env_ts[tgt] * (1.0 - lam_tensor)
        mask_out[src] = genomic_mask[src] | genomic_mask[tgt]
        targ_out[src] = targets[src] * lam_tensor + targets[tgt] * (1.0 - lam_tensor)

    return g_out, mask_out, env_out, targ_out


def _perturb_environment(env_ts: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Add Gaussian noise scaled to per-feature std (percentage set by sigma).
    """
    if sigma <= 0.0:
        return env_ts
    with torch.no_grad():
        feat_std = env_ts.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(env_ts) * (feat_std * sigma)
        return env_ts + noise


def apply_weather_jitter(env_ts: torch.Tensor, sigma: float = ENV_PERTURB_SIGMA) -> torch.Tensor:
    """
    Wrapper for light temporal augmentation during training.
    """
    return _perturb_environment(env_ts, sigma)


def _compute_env_ids(loc_ids: torch.Tensor, year_ids: torch.Tensor) -> torch.Tensor:
    if ENV_PAIR_LUT is not None:
        lut = torch.as_tensor(ENV_PAIR_LUT, device=loc_ids.device)
        loc = loc_ids.clamp(0, lut.shape[0] - 1).long()
        yr = year_ids.clamp(0, lut.shape[1] - 1).long()
        env_ids = lut[loc, yr]
        env_ids = torch.where(env_ids < 0, torch.zeros_like(env_ids), env_ids)
        return env_ids.long()
    if ENV_PAIR_TO_ID:
        flat_loc = loc_ids.view(-1).tolist()
        flat_year = year_ids.view(-1).tolist()
        env_list = [_resolve_env_id(int(l), int(y)) for l, y in zip(flat_loc, flat_year)]
        return torch.tensor(env_list, device=loc_ids.device, dtype=torch.long).view_as(loc_ids)
    return (loc_ids.long() * N_YEARS + year_ids.long()) % max(1, NUM_ENVIRONMENTS)


def _inverse_frequency_weights(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(ids, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights[ids]


def _compute_sample_weights(
    env_ids: Optional[torch.Tensor],
    pop_ids: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    weights = None
    if USE_ENV_WEIGHTED_LOSS and env_ids is not None:
        env_weights = _inverse_frequency_weights(env_ids, NUM_ENVIRONMENTS)
        weights = env_weights if weights is None else weights * env_weights
    if POP_LOSS_BOOST > 0 and pop_ids is not None:
        pop_weights = _inverse_frequency_weights(pop_ids.long(), N_POPULATIONS)
        pop_weights = 1.0 + POP_LOSS_BOOST * (pop_weights - 1.0)
        weights = pop_weights if weights is None else weights * pop_weights
    if weights is not None:
        weights = weights / weights.mean().clamp(min=1e-6)
    return weights


def _reduce_loss(loss_tensor: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if loss_tensor.dim() == 0:
        return loss_tensor
    if sample_weights is not None:
        sw = sample_weights
        while sw.dim() < loss_tensor.dim():
            sw = sw.unsqueeze(-1)
        loss_tensor = loss_tensor * sw
    return loss_tensor.mean()


def cutmix_block_mask(
    genomic: torch.Tensor,
    pad_mask: torch.Tensor,
    p_apply: float = BLOCK_MASK_P_APPLY,
    frac_range: Tuple[float, float] = BLOCK_MASK_FRAC_RANGE,
    num_blocks: int = BLOCK_MASK_NUM_BLOCKS,
    choose_chr_mode: str = BLOCK_MASK_CHR_MODE,
    min_block_tokens: int = BLOCK_MASK_MIN_TOKENS,
    keep_first_token: bool = BLOCK_MASK_KEEP_FIRST,
    min_keep_tokens_total: Optional[int] = BLOCK_MASK_MIN_KEEP_TOTAL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix-style contiguous masking within a chromosome strip.
    pad_mask: True=padding. Returns augmented genomic + updated pad_mask.
    """
    if p_apply <= 0.0:
        return genomic, pad_mask
    B, C, T, _ = genomic.shape
    if min_keep_tokens_total is None:
        min_keep_tokens_total = max(64, int(0.02 * C * T))
    x = genomic.clone()
    m = pad_mask.clone()
    valid_counts = (~m).sum(dim=2)  # [B,C]
    device = genomic.device

    for b in range(B):
        if torch.rand(1, device=device).item() > p_apply:
            continue
        for _ in range(num_blocks):
            counts = valid_counts[b]  # [C]
            if counts.max().item() < (min_block_tokens + 2):
                break
            if choose_chr_mode == "proportional":
                probs = counts.float().clamp(min=0)
                probs = probs / probs.sum().clamp(min=1.0)
                c = torch.multinomial(probs, num_samples=1).item()
            else:
                c = torch.randint(0, C, (1,), device=device).item()
            L = int(counts[c].item())
            if L < (min_block_tokens + 2):
                continue
            frac = float(torch.empty(1, device=device).uniform_(frac_range[0], frac_range[1]).item())
            block_len = int(round(frac * L))
            block_len = max(min_block_tokens, min(block_len, L - 1))
            if block_len <= 0:
                continue
            valid_pos = torch.where(~m[b, c])[0]
            if valid_pos.numel() < block_len:
                continue
            start_k = int(torch.randint(0, valid_pos.numel() - block_len + 1, (1,), device=device).item())
            sel = valid_pos[start_k : start_k + block_len]
            if keep_first_token:
                sel = sel[sel != 0]
                if sel.numel() == 0:
                    continue
            m[b, c, sel] = True
            x[b, c, sel, :] = 0.0
            kept = int((~m[b]).sum().item())
            if kept < min_keep_tokens_total:
                # revert this mask
                m[b, c, sel] = pad_mask[b, c, sel]
                x[b, c, sel, :] = genomic[b, c, sel, :]
            else:
                valid_counts[b] = (~m[b]).sum(dim=1)

    return x, m


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    adv_alpha: float = 0.0,
    gxe_stage: int = 0,
    residual_mode: bool = False
):
    model.train()
    total = 0.0
    preds_all, t_all = [], []
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    adv_criterion = nn.CrossEntropyLoss() if USE_ENV_ADVERSARY else None
    feature_names = getattr(loader.dataset, "feature_names", []) if hasattr(loader, "dataset") else []
    for batch_idx, batch in enumerate(loader):
        env_ids: Optional[torch.Tensor] = None
        sample_weights: Optional[torch.Tensor] = None
        if len(batch) != 9:
            raise RuntimeError(f"Unexpected batch length {len(batch)} in train_epoch")
        genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sample_keys = batch
        genomic = genomic.to(device)
        masks = masks.to(device)
        row_labels = row_labels.to(device)
        env_ts = env_ts.to(device)
        loc_ids = loc_ids.to(device)
        year_ids = year_ids.to(device)
        pop_ids = pop_ids.to(device)
        targets = targets.to(device)
        env_ids = _compute_env_ids(loc_ids, year_ids)
        sample_weights = _compute_sample_weights(env_ids, pop_ids)

        if ENV_PERTURB_SIGMA > 0.0 and model.training:
            env_ts = apply_weather_jitter(env_ts, ENV_PERTURB_SIGMA)
        if USE_TOKEN_DROPOUT and tensor_model and model.training and TOKEN_DROPOUT_P > 0.0:
            min_keep = max(64, int(0.01 * genomic.shape[1] * genomic.shape[2]))
            genomic, masks, _ = token_dropout(
                genomic,
                masks,
                p_drop=TOKEN_DROPOUT_P,
                keep_first_token=TOKEN_DROPOUT_KEEP_FIRST,
                min_keep_tokens=min_keep
            )
        if USE_BLOCK_MASKING and tensor_model and model.training and BLOCK_MASK_P_APPLY > 0.0:
            genomic, masks = cutmix_block_mask(
                genomic,
                masks,
                p_apply=BLOCK_MASK_P_APPLY,
                frac_range=BLOCK_MASK_FRAC_RANGE,
                num_blocks=BLOCK_MASK_NUM_BLOCKS,
                choose_chr_mode=BLOCK_MASK_CHR_MODE,
                min_block_tokens=BLOCK_MASK_MIN_TOKENS,
                keep_first_token=BLOCK_MASK_KEEP_FIRST,
                min_keep_tokens_total=BLOCK_MASK_MIN_KEEP_TOTAL or max(64, int(0.02 * genomic.shape[1] * genomic.shape[2])),
            )
        if USE_CHANNEL_DROPOUT and tensor_model and model.training and (CHANNEL_GROUP_DROP_P > 0 or CHANNEL_DROP_P > 0):
            genomic, _ = apply_channel_dropout(
                genomic,
                masks,
                feature_names=feature_names,
                p_group_drop=CHANNEL_GROUP_DROP_P,
                p_channel_drop=CHANNEL_DROP_P,
            )
        if use_chr_pool:
            genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)

        optimizer.zero_grad()
        if mixup_alpha := (MIXUP_ALPHA if model.training else 0.0):
            B, C, T, F = genomic.shape
            genomic_flat = genomic.reshape(B, C * T, F)
            mask_flat = masks.reshape(B, C * T)
            genomic_flat, mask_flat, env_ts, targets = _apply_temporal_mixup_environment_aware(
                genomic_flat, mask_flat, env_ts, targets, mixup_alpha, env_ids
            )
            genomic = genomic_flat.view(B, C, T, F)
            masks = mask_flat.view(B, C, T)
        if ENV_MOD_DROPOUT_P > 0.0 and model.training:
            if torch.rand(1, device=device).item() < ENV_MOD_DROPOUT_P:
                env_ts = torch.zeros_like(env_ts)
        adv_kwargs = {}
        if USE_ENV_ADVERSARY and hasattr(model, "environment_adversary"):
            adv_kwargs = {"adv_alpha": adv_alpha, "return_env_logits": True}
        adv_kwargs.update(_stage_kwargs(model, gxe_stage, return_components=residual_mode))
        dosage_override = _build_dosage_override_batch(sample_keys, device)
        if dosage_override is not None:
            adv_kwargs["dosage_override"] = dosage_override
        out_raw = _forward_maybe_row_labels(
            model,
            (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
            adv_kwargs,
            row_labels
        )
        meta_info: Dict[str, torch.Tensor] = {}
        if isinstance(out_raw, (tuple, list)):
            out = out_raw[0]
            if len(out_raw) > 1 and isinstance(out_raw[1], dict):
                meta_info = out_raw[1]
        else:
            out = out_raw
        if out.dim() > 1:
            out = out.squeeze(-1)

        env_logits = meta_info.get("env_logits") if meta_info else None
        env_ids_for_loss = meta_info.get("env_ids") if meta_info else env_ids
        if residual_mode:
            main_out = meta_info.get("main_out")
            if main_out is None:
                raise RuntimeError("Residual-mode training requires return_components=True with main_out.")
            residual_target = targets - main_out.detach()
            loss_raw = criterion(out, residual_target)
        else:
            loss_raw = criterion(out, targets)
        loss = _reduce_loss(loss_raw, sample_weights)
        if adv_criterion is not None and env_logits is not None and env_ids_for_loss is not None:
            loss = loss + ENV_ADVERSARY_WEIGHT * adv_criterion(env_logits, env_ids_for_loss.long())
        attn_penalty = meta_info.get("attention_diversity")
        interaction_penalty = meta_info.get("interaction_reg")
        if attn_penalty is not None:
            loss = loss + attn_penalty
        if interaction_penalty is not None:
            loss = loss + interaction_penalty
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.detach().item())
        preds_all.extend(out.detach().cpu().numpy().tolist())
        if residual_mode:
            t_all.extend(residual_target.detach().cpu().numpy().tolist())
        else:
            t_all.extend(targets.detach().cpu().numpy().tolist())
    avg = total / max(1, len(loader))
    if len(preds_all) > 1:
        t_arr = np.array(t_all, dtype=float)
        p_arr = np.array(preds_all, dtype=float)
        t_arr, p_arr = _filter_finite_pairs(t_arr, p_arr)
        r2 = r2_score(t_arr, p_arr) if len(p_arr) > 1 else float("nan")
    else:
        r2 = float("nan")
    return avg, r2


def train_epoch_regularized(
    model,
    loader,
    criterion,
    optimizer,
    device,
    l1_weight: float = 0.0,
    mixup_alpha: float = 0.0,
    adv_alpha: float = 0.0,
    gxe_stage: int = 0,
    residual_mode: bool = False,
    aux_dosage_w: float = 0.0,
    aux_complex_w: float = 0.0,
    aux_target_lookup: Optional[Dict[str, List[float]]] = None,
    aux_loss_weight: float = 0.0
):
    """
    Training loop with optional mixup augmentation and L1 regularization to discourage overfitting.
    """
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    preds_all, t_all = [], []
    adv_criterion = nn.CrossEntropyLoss() if USE_ENV_ADVERSARY else None

    params_list = list(model.parameters())
    n_params = max(1, len(params_list))
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    feature_names = getattr(loader.dataset, "feature_names", []) if hasattr(loader, "dataset") else []

    for batch_idx, batch in enumerate(loader):
        sample_weights: Optional[torch.Tensor] = None
        env_ids: Optional[torch.Tensor] = None
        if len(batch) == 9:
            genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sample_keys = batch
            genomic = genomic.to(device)
            masks = masks.to(device)
            row_labels = row_labels.to(device)
            env_ts = env_ts.to(device)
            loc_ids = loc_ids.to(device)
            year_ids = year_ids.to(device)
            pop_ids = pop_ids.to(device)
            targets = targets.to(device)
            env_ids = _compute_env_ids(loc_ids, year_ids)
            sample_weights = _compute_sample_weights(env_ids, pop_ids)

            if ENV_PERTURB_SIGMA > 0.0 and model.training:
                env_ts = apply_weather_jitter(env_ts, ENV_PERTURB_SIGMA)
            if USE_TOKEN_DROPOUT and tensor_model and model.training and TOKEN_DROPOUT_P > 0.0:
                min_keep = max(64, int(0.01 * genomic.shape[1] * genomic.shape[2]))
                genomic, masks, _ = token_dropout(
                    genomic,
                    masks,
                    p_drop=TOKEN_DROPOUT_P,
                    keep_first_token=TOKEN_DROPOUT_KEEP_FIRST,
                    min_keep_tokens=min_keep
                )
            if USE_BLOCK_MASKING and tensor_model and model.training and BLOCK_MASK_P_APPLY > 0.0:
                genomic, masks = cutmix_block_mask(
                    genomic,
                    masks,
                    p_apply=BLOCK_MASK_P_APPLY,
                    frac_range=BLOCK_MASK_FRAC_RANGE,
                    num_blocks=BLOCK_MASK_NUM_BLOCKS,
                    choose_chr_mode=BLOCK_MASK_CHR_MODE,
                    min_block_tokens=BLOCK_MASK_MIN_TOKENS,
                    keep_first_token=BLOCK_MASK_KEEP_FIRST,
                    min_keep_tokens_total=BLOCK_MASK_MIN_KEEP_TOTAL or max(64, int(0.02 * genomic.shape[1] * genomic.shape[2])),
                )
            if USE_CHANNEL_DROPOUT and tensor_model and model.training and (CHANNEL_GROUP_DROP_P > 0 or CHANNEL_DROP_P > 0):
                genomic, _ = apply_channel_dropout(
                    genomic,
                    masks,
                    feature_names=feature_names,
                    p_group_drop=CHANNEL_GROUP_DROP_P,
                    p_channel_drop=CHANNEL_DROP_P,
                )
            if use_chr_pool:
                genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)

            optimizer.zero_grad()
            if mixup_alpha > 0 and genomic.size(0) > 1:
                B, C, T, F = genomic.shape
                genomic_flat = genomic.reshape(B, C * T, F)
                mask_flat = masks.reshape(B, C * T)
                genomic_flat, mask_flat, env_ts, targets = _apply_temporal_mixup_environment_aware(
                    genomic_flat, mask_flat, env_ts, targets, mixup_alpha if model.training else 0.0, env_ids
                )
                genomic = genomic_flat.view(B, C, T, F)
                masks = mask_flat.view(B, C, T)
            if ENV_MOD_DROPOUT_P > 0.0 and model.training:
                if torch.rand(1, device=device).item() < ENV_MOD_DROPOUT_P:
                    env_ts = torch.zeros_like(env_ts)

            adv_kwargs = {}
            if USE_ENV_ADVERSARY and hasattr(model, "environment_adversary"):
                adv_kwargs = {"adv_alpha": adv_alpha, "return_env_logits": True}
            adv_kwargs.update(_stage_kwargs(model, gxe_stage, return_components=(residual_mode or USE_AUX)))
            dosage_override = _build_dosage_override_batch(sample_keys, device)
            if dosage_override is not None:
                adv_kwargs["dosage_override"] = dosage_override
            out_raw = _forward_maybe_row_labels(
                model,
                (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
                adv_kwargs,
                row_labels
            )
        elif len(batch) == 7:
            geno_vec, env_ts, loc_ids, year_ids, pop_ids, targets, sample_keys = batch
            geno_vec = geno_vec.to(device)
            env_ts = env_ts.to(device)
            loc_ids = loc_ids.to(device)
            year_ids = year_ids.to(device)
            pop_ids = pop_ids.to(device)
            targets = targets.to(device)
            env_ids = _compute_env_ids(loc_ids, year_ids)
            sample_weights = _compute_sample_weights(env_ids, pop_ids)
            optimizer.zero_grad()
            adv_kwargs = _stage_kwargs(model, gxe_stage, return_components=(residual_mode or USE_AUX))
            out_raw = model(geno_vec, env_ts, loc_ids, year_ids, pop_ids, **adv_kwargs)
        else:
            raise RuntimeError(f"Unexpected batch length {len(batch)} in train_epoch_regularized")
        meta_info: Dict[str, torch.Tensor] = {}
        if isinstance(out_raw, (tuple, list)):
            out = out_raw[0]
            if len(out_raw) > 1 and isinstance(out_raw[1], dict):
                meta_info = out_raw[1]
        else:
            out = out_raw
        if out.dim() > 1:
            out = out.squeeze(-1)

        env_logits = meta_info.get("env_logits") if meta_info else None
        env_ids_for_loss = meta_info.get("env_ids") if meta_info else env_ids
        if residual_mode:
            main_out = meta_info.get("main_out")
            if main_out is None:
                raise RuntimeError("Residual-mode training requires return_components=True with main_out.")
            residual_target = targets - main_out.detach()
            main_loss_raw = criterion(out, residual_target)
        else:
            main_loss_raw = criterion(out, targets)
        main_loss = _reduce_loss(main_loss_raw, sample_weights)
        # Auxiliary per-branch losses (only when both predictions are available and we're training full model).
        if (
            aux_dosage_w > 0.0
            and aux_complex_w > 0.0
            and not residual_mode
            and gxe_stage == 0
            and meta_info
        ):
            dos = meta_info.get("dosage_pred")
            cx = meta_info.get("complex_pred")
            if dos is not None and cx is not None:
                loss_d = _reduce_loss(criterion(dos, targets), sample_weights)
                loss_c = _reduce_loss(criterion(cx, targets), sample_weights)
                main_loss = main_loss + aux_dosage_w * loss_d + aux_complex_w * loss_c
        # Opt-in multi-trait auxiliary loss (hard parameter sharing). Fully no-op unless
        # ECOPOP_AUX_TARGETS is set (USE_AUX) AND an aux map + positive weight were threaded
        # in AND the model emitted 'aux_preds'. Aux targets are looked up by sample_keys
        # (the metadata keys already carried in the batch) and are NEVER model input.
        if (
            USE_AUX
            and aux_loss_weight > 0.0
            and aux_target_lookup is not None
            and not residual_mode
            and meta_info
            and "aux_preds" in meta_info
        ):
            aux_preds = meta_info["aux_preds"]
            n_traits = aux_preds.shape[1]
            aux_tgt = torch.full(
                (aux_preds.shape[0], n_traits),
                float("nan"),
                device=aux_preds.device,
                dtype=aux_preds.dtype,
            )
            for i, sk in enumerate(sample_keys):
                vec = aux_target_lookup.get(str(sk))
                if vec is not None:
                    aux_tgt[i] = torch.as_tensor(vec, device=aux_preds.device, dtype=aux_preds.dtype)
            aux_mask = torch.isfinite(aux_tgt)
            if bool(aux_mask.any()):
                sq_err = (aux_preds - torch.nan_to_num(aux_tgt, nan=0.0)) ** 2
                sq_err = sq_err * aux_mask.to(sq_err.dtype)
                per_trait_count = aux_mask.to(sq_err.dtype).sum(dim=0)
                valid = per_trait_count > 0
                if bool(valid.any()):
                    per_trait_mse = sq_err.sum(dim=0)[valid] / per_trait_count[valid]
                    aux_loss = per_trait_mse.mean()
                    main_loss = main_loss + aux_loss_weight * aux_loss
        l1_loss = torch.tensor(0.0, device=device)
        if l1_weight > 0:
            for p in params_list:
                l1_loss = l1_loss + p.abs().sum()
        total_reg_loss = main_loss + (l1_weight * l1_loss / n_params)
        if adv_criterion is not None and env_logits is not None and env_ids_for_loss is not None:
            adv_loss = adv_criterion(env_logits, env_ids_for_loss.long())
            total_reg_loss = total_reg_loss + ENV_ADVERSARY_WEIGHT * adv_loss
        attn_penalty = meta_info.get("attention_diversity")
        interaction_penalty = meta_info.get("interaction_reg")
        if attn_penalty is not None:
            total_reg_loss = total_reg_loss + attn_penalty
        if interaction_penalty is not None:
            total_reg_loss = total_reg_loss + interaction_penalty
        total_reg_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += float(main_loss.detach().item())
        total_reg += float(l1_loss.detach().item()) if l1_weight > 0 else 0.0
        preds_all.extend(out.detach().cpu().numpy().tolist())
        if residual_mode:
            t_all.extend(residual_target.detach().cpu().numpy().tolist())
        else:
            t_all.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    avg_reg = total_reg / max(1, len(loader))
    if len(preds_all) > 1:
        t_arr = np.array(t_all, dtype=float)
        p_arr = np.array(preds_all, dtype=float)
        t_arr, p_arr = _filter_finite_pairs(t_arr, p_arr)
        r2 = r2_score(t_arr, p_arr) if len(p_arr) > 1 else float("nan")
    else:
        r2 = float("nan")
    logging.info(f"  [Regularized] Loss={avg_loss:.4f} L1={avg_reg:.4f} R2={r2:.4f}")
    return avg_loss, r2

def evaluate(
    model,
    loader,
    criterion,
    device,
    baseline_lookup: Optional[Dict[str, float]] = None,
    env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    return_preds: bool = False,
    gxe_stage: int = 0,
    residual_mode: bool = False
):
    model.eval()
    if residual_mode:
        baseline_lookup = None
        env_target_stats = None
    total = 0.0
    preds_all, t_all = [], []
    baseline_vals = []
    env_ids_all: List[int] = []
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    with torch.no_grad():
        for batch in loader:
            sample_weights: Optional[torch.Tensor] = None
            if len(batch) == 9:
                genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                genomic = genomic.to(device); masks = masks.to(device)
                row_labels = row_labels.to(device)
                env_ts = env_ts.to(device); loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device); pop_ids = pop_ids.to(device)
                targets = targets.to(device)
                env_ids = _compute_env_ids(loc_ids, year_ids)
                sample_weights = _compute_sample_weights(env_ids, pop_ids)
                if env_target_stats is not None:
                    env_ids_all.extend(env_ids.detach().cpu().tolist())
                if use_chr_pool:
                    genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)
                stage_kwargs = _stage_kwargs(model, gxe_stage, return_components=residual_mode)
                dosage_override = _build_dosage_override_batch(sids, device)
                if dosage_override is not None:
                    stage_kwargs["dosage_override"] = dosage_override
                out_raw = _forward_maybe_row_labels(
                    model,
                    (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
                    stage_kwargs,
                    row_labels
                )

            elif len(batch) == 7:
                geno_vec, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                geno_vec = geno_vec.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                targets = targets.to(device)
                env_ids = _compute_env_ids(loc_ids, year_ids)
                sample_weights = _compute_sample_weights(env_ids, pop_ids)
                if env_target_stats is not None:
                    env_ids_all.extend(env_ids.detach().cpu().tolist())
                stage_kwargs = _stage_kwargs(model, gxe_stage, return_components=residual_mode)
                out_raw = model(geno_vec, env_ts, loc_ids, year_ids, pop_ids, **stage_kwargs)

            else:
                raise RuntimeError(f"Unexpected batch length {len(batch)} in evaluate")

            meta_info: Dict[str, torch.Tensor] = {}
            if isinstance(out_raw, (tuple, list)):
                out = out_raw[0]
                if len(out_raw) > 1 and isinstance(out_raw[1], dict):
                    meta_info = out_raw[1]
            else:
                out = out_raw
            if out.dim() > 1:
                out = out.squeeze(-1)

            if residual_mode:
                main_out = meta_info.get("main_out")
                if main_out is None:
                    raise RuntimeError("Residual-mode eval requires return_components=True with main_out.")
                residual_target = targets - main_out.detach()
                loss_raw = criterion(out, residual_target)
            else:
                loss_raw = criterion(out, targets)
            loss = _reduce_loss(loss_raw, sample_weights)
            total += float(loss.detach().item())
            preds_all.extend(out.detach().cpu().numpy().tolist())
            if residual_mode:
                t_all.extend(residual_target.detach().cpu().numpy().tolist())
            else:
                t_all.extend(targets.detach().cpu().numpy().tolist())
            if baseline_lookup is not None:
                baseline_vals.extend([baseline_lookup.get(s, 0.0) for s in sids])
    preds_all = np.array(preds_all, dtype=float)
    t_all = np.array(t_all, dtype=float)
    if residual_mode:
        preds_metrics = preds_all
        t_metrics = t_all
    else:
        preds_metrics = destandardize_targets(preds_all)
        t_metrics = destandardize_targets(t_all)
        if env_target_stats is not None and env_ids_all:
            preds_metrics = _apply_env_unscale(preds_metrics, env_ids_all, env_target_stats)
            t_metrics = _apply_env_unscale(t_metrics, env_ids_all, env_target_stats)
        if baseline_lookup is not None and baseline_vals:
            base_arr = np.array(baseline_vals, dtype=float)
            preds_metrics = preds_metrics + base_arr
            t_metrics = t_metrics + base_arr
    t_metrics, preds_metrics = _filter_finite_pairs(t_metrics, preds_metrics)
    avg = total / max(1, len(loader))
    if len(preds_metrics) > 1:
        r2 = r2_score(t_metrics, preds_metrics)
        rmse = math.sqrt(mean_squared_error(t_metrics, preds_metrics))
        mae = mean_absolute_error(t_metrics, preds_metrics)
        ccc = concordance_correlation_coefficient(t_metrics, preds_metrics)
    else:
        r2 = rmse = mae = ccc = float("nan")
    if return_preds:
        return avg, r2, rmse, mae, ccc, preds_metrics, t_metrics
    logging.info(f"CCC: {ccc:.4f}")
    return avg, r2, rmse, mae, ccc


def _aggregate_snapshot_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snapshot_paths: List[str],
    baseline_lookup: Optional[Dict[str, float]] = None,
    env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    gxe_stage: int = 0,
    residual_mode: bool = False
) -> Tuple[float, float, float, float, float]:
    if not snapshot_paths:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    original_state = copy.deepcopy(model.state_dict())
    preds_sum = None
    targets_ref = None
    for path in snapshot_paths:
        if not load_checkpoint_safely(model, path, device, allow_shape_mismatch=False):
            continue
        _, _, _, _, _, preds_metrics, t_metrics = evaluate(
            model,
            loader,
            criterion,
            device,
            baseline_lookup=baseline_lookup,
            env_target_stats=env_target_stats,
            gxe_stage=gxe_stage,
            residual_mode=residual_mode,
            return_preds=True
        )
        preds_metrics = np.array(preds_metrics, dtype=float)
        t_metrics = np.array(t_metrics, dtype=float)
        if preds_sum is None:
            preds_sum = preds_metrics
            targets_ref = t_metrics
        else:
            preds_sum += preds_metrics

    model.load_state_dict(original_state)
    if preds_sum is None or targets_ref is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    ensemble_preds = preds_sum / len(snapshot_paths)
    targets_ref, ensemble_preds = _filter_finite_pairs(targets_ref, ensemble_preds)
    if len(ensemble_preds) > 1:
        r2 = r2_score(targets_ref, ensemble_preds)
        rmse = math.sqrt(mean_squared_error(targets_ref, ensemble_preds))
        mae = mean_absolute_error(targets_ref, ensemble_preds)
        ccc = concordance_correlation_coefficient(targets_ref, ensemble_preds)
    else:
        r2 = rmse = mae = ccc = float("nan")
    return float("nan"), r2, rmse, mae, ccc


def train_and_eval_once(
    model,
    train_eval_loader,
    train_loader,
    val_loader,
    test_loader,
    device,
    optimizer,
    scheduler,
    criterion,
    max_epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    baseline_lookup: Optional[Dict[str, float]] = None,
    env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    snapshot_cycle_length: int = 0,
    snapshot_prefix: str = "snapshot",
    monitor_test: bool = False,
    gxe_stage: int = 0,
    residual_mode: bool = False,
    aux_target_lookup: Optional[Dict[str, List[float]]] = None,
    aux_loss_weight: float = 0.0
):
    """
    Runs a train/val loop with early stopping, returns dict of val/test metrics.
    Leaves caller responsible for model construction; uses a temp 'best_model_tmp.pt'.
    """
    best_val_r2 = float("-inf")
    epochs_since_improve = 0
    snapshot_paths = []
    if snapshot_cycle_length > 0:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    # Remove stale temp checkpoint to avoid loading old shapes if no new best is saved
    if os.path.exists("best_model_tmp.pt"):
        try:
            os.remove("best_model_tmp.pt")
        except Exception:
            pass

    for epoch in range(1, max_epochs + 1):
        adv_alpha = 0.0
        if USE_ENV_ADVERSARY:
            adv_alpha = min(ENV_ADVERSARY_MAX_ALPHA, epoch / float(max(1, ENV_ADVERSARY_WARMUP_EPOCHS)))
        tr_loss, tr_r2 = train_epoch_regularized(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            l1_weight=0.0,
            mixup_alpha=MIXUP_ALPHA,
            adv_alpha=adv_alpha,
            aux_dosage_w=0.05,
            aux_complex_w=0.05,
            aux_target_lookup=aux_target_lookup,
            aux_loss_weight=aux_loss_weight,
        )

        tr_eval_loss, tr_eval_r2, tr_eval_rmse, tr_eval_mae, tr_eval_ccc = evaluate(
            model, train_eval_loader, criterion, device,
            baseline_lookup=baseline_lookup if not USE_CV else None,
            env_target_stats=env_target_stats if (env_target_stats is not None and not USE_CV) else None
        )

        va_loss, va_r2, va_rmse, va_mae, va_ccc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            baseline_lookup=baseline_lookup,
            env_target_stats=env_target_stats,
            gxe_stage=gxe_stage,
            residual_mode=residual_mode
        )

        if monitor_test:
            te_loss, te_r2, te_rmse, te_mae, te_ccc = evaluate(
                model,
                test_loader,
                criterion,
                device,
                baseline_lookup=baseline_lookup,
                env_target_stats=env_target_stats,
                gxe_stage=gxe_stage,
                residual_mode=residual_mode
            )
            logging.info(
                f"Epoch {epoch:03d}/{max_epochs} | Train Loss {tr_loss:.4f} R2 {tr_r2:.4f} | "
                f"TrainEval R2 {tr_eval_r2:.4f} RMSE {tr_eval_rmse:.4f} MAE {tr_eval_mae:.4f} CCC {tr_eval_ccc:.4f}"
                f"| Val Loss {va_loss:.4f} R2 {va_r2:.4f} RMSE {va_rmse:.4f} MAE {va_mae:.4f} CCC {va_ccc:.4f}"
                f"| Test R2 {te_r2:.4f} RMSE {te_rmse:.4f} MAE {te_mae:.4f} CCC {te_ccc:.4f}"
            )
            if va_r2 > best_val_r2 + early_stop_min_delta:
                best_val_r2 = va_r2
                torch.save(model.state_dict(), "best_model_tmp.pt")
                logging.info(f" ✓ Best model saved (Val R² = {va_r2:.4f})")
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
        else:
            logging.info(
                f"Epoch {epoch:03d}/{max_epochs} | Train Loss {tr_loss:.4f} R2 {tr_r2:.4f} "
                f"| Val Loss {va_loss:.4f} R2 {va_r2:.4f} RMSE {va_rmse:.4f} MAE {va_mae:.4f} CCC {va_ccc:.4f}"
            )
            if va_r2 > best_val_r2 + early_stop_min_delta:
                best_val_r2 = va_r2
                torch.save(model.state_dict(), "best_model_tmp.pt")
                logging.info(f" ✓ New best model saved (Val R² = {va_r2:.4f})")
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        step_scheduler(scheduler, epoch - 1)
        if epochs_since_improve >= early_stop_patience:
            metric_name = "test R2" if monitor_test else "val R2"
            logging.info(f"Early stopping triggered (no {metric_name} improvement in {early_stop_patience} epochs).")
            break
        if snapshot_cycle_length > 0 and epoch % snapshot_cycle_length == 0:
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"{snapshot_prefix}_epoch{epoch}.pt")
            torch.save(model.state_dict(), snapshot_path)
            snapshot_paths.append(snapshot_path)

    # Load best for eval
    if os.path.exists("best_model_tmp.pt"):
        load_checkpoint_safely(model, "best_model_tmp.pt", device, allow_shape_mismatch=False)
    va_loss, va_r2, va_rmse, va_mae, va_ccc = evaluate(
        model,
        val_loader,
        criterion,
        device,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        gxe_stage=gxe_stage,
        residual_mode=residual_mode
    )
    te_loss, te_r2, te_rmse, te_mae, te_ccc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        gxe_stage=gxe_stage,
        residual_mode=residual_mode
    )
    snapshot_metrics = {}
    if snapshot_paths:
        snapshot_metrics["val"] = _aggregate_snapshot_metrics(
            model,
            val_loader,
            criterion,
            device,
            snapshot_paths,
            baseline_lookup=baseline_lookup,
            env_target_stats=env_target_stats,
            gxe_stage=gxe_stage,
            residual_mode=residual_mode
        )
        snapshot_metrics["test"] = _aggregate_snapshot_metrics(
            model,
            test_loader,
            criterion,
            device,
            snapshot_paths,
            baseline_lookup=baseline_lookup,
            env_target_stats=env_target_stats,
            gxe_stage=gxe_stage,
            residual_mode=residual_mode
        )
        load_checkpoint_safely(model, "best_model_tmp.pt", device, allow_shape_mismatch=False)
    result = {
        "val": (va_loss, va_r2, va_rmse, va_mae, va_ccc),
        "test": (te_loss, te_r2, te_rmse, te_mae, te_ccc),
        "snapshots": snapshot_metrics,
        "snapshot_paths": snapshot_paths
    }
    return result


def train_and_eval_two_stage(
    model,
    train_eval_loader,
    train_loader,
    val_loader,
    test_loader,
    device,
    criterion,
    baseline_lookup: Optional[Dict[str, float]] = None,
    env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    snapshot_cycle_length: int = 0,
    snapshot_prefix: str = "stage",
    monitor_test: bool = False,
    aux_target_lookup: Optional[Dict[str, List[float]]] = None,
    aux_loss_weight: float = 0.0
):
    """
    Two-stage training:
      Stage 1: main effects only.
      Stage 2: GxE residuals only (targets = y - y_main).
    Returns dict with stage metrics and final combined metrics.
    """
    if not hasattr(model, "set_gxe_stage"):
        raise RuntimeError("Model does not support staged GxE training.")

    # Stage 1: main effects
    model.set_gxe_stage(1)
    optimizer = create_gxe_optimizer(
        model,
        lr=LEARNING_RATE,
        weight_decay=MAIN_WEIGHT_DECAY,
        pop_weight_decay=POP_EMBED_WEIGHT_DECAY,
        metadata_weight_decay=METADATA_WEIGHT_DECAY
    )
    scheduler = build_scheduler(optimizer, max_epochs=NUM_EPOCHS)
    stage1_metrics = train_and_eval_once(
        model,
        train_eval_loader,
        train_loader,
        val_loader,
        test_loader,
        device,
        optimizer,
        scheduler,
        criterion,
        max_epochs=NUM_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        snapshot_cycle_length=snapshot_cycle_length,
        snapshot_prefix=f"{snapshot_prefix}_stage1",
        monitor_test=monitor_test,
        gxe_stage=1,
        residual_mode=False,
        aux_target_lookup=aux_target_lookup,
        aux_loss_weight=aux_loss_weight
    )
    torch.save(model.state_dict(), "best_model_stage1.pt")

    # Stage 2: GxE residuals
    model.set_gxe_stage(2)
    stage2_lr = STAGE2_LEARNING_RATE if STAGE2_LEARNING_RATE > 0 else LEARNING_RATE * 0.5
    optimizer = create_gxe_optimizer(
        model,
        lr=stage2_lr,
        weight_decay=GXE_WEIGHT_DECAY,
        pop_weight_decay=POP_EMBED_WEIGHT_DECAY,
        metadata_weight_decay=METADATA_WEIGHT_DECAY
    )
    scheduler = build_scheduler(optimizer, max_epochs=NUM_EPOCHS)
    stage2_metrics = train_and_eval_once(
        model,
        train_eval_loader,
        train_loader,
        val_loader,
        test_loader,
        device,
        optimizer,
        scheduler,
        criterion,
        max_epochs=NUM_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        snapshot_cycle_length=snapshot_cycle_length,
        snapshot_prefix=f"{snapshot_prefix}_stage2",
        monitor_test=monitor_test,
        gxe_stage=2,
        residual_mode=True
    )

    # Final evaluation with combined prediction
    model.set_gxe_stage(0)
    va_loss, va_r2, va_rmse, va_mae, va_ccc = evaluate(
        model,
        val_loader,
        criterion,
        device,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        gxe_stage=0,
        residual_mode=False
    )
    te_loss, te_r2, te_rmse, te_mae, te_ccc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        baseline_lookup=baseline_lookup,
        env_target_stats=env_target_stats,
        gxe_stage=0,
        residual_mode=False
    )

    return {
        "val": (va_loss, va_r2, va_rmse, va_mae, va_ccc),
        "test": (te_loss, te_r2, te_rmse, te_mae, te_ccc),
        "stage1": stage1_metrics,
        "stage2": stage2_metrics
    }


def _select_snapshot_metrics(metrics: Dict[str, Any], stage: str) -> Tuple[float, float, float, float, float]:
    """Prefer snapshot-based metrics when available; fall back to best-model stats."""
    snapshot = metrics.get("snapshots", {}).get(stage)
    if snapshot and not math.isnan(snapshot[1]):
        return snapshot
    return metrics.get(stage, (float("nan"),) * 5)

def predict_with_ids(
    model,
    loader,
    device,
    baseline_lookup: Optional[Dict[str, float]] = None,
    env_target_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    apply_baseline: bool = True
):
    """
    Runs model on a loader and returns list of dicts with sample ids, preds, targets.
    """
    model.eval()
    out_rows = []
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 9:
                genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                genomic = genomic.to(device)
                masks = masks.to(device)
                row_labels = row_labels.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                if use_chr_pool:
                    genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)
                forward_kwargs: Dict[str, Any] = {}
                dosage_override = _build_dosage_override_batch(sids, device)
                if dosage_override is not None:
                    forward_kwargs["dosage_override"] = dosage_override
                out_raw = _forward_maybe_row_labels(
                    model,
                    (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
                    forward_kwargs,
                    row_labels
                )
                out = out_raw[0] if isinstance(out_raw, (tuple, list)) else out_raw
                out = out.squeeze(-1).cpu().numpy()
                t = targets.numpy()

            elif len(batch) == 7:
                geno_vec, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                geno_vec = geno_vec.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                out_raw = model(geno_vec, env_ts, loc_ids, year_ids, pop_ids)
                out = out_raw[0] if isinstance(out_raw, (tuple, list)) else out_raw
                out = out.squeeze(-1).cpu().numpy()
                t = targets.numpy()

            else:
                raise RuntimeError(f"Unexpected batch length {len(batch)} in predict_with_ids")

            out = np.atleast_1d(destandardize_targets(out)).astype(float)
            t = np.atleast_1d(destandardize_targets(t)).astype(float)
            if env_target_stats is not None and len(batch) == 9:
                env_ids = _compute_env_ids(loc_ids, year_ids).detach().cpu().tolist()
                out = np.atleast_1d(_apply_env_unscale(out, env_ids, env_target_stats)).astype(float)
                t = np.atleast_1d(_apply_env_unscale(t, env_ids, env_target_stats)).astype(float)
            if out.shape[0] != len(sids):
                logging.warning("predict_with_ids: length mismatch (preds=%d, sids=%d); skipping batch.", out.shape[0], len(sids))
                continue
            for sid, pred, true in zip(sids, out.tolist(), t.tolist()):
                base = baseline_lookup.get(sid, 0.0) if baseline_lookup else 0.0
                if apply_baseline:
                    pred = pred + base
                    true = true + base
                out_rows.append({"SampleID": sid, "pred": float(pred), "true": float(true)})
        return out_rows


def analyze_env_performance(preds, meta_df, key_col):
    """
    Break down RÂ² by environment (Location_Code Ãƒâ€” Year_Code) for a set of predictions.
    """
    results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"true": [], "pred": []})
    if not preds:
        return
    key_series = meta_df[key_col].astype(str)
    for row in preds:
        key = str(row.get("SampleID"))
        match = meta_df.loc[key_series == key]
        if match.empty:
            continue
        meta_row = match.iloc[0]
        env = f"Loc{meta_row.get('Location_Code', 'NA')}_Year{meta_row.get('Year_Code', 'NA')}"
        results[env]["true"].append(row.get("true", 0.0))
        results[env]["pred"].append(row.get("pred", 0.0))

    if not results:
        return
    print("\nEnvironment-wise RÂ²:")
    for env, data in sorted(results.items()):
        if len(data["true"]) >= 5:
            r2 = r2_score(data["true"], data["pred"])
            print(f"  {env}: RÂ²={r2:.4f} (n={len(data['true'])})")


def export_penultimate_embeddings(
    model,
    loader,
    device,
    path: str = "penultimate_embeddings.csv",
    view: str = "fused"
):
    """
    Exports the concatenated representation before the final head for each sample in loader.
    """
    model.eval()
    rows = []
    warned_temporal = False
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    view = str(view).lower().strip()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 9:
                genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                if view == "fused":
                    encode_fn = getattr(model, "encode_combined", None)
                elif view == "genomic":
                    encode_fn = getattr(model, "encode_genomic_only", None)
                elif view == "pop":
                    encode_fn = getattr(model, "encode_population_only", None)
                elif view in ("loc", "location"):
                    encode_fn = getattr(model, "encode_location_only", None)
                elif view == "year":
                    encode_fn = getattr(model, "encode_year_only", None)
                else:
                    encode_fn = None
                if encode_fn is None:
                    if not warned_temporal:
                        logging.warning("Skipping %s embedding export (model missing encoder).", view)
                        warned_temporal = True
                    continue
                genomic = genomic.to(device)
                masks = masks.to(device)
                row_labels = row_labels.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                if use_chr_pool:
                    genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)
                if view == "fused":
                    combined = _forward_maybe_row_labels(
                        encode_fn,
                        (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
                        {},
                        row_labels
                    )
                elif view == "genomic":
                    combined = _forward_maybe_row_labels(
                        encode_fn,
                        (genomic, masks),
                        {},
                        row_labels
                    )
                elif view == "pop":
                    combined = encode_fn(pop_ids)
                elif view in ("loc", "location"):
                    combined = encode_fn(loc_ids)
                elif view == "year":
                    combined = encode_fn(year_ids)
                else:
                    continue
                combined_np = combined.cpu().numpy()
                targets_np = destandardize_targets(targets.numpy())
                for sid, vec, t in zip(sids, combined_np, targets_np):
                    row = {"SampleID": sid, "target": float(t)}
                    for i, v in enumerate(vec):
                        row[f"repr_{i}"] = float(v)
                    rows.append(row)

            elif len(batch) == 7:
                geno_vec, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                if view == "fused":
                    encode_fn = getattr(model, "encode_combined", None)
                elif view == "genomic":
                    encode_fn = getattr(model, "encode_genomic_only", None)
                elif view == "pop":
                    encode_fn = getattr(model, "encode_population_only", None)
                elif view in ("loc", "location"):
                    encode_fn = getattr(model, "encode_location_only", None)
                elif view == "year":
                    encode_fn = getattr(model, "encode_year_only", None)
                else:
                    encode_fn = None
                if encode_fn is None:
                    if not warned_temporal:
                        logging.warning("Skipping %s embedding export (model missing encoder for raw-genotype batch).", view)
                        warned_temporal = True
                    continue
                geno_vec = geno_vec.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                if view == "fused":
                    combined = encode_fn(geno_vec, env_ts, loc_ids, year_ids, pop_ids)
                elif view == "genomic":
                    combined = encode_fn(geno_vec)
                elif view == "pop":
                    combined = encode_fn(pop_ids)
                elif view in ("loc", "location"):
                    combined = encode_fn(loc_ids)
                elif view == "year":
                    combined = encode_fn(year_ids)
                else:
                    continue
                combined_np = combined.cpu().numpy()
                targets_np = destandardize_targets(targets.numpy())
                for sid, vec, t in zip(sids, combined_np, targets_np):
                    row = {"SampleID": sid, "target": float(t)}
                    for i, v in enumerate(vec):
                        row[f"repr_{i}"] = float(v)
                    rows.append(row)

            else:
                raise RuntimeError(f"Unexpected batch length {len(batch)} in export_penultimate_embeddings")
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        logging.info(f"Exported {view} embeddings to {path}")


def collect_embeddings(model, loader, device, view: str = "fused"):
    """
    Returns (embeddings [N,D], targets [N], sample_ids list) without writing to disk.
    """
    model.eval()
    vecs = []
    targets_all = []
    ids = []
    warned_temporal = False
    tensor_model = _is_tensor_model(model)
    use_chr_pool = tensor_model and USE_CHR_POOLING and not getattr(model, "uses_habe", False)
    view = str(view).lower().strip()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 9:
                genomic, masks, row_labels, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                if view == "fused":
                    encode_fn = getattr(model, "encode_combined", None)
                elif view == "genomic":
                    encode_fn = getattr(model, "encode_genomic_only", None)
                elif view == "pop":
                    encode_fn = getattr(model, "encode_population_only", None)
                elif view in ("loc", "location"):
                    encode_fn = getattr(model, "encode_location_only", None)
                elif view == "year":
                    encode_fn = getattr(model, "encode_year_only", None)
                else:
                    encode_fn = None
                if encode_fn is None:
                    if not warned_temporal:
                        logging.warning("Skipping %s embedding collection (model missing encoder).", view)
                        warned_temporal = True
                    continue
                genomic = genomic.to(device)
                masks = masks.to(device)
                row_labels = row_labels.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                if use_chr_pool:
                    genomic, masks = pool_within_chromosomes(genomic, masks, window_size=CHR_POOL_WINDOW)
                if view == "fused":
                    combined = _forward_maybe_row_labels(
                        encode_fn,
                        (genomic, masks, env_ts, loc_ids, year_ids, pop_ids),
                        {},
                        row_labels
                    )
                elif view == "genomic":
                    combined = _forward_maybe_row_labels(
                        encode_fn,
                        (genomic, masks),
                        {},
                        row_labels
                    )
                elif view == "pop":
                    combined = encode_fn(pop_ids)
                elif view in ("loc", "location"):
                    combined = encode_fn(loc_ids)
                elif view == "year":
                    combined = encode_fn(year_ids)
                else:
                    continue
                vecs.append(combined.cpu().numpy())
                targets_all.append(destandardize_targets(targets.numpy()))
                ids.extend(sids)

            elif len(batch) == 7:
                geno_vec, env_ts, loc_ids, year_ids, pop_ids, targets, sids = batch
                if view == "fused":
                    encode_fn = getattr(model, "encode_combined", None)
                elif view == "genomic":
                    encode_fn = getattr(model, "encode_genomic_only", None)
                elif view == "pop":
                    encode_fn = getattr(model, "encode_population_only", None)
                elif view in ("loc", "location"):
                    encode_fn = getattr(model, "encode_location_only", None)
                elif view == "year":
                    encode_fn = getattr(model, "encode_year_only", None)
                else:
                    encode_fn = None
                if encode_fn is None:
                    if not warned_temporal:
                        logging.warning("Skipping %s embedding collection (model missing encoder for raw-genotype batch).", view)
                        warned_temporal = True
                    continue
                geno_vec = geno_vec.to(device)
                env_ts = env_ts.to(device)
                loc_ids = loc_ids.to(device)
                year_ids = year_ids.to(device)
                pop_ids = pop_ids.to(device)
                if view == "fused":
                    combined = encode_fn(geno_vec, env_ts, loc_ids, year_ids, pop_ids)
                elif view == "genomic":
                    combined = encode_fn(geno_vec)
                elif view == "pop":
                    combined = encode_fn(pop_ids)
                elif view in ("loc", "location"):
                    combined = encode_fn(loc_ids)
                elif view == "year":
                    combined = encode_fn(year_ids)
                else:
                    continue
                vecs.append(combined.cpu().numpy())
                targets_all.append(destandardize_targets(targets.numpy()))
                ids.extend(sids)
            else:
                logging.warning("Skipping batch in collect_embeddings: unexpected length %d", len(batch))
                continue
    if not vecs:
        return np.empty((0, 0)), np.empty((0,)), []
    vecs = np.concatenate(vecs, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return vecs, targets_all, ids


def plot_embeddings_tsne(emb_path: str, meta_df: pd.DataFrame, out_prefix: str = "emb_tsne"):
    try:
        emb = pd.read_csv(emb_path)
    except Exception as e:
        logging.warning(f"Could not read embeddings at {emb_path}: {e}")
        return
    repr_cols = [c for c in emb.columns if c.startswith("repr_")]
    if len(repr_cols) == 0 or len(emb) < 2:
        logging.warning("Not enough embeddings to plot.")
        return

    # Build robust join key to handle SampleID vs SplitKey
    emb = emb.copy()
    emb["JoinKey"] = emb.get("SampleID", emb.iloc[:, 0]).astype(str)
    meta_copy = meta_df.copy()
    meta_copy["SampleID_str"] = meta_copy.get("SampleID", meta_copy.iloc[:, 0]).astype(str)
    if "SplitKey" in meta_copy.columns:
        meta_copy["SplitKey_str"] = meta_copy["SplitKey"].astype(str)
    lookup = {}
    for _, row in meta_copy.iterrows():
        loc = row.get("Location_Code", np.nan)
        yr = row.get("Year_Code", np.nan)
        pop = row.get("Pop_Code", np.nan)
        lookup[row["SampleID_str"]] = (loc, yr, pop)
        if "SplitKey_str" in row:
            lookup[row["SplitKey_str"]] = (loc, yr, pop)

    emb["Location_Code"] = emb["JoinKey"].map(lambda k: lookup.get(k, (np.nan, np.nan, np.nan))[0])
    emb["Year_Code"] = emb["JoinKey"].map(lambda k: lookup.get(k, (np.nan, np.nan, np.nan))[1])
    emb["Pop_Code"] = emb["JoinKey"].map(lambda k: lookup.get(k, (np.nan, np.nan, np.nan))[2])
    merged = emb
    X = merged[repr_cols].values
    perplexity = max(5, min(30, len(merged) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=SEED)
    pts = tsne.fit_transform(X)

    def _scatter_cont(color_vals, title, fname):
        plt.figure(figsize=(5, 4))
        sc = plt.scatter(pts[:, 0], pts[:, 1], c=color_vals, cmap="viridis", alpha=0.8, s=10)
        plt.title(title)
        plt.xticks([]); plt.yticks([])
        plt.colorbar(sc, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        logging.info(f"Saved {fname}")

    def _scatter_cat(cat_vals, title, fname):
        vals = pd.Series(cat_vals).fillna("NA").astype(str)
        cats = vals.unique().tolist()
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(cats))))
        plt.figure(figsize=(6, 5))
        for c, col in zip(cats, colors):
            mask = vals == c
            plt.scatter(pts[mask, 0], pts[mask, 1], label=c, color=col, alpha=0.8, s=12)
        plt.title(title)
        plt.xticks([]); plt.yticks([])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        logging.info(f"Saved {fname}")

    _scatter_cont(merged["target"].values, "t-SNE colored by target", f"{out_prefix}_target.png")
    _scatter_cat(merged["Location_Code"], "t-SNE colored by Location", f"{out_prefix}_loc.png")
    _scatter_cat(merged["Year_Code"], "t-SNE colored by Year", f"{out_prefix}_year.png")
    _scatter_cat(merged["Pop_Code"], "t-SNE colored by Pop", f"{out_prefix}_pop.png")


def save_latent_space_summary(df: pd.DataFrame, out_dir: str = "results/01_latent_space", tag: Optional[str] = None):
    """
    Save a 3-panel latent space summary (phenotype, genotype, environment) similar to prior CNN plots.
    Expects columns: z1, z2, Trait (continuous), Name (categorical), EnvID (categorical).
    """
    required = {"z1", "z2", "Trait", "Name", "EnvID"}
    if df is None or df.empty or not required.issubset(df.columns):
        logging.warning("Latent space summary skipped (missing columns or empty DF).")
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sc1 = axes[0].scatter(df["z1"], df["z2"], c=df["Trait"], cmap="coolwarm", s=16)
    axes[0].set_title("Colored by phenotype"); fig.colorbar(sc1, ax=axes[0])

    axes[1].scatter(df["z1"], df["z2"], c=pd.Categorical(df["Name"]).codes, cmap="tab20", s=16)
    axes[1].set_title("Colored by genotype")

    axes[2].scatter(df["z1"], df["z2"], c=pd.Categorical(df["EnvID"]).codes, cmap="Set3", s=16)
    axes[2].set_title("Colored by environment")

    for ax in axes:
        ax.set_xlabel("t-SNE1")
        ax.set_ylabel("t-SNE2")
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    plt.savefig(os.path.join(out_dir, f"latent_space_summary{suffix}.png"), dpi=300)
    plt.close()
    logging.info(f"Saved latent space summary to {out_dir}/latent_space_summary{suffix}.png")


def plot_latent_space_summary_from_embeddings(
    emb_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    out_dir: str = "results/01_latent_space",
    tag: Optional[str] = None
):
    """
    Build t-SNE from embedding dataframe and write a latent space summary plot.
    """
    repr_cols = [c for c in emb_df.columns if c.startswith("repr_")]
    if not repr_cols or len(emb_df) < 2:
        logging.warning("Latent space summary skipped: insufficient embeddings.")
        return

    df = emb_df.copy()
    df["JoinKey"] = df.get("SampleID", df.iloc[:, 0]).astype(str)
    meta_copy = meta_df.copy()
    meta_copy["SampleID_str"] = meta_copy.get("SampleID", meta_copy.iloc[:, 0]).astype(str)
    if "SplitKey" in meta_copy.columns:
        meta_copy["SplitKey_str"] = meta_copy["SplitKey"].astype(str)

    key_to_sid = {}
    key_to_env = {}
    for _, row in meta_copy.iterrows():
        sid = row["SampleID_str"]
        env_id = f"{row.get('Location_Code', 'NA')}|{row.get('Year_Code', 'NA')}"
        key_to_sid[sid] = sid
        key_to_env[sid] = env_id
        if "SplitKey_str" in meta_copy.columns:
            key_to_sid[row["SplitKey_str"]] = sid
            key_to_env[row["SplitKey_str"]] = env_id

    df["Name"] = df["JoinKey"].map(lambda k: key_to_sid.get(k, k))
    df["EnvID"] = df["JoinKey"].map(lambda k: key_to_env.get(k, "NA"))
    df["Trait"] = df.get("target", np.nan)
    X = df[repr_cols].values
    perplexity = max(5, min(30, len(df) - 1))
    pts = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=SEED).fit_transform(X)
    df["z1"] = pts[:, 0]
    df["z2"] = pts[:, 1]
    save_latent_space_summary(df[["z1", "z2", "Trait", "Name", "EnvID"]], out_dir=out_dir, tag=tag)


def plot_calibration(pred_rows, bins: int = 10, fname: str = "calibration_plot.png"):
    if not pred_rows:
        return
    y_true = np.array([r["true"] for r in pred_rows], dtype=float)
    y_pred = np.array([r["pred"] for r in pred_rows], dtype=float)
    q = np.linspace(0, 1, bins + 1)
    preds_q = np.quantile(y_pred, q)
    bins_idx = np.digitize(y_pred, preds_q[1:-1], right=True)
    bin_true = [y_true[bins_idx == i].mean() if np.any(bins_idx == i) else np.nan for i in range(bins)]
    bin_pred = [y_pred[bins_idx == i].mean() if np.any(bins_idx == i) else np.nan for i in range(bins)]
    plt.figure(figsize=(5, 5))
    plt.plot(bin_pred, bin_true, marker="o")
    lims = [np.nanmin([bin_pred, bin_true]), np.nanmax([bin_pred, bin_true])]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlabel("Predicted (bin mean)")
    plt.ylabel("Observed (bin mean)")
    plt.title("Calibration (quantile bins)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    logging.info(f"Saved calibration plot to {fname}")


def plot_pairwise_dist_vs_target(vecs: np.ndarray, targets: np.ndarray, fname: str = "pairwise_dist_vs_target_diff.png"):
    if vecs.size == 0 or targets.size == 0:
        return
    n = vecs.shape[0]
    max_n = min(500, n)
    idx = np.random.choice(n, size=max_n, replace=False) if n > max_n else np.arange(n)
    sub_vecs = vecs[idx]
    sub_targets = targets[idx]
    dists = pdist(sub_vecs, metric="euclidean")
    targ_diffs = pdist(sub_targets.reshape(-1, 1), metric="euclidean")
    plt.figure(figsize=(5, 4))
    plt.scatter(dists, targ_diffs, alpha=0.4, s=6)
    plt.xlabel("Embedding distance (euclidean)")
    plt.ylabel("Phenotype difference")
    plt.title("Embedding distance vs phenotype difference")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    logging.info(f"Saved pairwise dist plot to {fname}")


def plot_tsne_basic(vecs: np.ndarray, targets: np.ndarray, out_prefix: str):
    if vecs.size == 0 or targets.size == 0 or vecs.shape[0] < 2:
        return
    perplexity = max(5, min(30, vecs.shape[0] - 1))
    pts = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=SEED).fit_transform(vecs)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=targets, cmap="coolwarm", alpha=0.75, s=16)
    plt.colorbar(sc, label="Target")
    plt.title("t-SNE of penultimate embeddings (target color)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_target.png", dpi=300)
    plt.close()

    bins = np.digitize(targets, np.nanquantile(targets, [0.25, 0.5, 0.75]))
    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], c=bins, cmap="Set1", alpha=0.75, s=16)
    plt.title("t-SNE embeddings colored by target quartiles")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_quartiles.png", dpi=300)
    plt.close()


def _boxplot_whisker_bounds(series: pd.Series, whisker_k: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker_k * iqr
    upper = q3 + whisker_k * iqr
    return float(lower), float(upper)


def save_trait_boxplot(series: pd.Series, whisker_k: float = 1.5, path: str = "trait_boxplot.png"):
    """
    Save a simple boxplot of the target with whiskers at +/- whisker_k * IQR.
    """
    if series.empty:
        logging.warning("Boxplot skipped: empty series.")
        return
    plt.figure(figsize=(4, 6))
    try:
        plt.boxplot(series.dropna(), whis=whisker_k, tick_labels=[TARGET_COL])
    except TypeError:
        plt.boxplot(series.dropna(), whis=whisker_k, labels=[TARGET_COL])
    plt.title(f"Trait boxplot (whis={whisker_k})")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    logging.info(f"Saved trait boxplot to {path}")

def analyze_splits(meta_df, train_keys, val_keys, test_keys, key_col, id_col, loc_col, year_col):
    """Helper function to print statistics about data splits."""
    
    meta_df['is_train'] = meta_df[key_col].isin(train_keys)
    meta_df['is_val'] = meta_df[key_col].isin(val_keys)
    meta_df['is_test'] = meta_df[key_col].isin(test_keys)

    train_df = meta_df[meta_df['is_train']]
    val_df = meta_df[meta_df['is_val']]
    test_df = meta_df[meta_df['is_test']]

    print("-" * 50)
    print("Data Split Analysis")
    print("-" * 50)

    # --- Genotype (SampleID) Analysis ---
    train_sids = set(train_df[id_col])
    val_sids = set(val_df[id_col])
    test_sids = set(test_df[id_col])

    print(f"\nGenotype (SampleID) Counts:")
    print(f"  Train: {len(train_sids)} unique genotypes")
    print(f"  Val:   {len(val_sids)} unique genotypes")
    print(f"  Test:  {len(test_sids)} unique genotypes")

    # Check for genotype leakage
    leak_tr_val = train_sids.intersection(val_sids)
    leak_tr_test = train_sids.intersection(test_sids)
    leak_val_test = val_sids.intersection(test_sids)

    print("\nGenotype Leakage Check:")
    print(f"  Train Ã¢Ë†Â© Val:  {len(leak_tr_val)} overlapping genotypes")
    if len(leak_tr_val) > 0: print(f"    WARNING: Leakage detected! {list(leak_tr_val)[:5]}")
    print(f"  Train Ã¢Ë†Â© Test: {len(leak_tr_test)} overlapping genotypes")
    if len(leak_tr_test) > 0: print(f"    WARNING: Leakage detected! {list(leak_tr_test)[:5]}")
    print(f"  Val Ã¢Ë†Â© Test:   {len(leak_val_test)} overlapping genotypes")
    if len(leak_val_test) > 0: print(f"    WARNING: Leakage detected! {list(leak_val_test)[:5]}")


    # --- Environment (Location) Analysis ---
    train_locs = set(train_df[loc_col])
    val_locs = set(val_df[loc_col])
    test_locs = set(test_df[loc_col])

    print(f"\nEnvironment (Location) Counts:")
    print(f"  Train: {len(train_locs)} unique locations")
    print(f"  Val:   {len(val_locs)} unique locations")
    print(f"  Test:  {len(test_locs)} unique locations")

    # Check for new environments in test set
    new_locs_in_test = test_locs - (train_locs | val_locs)
    print("\nNew Locations in Test Set:")
    print(f"  Found {len(new_locs_in_test)} locations in Test that are NOT in Train or Val.")
    if len(new_locs_in_test) > 0:
        print(f"    INFO: These are 'unseen' environments: {list(new_locs_in_test)}")

    # --- Environment (Year) Analysis ---
    train_years = set(train_df[year_col])
    val_years = set(val_df[year_col])
    test_years = set(test_df[year_col])

    print(f"\nEnvironment (Year) Counts:")
    print(f"  Train: {len(train_years)} unique years")
    print(f"  Val:   {len(val_years)} unique years")
    print(f"  Test:  {len(test_years)} unique years")

    # Check for new environments in test set
    new_years_in_test = test_years - (train_years | val_years)
    print("\nNew Years in Test Set:")
    print(f"  Found {len(new_years_in_test)} years in Test that are NOT in Train or Val.")
    if len(new_years_in_test) > 0:
        print(f"    INFO: These are 'unseen' environments: {list(new_years_in_test)}")
    
    print("-" * 50)

"""
Additional diagnostic functions to add to your training script.
Insert these functions before your main() function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def analyze_target_distribution(meta_df, train_keys, val_keys, test_keys, key_col, target_col):
    """
    Analyze and compare target distributions across train/val/test splits.
    
    IMPORTANT: Call this function with the ACTUAL split keys, not combined keys!
    - In single-split mode: Use train_samples, val_samples, test_samples
    - In CV mode: Call inside each fold with tr_ids, va_ids, te_ids
    """
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Validate inputs
    if len(train_keys) == len(val_keys) == len(test_keys):
        print("  NOTE: All splits have the same length.")
        print("   This can be expected for Tier-2 when each genotype contributes the same number of env rows (e.g., 1/1/1).")
        print(f"   Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")
    
    train_mask = meta_df[key_col].astype(str).isin([str(k) for k in train_keys])
    val_mask = meta_df[key_col].astype(str).isin([str(k) for k in val_keys])
    test_mask = meta_df[key_col].astype(str).isin([str(k) for k in test_keys])
    
    # Check for overlaps
    train_set = set(str(k) for k in train_keys)
    val_set = set(str(k) for k in val_keys)
    test_set = set(str(k) for k in test_keys)
    
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    
    if overlap_tv or overlap_tt or overlap_vt:
        print("\n  WARNING: Overlapping keys detected!")
        print(f"   Train Ã¢Ë†Â© Val: {len(overlap_tv)} keys")
        print(f"   Train Ã¢Ë†Â© Test: {len(overlap_tt)} keys")
        print(f"   Val Ã¢Ë†Â© Test: {len(overlap_vt)} keys")
        if overlap_tv:
            print(f"   Example overlaps (Train Ã¢Ë†Â© Val): {list(overlap_tv)[:3]}")
    
    train_targets = pd.to_numeric(meta_df.loc[train_mask, target_col], errors='coerce').dropna()
    val_targets = pd.to_numeric(meta_df.loc[val_mask, target_col], errors='coerce').dropna()
    test_targets = pd.to_numeric(meta_df.loc[test_mask, target_col], errors='coerce').dropna()

    def _print_stats(name, arr):
        if len(arr) == 0:
            print(f"\n{name} Target Statistics (n=0):")
            print("  No data available.")
            return
        print(f"\n{name} Target Statistics (n={len(arr)}):")
        print(f"  Mean: {arr.mean():.4f}")
        print(f"  Std:  {arr.std():.4f}")
        print(f"  Min:  {arr.min():.4f}")
        print(f"  Max:  {arr.max():.4f}")
        print(f"  Median: {arr.median():.4f}")

    _print_stats("Train", train_targets)
    _print_stats("Val", val_targets)
    _print_stats("Test", test_targets)
    
    # Basic statistics
    print(f"\nTrain Target Statistics (n={len(train_targets)}):")
    print(f"  Mean: {train_targets.mean():.4f}")
    print(f"  Std:  {train_targets.std():.4f}")
    print(f"  Min:  {train_targets.min():.4f}")
    print(f"  Max:  {train_targets.max():.4f}")
    print(f"  Median: {train_targets.median():.4f}")
    
    print(f"\nVal Target Statistics (n={len(val_targets)}):")
    print(f"  Mean: {val_targets.mean():.4f}")
    print(f"  Std:  {val_targets.std():.4f}")
    print(f"  Min:  {val_targets.min():.4f}")
    print(f"  Max:  {val_targets.max():.4f}")
    print(f"  Median: {val_targets.median():.4f}")
    
    print(f"\nTest Target Statistics (n={len(test_targets)}):")
    print(f"  Mean: {test_targets.mean():.4f}")
    print(f"  Std:  {test_targets.std():.4f}")
    print(f"  Min:  {test_targets.min():.4f}")
    print(f"  Max:  {test_targets.max():.4f}")
    print(f"  Median: {test_targets.median():.4f}")
    
    # Check for distribution shift
    print(f"\nDistribution Shift Analysis:")
    
    # Check if test targets are within training range
    if len(test_targets) == 0:
        test_below_train = test_above_train = test_out_of_range = 0
    else:
        min_train = train_targets.min() if len(train_targets) > 0 else float("nan")
        max_train = train_targets.max() if len(train_targets) > 0 else float("nan")
        test_below_train = int((test_targets < min_train).sum()) if len(train_targets) > 0 else 0
        test_above_train = int((test_targets > max_train).sum()) if len(train_targets) > 0 else 0
        test_out_of_range = test_below_train + test_above_train
    total_test = len(test_targets)
    def _format_pct(count, total):
        if total == 0:
            return "nan%"
        return f"{100 * count / total:.1f}%"
    print(f"  Test samples below training min: {test_below_train} ({_format_pct(test_below_train, total_test)})")
    print(f"  Test samples above training max: {test_above_train} ({_format_pct(test_above_train, total_test)})")
    print(f"  Test samples out of training range: {test_out_of_range} ({_format_pct(test_out_of_range, total_test)})")
    
    if test_out_of_range > 0:
        print(f"    WARNING: {test_out_of_range} test samples are outside the training range!")
        print(f"      The model has never seen targets in this range and will likely fail.")
    
    # Statistical tests for distribution differences
    if len(train_targets) > 1 and len(val_targets) > 1:
        ks_stat_tv, ks_pval_tv = stats.ks_2samp(train_targets, val_targets)
        print(f"\nKolmogorov-Smirnov Test (tests if distributions are different):")
        print(f"  Train vs Val:  KS={ks_stat_tv:.4f}, p-value={ks_pval_tv:.4f}")
        if ks_pval_tv < 0.05:
            print(f"      Significant difference detected (p < 0.05)")
        else:
            print(f"   âœ“ Distributions are similar (p >= 0.05)")
    
    if len(train_targets) > 1 and len(test_targets) > 1:
        ks_stat_tt, ks_pval_tt = stats.ks_2samp(train_targets, test_targets)
        print(f"  Train vs Test: KS={ks_stat_tt:.4f}, p-value={ks_pval_tt:.4f}")
        if ks_pval_tt < 0.05:
            print(f"      Significant difference detected (p < 0.05)")
        else:
            print(f"   âœ“ Distributions are similar (p >= 0.05)")
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Histogram
    if len(train_targets) == 0 or len(val_targets) == 0 or len(test_targets) == 0:
        print("\n  Skipping visualizations because at least one split has no samples.")
        print("=" * 60)
        return {
            'train': train_targets,
            'val': val_targets,
            'test': test_targets,
            'test_out_of_range': test_out_of_range,
            'ks_pval_train_test': ks_pval_tt if len(train_targets) > 1 and len(test_targets) > 1 else None
        }
    plt.subplot(1, 3, 1)
    min_val = min(train_targets.min(), val_targets.min(), test_targets.min())
    max_val = max(train_targets.max(), val_targets.max(), test_targets.max())
    if min_val == max_val:
        bins = np.linspace(min_val - 0.5, max_val + 0.5, 30)
    else:
        bins = np.linspace(min_val, max_val, 30)
    plt.hist(train_targets, bins=bins, alpha=0.5, label=f'Train (n={len(train_targets)})', density=True)
    plt.hist(val_targets, bins=bins, alpha=0.5, label=f'Val (n={len(val_targets)})', density=True)
    plt.hist(test_targets, bins=bins, alpha=0.5, label=f'Test (n={len(test_targets)})', density=True)
    plt.xlabel(target_col)
    plt.ylabel('Density')
    plt.title('Target Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Box plot
    plt.subplot(1, 3, 2)
    _bp_labels = [f'Train\n(n={len(train_targets)})', f'Val\n(n={len(val_targets)})', f'Test\n(n={len(test_targets)})']
    try:
        plt.boxplot([train_targets, val_targets, test_targets], tick_labels=_bp_labels)
    except TypeError:
        plt.boxplot([train_targets, val_targets, test_targets], labels=_bp_labels)
    plt.ylabel(target_col)
    plt.title('Target Distribution (Boxplot)')
    plt.grid(alpha=0.3)
    
    # Violin plot
    plt.subplot(1, 3, 3)
    parts = plt.violinplot([train_targets, val_targets, test_targets], 
                           positions=[1, 2, 3], showmeans=True, showmedians=True)
    plt.xticks([1, 2, 3], [f'Train\n(n={len(train_targets)})', 
                           f'Val\n(n={len(val_targets)})', 
                           f'Test\n(n={len(test_targets)})'])
    plt.ylabel(target_col)
    plt.title('Target Distribution (Violin)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_distribution_analysis.png', dpi=200)
    plt.close()
    print(f"\nâœ“â€œ Saved visualization to 'target_distribution_analysis.png'")
    
    print("=" * 60)
    
    return {
        'train': train_targets,
        'val': val_targets,
        'test': test_targets,
        'test_out_of_range': test_out_of_range,
        'ks_pval_train_test': ks_pval_tt if len(train_targets) > 1 and len(test_targets) > 1 else None
    }



def main():
    global N_POPULATIONS, N_ENV_FEATURES_PER_MONTH, N_LOCATIONS, N_YEARS, NUM_ENVIRONMENTS, ENV_PAIR_TO_ID, ENV_PAIR_LUT
    global DOSAGE_OVERRIDE_ENABLED, DOSAGE_OVERRIDE_MAP, DOSAGE_OVERRIDE_DIM
    global DOSAGE_OVERRIDE_SAMPLE_TO_SID, _DOSAGE_OVERRIDE_MISSING_WARNED
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model_dropout, model_weight_decay = get_regularization_for_model("gxetensor")
    logging.info(f"Regularization for gxetensor: dropout={model_dropout} weight_decay={model_weight_decay}")

    # Load metadata
    meta = pd.read_csv(METADATA_FILE, sep=None, engine="python")

    # ========================================================================
    # NEW: Extract population structure from metadata
    # ========================================================================

    if 'Pop' in meta.columns:
        logging.info("Extracting population structure from 'Pop' column...")
        meta['Pop'] = meta['Pop'].astype(str).str.strip()
        unique_pops = sorted(meta['Pop'].dropna().unique())
        n_pops_found = len(unique_pops)
        if n_pops_found == 0:
            raise ValueError("'Pop' column exists but has no valid values!")

        pop_map = {pop: idx for idx, pop in enumerate(unique_pops)}
        meta['PopID'] = meta['Pop'].map(pop_map)

        n_missing = meta['PopID'].isna().sum()
        if n_missing > 0:
            logging.warning(f"{n_missing} samples missing population assignment - assigning to separate cluster")
            meta['PopID'] = meta['PopID'].fillna(n_pops_found).astype(int)
            N_POPULATIONS = n_pops_found + 1
        else:
            N_POPULATIONS = n_pops_found

        pop_counts = meta['PopID'].value_counts()
        logging.info(f"Population distribution:\n{pop_counts}")
        if pop_counts.min() < 5:
            logging.warning(f"Some populations have < 5 samples: {pop_counts[pop_counts < 5].to_dict()}")
    else:
        raise ValueError("'Pop' column required for population structure modeling!")

    if "SampleID" not in meta.columns:
        if "IID" in meta.columns:
            meta["SampleID"] = meta["IID"]
        elif "FID" in meta.columns:
            meta["SampleID"] = meta["FID"]
        else:
            raise KeyError("Metadata must contain 'SampleID', 'IID', or 'FID'.")

    # Environment
    env = pd.read_csv(ENVIRONMENT_FILE)

    if "SampleID" not in env.columns:
        env.rename(columns={env.columns[0]: "SampleID"}, inplace=True)
    if "Location" in env.columns:
        _normalize_location_column(env, "environment data")
    if "Year" in env.columns:
        env["Year"] = env["Year"].astype(str).str.strip()
    env_cols = [c for c in env.columns if c not in {"SampleID", "Location", "Year"}]
    env[env_cols] = env[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    overlap = set(env_cols) & {TARGET_COL}
    if overlap:
        raise RuntimeError(f"Target column(s) present in environment data: {overlap}")

    # Optional PCA on environmental covariates to capture main gradients
    if USE_ENV_PCA and len(env_cols) > 0:
        mat = env[env_cols].values
        scaler = StandardScaler()
        mat_std = scaler.fit_transform(mat)
        n_comp = min(ENV_PCA_COMPONENTS, mat_std.shape[1])
        pca = PCA(n_components=n_comp, random_state=SEED)
        pcs = pca.fit_transform(mat_std)
        pc_names = [f"EnvPC{i+1}" for i in range(n_comp)]
        for i, name in enumerate(pc_names):
            env[name] = pcs[:, i]
        env.drop(columns=env_cols, inplace=True)
        logging.info(f"Applied PCA to environment covariates (n_components={n_comp}); replaced raw env columns with PCs.")

    # Normalize metadata text columns before coding
    if "Location" in meta.columns:
        _normalize_location_column(meta, "metadata")
    if "Year" in meta.columns:
        meta["Year"] = meta["Year"].astype(str).str.strip()

    # Joint categorical encodings for Location and Year across meta + env
    if "Location" in meta.columns or "Location" in env.columns:
        loc_all = pd.concat(
            [meta.get("Location", pd.Series([], dtype=str)).astype(str),
             env.get("Location", pd.Series([], dtype=str)).astype(str)],
            ignore_index=True
        ).astype("category")
        loc_map = {cat: i for i, cat in enumerate(loc_all.cat.categories)}
        meta["Location_Code"] = meta.get("Location", "").astype(str).map(loc_map).fillna(-1).astype(int)
        env["Location_Code"] = env.get("Location", "").astype(str).map(loc_map).fillna(-1).astype(int)
    else:
        meta["Location_Code"] = 0
        env["Location_Code"] = 0
        loc_map = {}

    if "Year" in meta.columns or "Year" in env.columns:
        year_all = pd.concat(
            [meta.get("Year", pd.Series([], dtype=str)).astype(str),
             env.get("Year", pd.Series([], dtype=str)).astype(str)],
            ignore_index=True
        ).astype("category")
        year_map = {cat: i for i, cat in enumerate(year_all.cat.categories)}
        meta["Year_Code"] = meta.get("Year", "").astype(str).map(year_map).fillna(-1).astype(int)
        env["Year_Code"] = env.get("Year", "").astype(str).map(year_map).fillna(-1).astype(int)
    else:
        meta["Year_Code"] = 0
        env["Year_Code"] = 0
        year_map = {}

    # Seeding date â†’â€™ day-of-year (captures planting timing)
    if "SD" in meta.columns:
        try:
            meta["SeedingDOY"] = pd.to_datetime(meta["SD"], errors="coerce").dt.dayofyear
            meta["SeedingDOY"] = meta["SeedingDOY"].fillna(0).astype(int)
        except Exception as e:
            logging.warning(f"Failed to parse 'SeedingDate' to day-of-year: {e}")
            meta["SeedingDOY"] = 0
    else:
        logging.warning("'SeedingDate' column missing - cannot model planting timing effects")
        meta["SeedingDOY"] = 0

    # Ensure env codes exist before constructing composite key
    if "Location_Code" not in env.columns:
        env["Location_Code"] = env.get("Location", "").astype(str).map(loc_map).fillna(-1).astype(int)
    if "Year_Code" not in env.columns:
        env["Year_Code"] = env.get("Year", "").astype(str).map(year_map).fillna(-1).astype(int)

    env["EnvKey"] = (
        env["SampleID"].astype(str)
        + "|"
        + env["Location_Code"].astype(str)
        + "|"
        + env["Year_Code"].astype(str)
    )
    env_data_dict = None
    num_locations_env = None
    num_years_env = None
    if USE_TEMPORAL_ENV_ENCODING:
        logging.info("Preprocessing environmental data into temporal format...")
        cache_used = False
        env_temporal = None
        location_ids_array = None
        year_ids_array = None
        env_keys_for_temporal = None
        env_feature_names = None
        if USE_ENV_CACHE and os.path.exists(ENV_CACHE_FILE):
            try:
                cache = np.load(ENV_CACHE_FILE, allow_pickle=True)
                env_temporal = cache["env_temporal"]
                location_ids_array = cache["location_ids"]
                year_ids_array = cache["year_ids"]
                env_keys_for_temporal = cache["env_keys"].astype(str).tolist()
                env_feature_names = (
                    cache.get("env_feature_names", None).astype(str).tolist()
                    if "env_feature_names" in cache
                    else None
                )
                cached_sig = cache.get("env_feature_signature", None)
                expected_sig = _env_feature_signature()
                loc_map_json = cache.get("location_map_json", None)
                year_map_json = cache.get("year_map_json", None)
                loc_map = json.loads(str(loc_map_json)) if loc_map_json is not None else {}
                year_map = json.loads(str(year_map_json)) if year_map_json is not None else {}
                expected_dim = len(_build_env_feature_names())
                cache_mismatch = False
                if cached_sig is not None and str(cached_sig) != expected_sig:
                    logging.warning("Env cache signature mismatch; recomputing engineered features.")
                    cache_mismatch = True
                elif env_temporal is not None and env_temporal.shape[2] != expected_dim:
                    logging.warning(
                        f"Env cache feature dim mismatch (cache={env_temporal.shape[2]}, expected={expected_dim}); "
                        "recomputing engineered features."
                    )
                    cache_mismatch = True
                if cache_mismatch:
                    env_temporal = None
                    location_ids_array = None
                    year_ids_array = None
                    env_keys_for_temporal = None
                    env_feature_names = None
                else:
                    cache_used = True
                    logging.info(f"Loaded temporal env cache from {ENV_CACHE_FILE} (shape={env_temporal.shape}).")
            except Exception as e:
                logging.warning(f"Failed to load env cache {ENV_CACHE_FILE}: {e}; recomputing.")
                env_temporal = None
                location_ids_array = None
                year_ids_array = None
                env_keys_for_temporal = None
                env_feature_names = None
        # Validate presence of required columns and temporal coverage before heavy processing
        if not cache_used:
            def _has_feature(feat_name: str) -> bool:
                """Return True if the env frame has the raw column or any wide E_<feat>_XX columns."""
                if feat_name in env.columns:
                    return True
                prefix = f"E_{feat_name}_"
                return any(col.startswith(prefix) for col in env.columns)

            missing_features = {f for f in CRITICAL_ENV_FEATURES if not _has_feature(f)}
            if missing_features:
                raise ValueError(
                    f"Environment data missing critical features: {missing_features}\n"
                    f"Available columns: {list(env.columns)}"
                )
            missing_by_feature: Dict[str, List[str]] = {}
            for feat in CRITICAL_ENV_FEATURES:
                missing_cols = [f"E_{feat}_{m:02d}" for m in range(N_MONTHS) if f"E_{feat}_{m:02d}" not in env.columns]
                if missing_cols:
                    missing_by_feature[feat] = missing_cols
            if missing_by_feature:
                flat_missing = [c for cols in missing_by_feature.values() for c in cols]
                raise ValueError(
                    f"Environment data missing critical temporal columns: {flat_missing}\n"
                    f"Missing by feature: {missing_by_feature}\n"
                    f"Available columns: {list(env.columns)}"
                )
            (
                env_temporal,
                location_ids_array,
                year_ids_array,
                location_map,
                year_map,
                env_feature_names,
            ) = preprocess_environmental_data(
                env,
                critical_features=CRITICAL_ENV_FEATURES,
                strict=ENV_LOOKUP_FAIL_ON_MISSING,
                n_steps=N_MONTHS,
                engineer_features=ENV_ENGINEERED_FEATURES,
                use_stage_summaries=ENV_STAGE_SUMMARIES,
                return_feature_names=True,
            )
            env_keys_for_temporal = env["EnvKey"] if MULTI_ENV else env["SampleID"].astype(str)
            if USE_ENV_CACHE:
                try:
                    np.savez_compressed(
                        ENV_CACHE_FILE,
                        env_temporal=env_temporal,
                        location_ids=location_ids_array,
                        year_ids=year_ids_array,
                        env_keys=np.array(list(env_keys_for_temporal), dtype=object),
                        env_feature_names=np.array(env_feature_names, dtype=object),
                        env_feature_signature=_env_feature_signature(),
                        location_map_json=json.dumps(loc_map),
                        year_map_json=json.dumps(year_map)
                    )
                    logging.info(f"Saved temporal env cache to {ENV_CACHE_FILE}.")
                except Exception as e:
                    logging.warning(f"Failed to save env cache to {ENV_CACHE_FILE}: {e}")
        num_locations_env = int(location_ids_array.max()) + 1 if len(location_ids_array) > 0 else 0
        num_years_env = int(year_ids_array.max()) + 1 if len(year_ids_array) > 0 else 0
        env_keys_for_temporal = list(map(str, env_keys_for_temporal))
        env_wide_cols = [c for c in env.columns if c.startswith("E_")]
        env_wide = None
        env_wide_mean = None
        env_wide_std = None
        if env_wide_cols:
            env_wide = env[env_wide_cols].to_numpy(dtype=float)
            env_wide_mean = env_wide.mean(axis=0, keepdims=True)
            env_wide_std = env_wide.std(axis=0, keepdims=True) + 1e-8
            env_wide = (env_wide - env_wide_mean) / env_wide_std
        if env_temporal is not None:
            if USE_ENV_MATRIX_AS_MLP and env_wide is not None:
                N_ENV_FEATURES_PER_MONTH = int(env_wide.shape[1])
            else:
                N_ENV_FEATURES_PER_MONTH = int(env_temporal.shape[2])
            logging.info(f"Env feature dim (per step) -> {N_ENV_FEATURES_PER_MONTH}")

        # Align location/year ids to the same categorical coding used in metadata to avoid raw years (e.g., 2019) leaking into embeddings.
        try:
            loc_codes_env = env["Location_Code"].astype(int).to_numpy()
            year_codes_env = env["Year_Code"].astype(int).to_numpy()
            if len(loc_codes_env) == len(location_ids_array):
                location_ids_array = loc_codes_env
            else:
                logging.warning(
                    f"Env location id length mismatch (env={len(loc_codes_env)}, temporal={len(location_ids_array)}); "
                    "keeping temporal location ids."
                )
            if len(year_codes_env) == len(year_ids_array):
                year_ids_array = year_codes_env
            else:
                logging.warning(
                    f"Env year id length mismatch (env={len(year_codes_env)}, temporal={len(year_ids_array)}); "
                    "keeping temporal year ids."
                )
        except Exception as e:
            logging.warning(f"Failed to align env location/year codes to categorical indices: {e}")

        # Ensure IDs are contiguous 0..N-1
        if location_ids_array is not None and location_ids_array.size:
            uniq_loc = {val: i for i, val in enumerate(sorted(set(location_ids_array.tolist())))}
            location_ids_array = np.array([uniq_loc[v] for v in location_ids_array], dtype=np.int64)
        if year_ids_array is not None and year_ids_array.size:
            uniq_year = {val: i for i, val in enumerate(sorted(set(year_ids_array.tolist())))}
            year_ids_array = np.array([uniq_year[v] for v in year_ids_array], dtype=np.int64)

        num_locations_env = int(location_ids_array.max()) + 1 if location_ids_array is not None and len(location_ids_array) > 0 else 0
        num_years_env = int(year_ids_array.max()) + 1 if year_ids_array is not None and len(year_ids_array) > 0 else 0

        env_data_dict = {
            "temporal": env_temporal,
            "location_ids": location_ids_array,
            "year_ids": year_ids_array,
            "key_to_idx": {str(k): i for i, k in enumerate(env_keys_for_temporal)},
            "feature_names": env_feature_names,
            "env_wide": env_wide,
            "env_wide_mean": env_wide_mean,
            "env_wide_std": env_wide_std,
            "env_wide_cols": env_wide_cols,
        }
        logging.info(f"Environmental data shape: {env_temporal.shape}")
        logging.info(f"Number of locations: {num_locations_env}")
        logging.info(f"Number of years: {num_years_env}")
        # sanity: temporal preprocessing should not produce IDs beyond the learned maps
        if num_locations_env is not None:
            assert num_locations_env <= max(1, len(loc_map))
        if num_years_env is not None:
            assert num_years_env <= max(1, len(year_map))

    env_anomaly_mean = None
    if USE_ENV_ANOMALIES and env_data_dict is not None and env_data_dict.get("temporal") is not None:
        try:
            env_anomaly_mean = np.nanmean(env_data_dict["temporal"], axis=0)
            logging.info(
                "Computed env anomaly mean with shape %s for GxE shocks.",
                tuple(env_anomaly_mean.shape)
            )
        except Exception as exc:
            logging.warning("Failed to compute env anomaly mean: %s", exc)
            env_anomaly_mean = None

    env_feature_cols = [c for c in env.columns if c not in {"SampleID", "Location", "Year", "EnvKey", "Location_Code", "Year_Code"}]

    if "Pop" in meta.columns:
        meta["Pop_Code"] = meta["Pop"].astype("category").cat.codes
    else:
        meta["Pop_Code"] = 0

    # Derive categorical cardinalities from both metadata and env (including cached temporal arrays) to keep embeddings in-range.
    def _cat_size_from_series(series: pd.Series) -> int:
        if series is None or series.empty:
            return 0
        try:
            return int(series.max()) + 1
        except Exception:
            return 0

    loc_size_meta = _cat_size_from_series(meta.get("Location_Code"))
    year_size_meta = _cat_size_from_series(meta.get("Year_Code"))
    loc_size_env = _cat_size_from_series(env.get("Location_Code"))
    year_size_env = _cat_size_from_series(env.get("Year_Code"))
    if env_data_dict is not None:
        loc_ids_arr = env_data_dict.get("location_ids")
        yr_ids_arr = env_data_dict.get("year_ids")
        if loc_ids_arr is not None and len(loc_ids_arr) > 0:
            try:
                loc_size_env = max(loc_size_env, int(np.max(loc_ids_arr)) + 1)
            except Exception:
                pass
        if yr_ids_arr is not None and len(yr_ids_arr) > 0:
            try:
                year_size_env = max(year_size_env, int(np.max(yr_ids_arr)) + 1)
            except Exception:
                pass
    N_LOCATIONS = max(loc_size_meta, loc_size_env, 1)
    N_YEARS = max(year_size_meta, year_size_env, 1)
    ENV_PAIR_TO_ID = {}
    ENV_PAIR_LUT = None
    if env_data_dict is not None:
        loc_ids_arr = env_data_dict.get("location_ids")
        yr_ids_arr = env_data_dict.get("year_ids")
        if loc_ids_arr is not None and yr_ids_arr is not None and len(loc_ids_arr) == len(yr_ids_arr):
            pair_to_id, env_pair_ids, lut = _build_env_pair_map(
                np.asarray(loc_ids_arr),
                np.asarray(yr_ids_arr),
                N_LOCATIONS,
                N_YEARS
            )
            env_data_dict["env_pair_ids"] = env_pair_ids
            env_data_dict["env_pair_to_id"] = pair_to_id
            ENV_PAIR_TO_ID = pair_to_id
            ENV_PAIR_LUT = lut
            NUM_ENVIRONMENTS = max(1, len(pair_to_id))
            logging.info(
                "Inferred categorical sizes -> locations=%d, years=%d, env_pairs=%d",
                N_LOCATIONS,
                N_YEARS,
                NUM_ENVIRONMENTS
            )
        else:
            NUM_ENVIRONMENTS = max(1, N_LOCATIONS * N_YEARS)
            logging.info(
                "Inferred categorical sizes -> locations=%d, years=%d (env rows may exceed metadata coverage)",
                N_LOCATIONS,
                N_YEARS
            )
    else:
        NUM_ENVIRONMENTS = max(1, N_LOCATIONS * N_YEARS)
        logging.info(
            "Inferred categorical sizes -> locations=%d, years=%d (env rows may exceed metadata coverage)",
            N_LOCATIONS,
            N_YEARS
        )

    # Discover available samples from tensor NPZs (one per sample)
    tensor_paths = glob.glob(os.path.join(TENSOR_DIR, "*_tensor.npz"))
    tensor_paths.extend(glob.glob(os.path.join(TENSOR_DIR, "*", "*_tensor.npz")))
    # Ignore haplotype-only tensors; SNP tensors already embed block features
    tensor_paths = [
        p for p in tensor_paths
        if not os.path.basename(p).endswith("_haplo_tensor.npz")
    ]
    tensor_sids = set()
    for p in tensor_paths:
        base = os.path.basename(p).replace("_tensor.npz", "")
        tensor_sids.add(base)
        parent = os.path.basename(os.path.dirname(p))
        tensor_sids.add(parent)
    meta_sids = set(meta["SampleID"].astype(str).tolist())
    available = sorted(list(meta_sids.intersection(tensor_sids)))
    if not available:
        logging.error("No overlapping SampleID between metadata and tensor files.")
        logging.error(f"First few metadata IDs: {sorted(list(meta_sids))[:5]}")
        logging.error(f"First few tensor IDs: {sorted(list(tensor_sids))[:5]}")
        return

    meta = meta[meta["SampleID"].astype(str).isin(available)].reset_index(drop=True)

    # Optional boxplot-based trim (off by default)
    if BOXPLOT_TRIM or BOXPLOT_SAVE_PATH:
        meta[TARGET_COL] = pd.to_numeric(meta[TARGET_COL], errors="coerce")
        trait_series = meta[TARGET_COL].dropna()
        if BOXPLOT_SAVE_PATH:
            save_trait_boxplot(trait_series, whisker_k=BOXPLOT_WHISKER_K, path=BOXPLOT_SAVE_PATH)
        if BOXPLOT_TRIM and not trait_series.empty:
            lower, upper = _boxplot_whisker_bounds(trait_series, whisker_k=BOXPLOT_WHISKER_K)
            before = len(meta)
            meta = meta[(meta[TARGET_COL] >= lower) & (meta[TARGET_COL] <= upper)].reset_index(drop=True)
            removed = before - len(meta)
            logging.info(
                f"Boxplot trim (whis={BOXPLOT_WHISKER_K}) removed={removed} kept={len(meta)} of {before} "
                f"bounds=[{lower:.4f}, {upper:.4f}]"
            )
            if len(meta) == 0:
                logging.error("All rows removed by boxplot trim; consider disabling or loosening whisker.")
                return

    if MULTI_ENV:
        # Build composite key to allow multiple rows per SampleID (Location, Year); keep all rows
        meta["SplitKey"] = (
            meta["SampleID"].astype(str)
            + "|"
            + meta["Location_Code"].astype(str)
            + "|"
            + meta["Year_Code"].astype(str)
        )
        meta["SampleID_str"] = meta["SampleID"].astype(str)
        meta["EnvKey"] = meta["SplitKey"]
        # Split units: unique SampleIDs (prevents genotype reuse across splits)
        unique_ids = sorted(meta["SampleID_str"].unique().tolist())
        logging.info(f"Found {len(unique_ids)} samples with ChromoMap tiles (unique SampleIDs).")
        sample_key_to_sid = dict(zip(meta["SplitKey"], meta["SampleID_str"]))
        if USE_CV:
            # Use all environment rows for CV; no held-out test set
            train_samples = val_samples = test_samples = meta["SplitKey"].tolist()
            logging.info(f"CV mode: using all {len(unique_ids)} unique genotypes ({len(train_samples)} environment rows); no held-out test set.")
        else:
            train_sid, test_sid = train_test_split(unique_ids, test_size=0.2, random_state=SEED)
            train_sid, val_sid = train_test_split(train_sid, test_size=0.2, random_state=SEED)
            assert set(train_sid).isdisjoint(val_sid), "Train/Val overlap detected."
            assert set(train_sid).isdisjoint(test_sid), "Train/Test overlap detected."
            assert set(val_sid).isdisjoint(test_sid), "Val/Test overlap detected."
            logging.info(f"Train={len(train_sid)}, Val={len(val_sid)}, Test={len(test_sid)}")

            # Expand back to row-level keys for each split (retain all rows per SampleID)
            train_samples = meta.loc[meta["SampleID_str"].isin(train_sid), "SplitKey"].tolist()
            val_samples = meta.loc[meta["SampleID_str"].isin(val_sid), "SplitKey"].tolist()
            test_samples = meta.loc[meta["SampleID_str"].isin(test_sid), "SplitKey"].tolist()

            # Sanity: ensure no SampleID overlaps across splits (genotype leakage guard)
            set_tr, set_va, set_te = set(train_sid), set(val_sid), set(test_sid)
            overlap_tv = set_tr & set_va
            overlap_tt = set_tr & set_te
            overlap_vt = set_va & set_te
            if overlap_tv or overlap_tt or overlap_vt:
                logging.error(f"Leakage detected across splits: trainÃ¢Ë†Â©val={overlap_tv}, trainÃ¢Ë†Â©test={overlap_tt}, valÃ¢Ë†Â©test={overlap_vt}")
                return
        metadata_key_col = "SplitKey"
    else:
        # Single-environment mode: keep one row per SampleID
        meta = meta.drop_duplicates(subset=["SampleID"], keep="first").reset_index(drop=True)
        meta["SampleID_str"] = meta["SampleID"].astype(str)
        meta["EnvKey"] = (
            meta["SampleID_str"].astype(str)
            + "|"
            + meta["Location_Code"].astype(str)
            + "|"
            + meta["Year_Code"].astype(str)
        )
        sample_key_to_sid = {sid: sid for sid in meta["SampleID_str"]}
        unique_ids = sorted(meta["SampleID_str"].unique().tolist())
        logging.info(f"Found {len(unique_ids)} samples with ChromoMap tiles (unique SampleIDs).")
        if USE_CV:
            # Use all genotypes for CV; no held-out test set
            train_samples = val_samples = test_samples = unique_ids
            logging.info(f"CV mode: using all {len(unique_ids)} unique genotypes; no held-out test set.")
        else:
            train_sid, test_sid = train_test_split(unique_ids, test_size=0.2, random_state=SEED)
            train_sid, val_sid = train_test_split(train_sid, test_size=0.2, random_state=SEED)
            assert set(train_sid).isdisjoint(val_sid), "Train/Val overlap detected."
            assert set(train_sid).isdisjoint(test_sid), "Train/Test overlap detected."
            assert set(val_sid).isdisjoint(test_sid), "Val/Test overlap detected."
            logging.info(f"(Single-env) Train={len(train_sid)}, Val={len(val_sid)}, Test={len(test_sid)}")
            train_samples = train_sid
            val_samples = val_sid
            test_samples = test_sid
        metadata_key_col = "SampleID"
    sample_key_to_pop = meta.set_index(metadata_key_col)["Pop_Code"].to_dict()

    # Build the aux-trait source table (opt-in MTL). Keyed by metadata_key_col so aux
    # values can be looked up by the same sample_keys the tensor batches carry. If the aux
    # pheno file is the main phenotype CSV, pull columns straight from `meta`; otherwise
    # merge a separate file onto meta by genotype SampleID (aux values replicate per env row).
    AUX_SOURCE_DF = None
    if USE_AUX:
        if AUX_PHENO_PATH == METADATA_FILE:
            keep_cols = [metadata_key_col] + [c for c in AUX_TARGETS if c in meta.columns]
            AUX_SOURCE_DF = meta[keep_cols].copy()
            missing = [c for c in AUX_TARGETS if c not in meta.columns]
            if missing:
                logging.warning(f"[AUX] trait columns not found in main metadata: {missing}")
        else:
            aux_raw = pd.read_csv(AUX_PHENO_PATH, sep=None, engine="python")
            aux_join_col = "SampleID" if "SampleID" in aux_raw.columns else aux_raw.columns[0]
            merged = meta[[metadata_key_col, "SampleID_str"]].merge(
                aux_raw, left_on="SampleID_str", right_on=aux_join_col, how="left"
            )
            keep_cols = [metadata_key_col] + [c for c in AUX_TARGETS if c in merged.columns]
            AUX_SOURCE_DF = merged[keep_cols].copy()
            missing = [c for c in AUX_TARGETS if c not in merged.columns]
            if missing:
                logging.warning(f"[AUX] trait columns not found in aux pheno {AUX_PHENO_PATH}: {missing}")
        logging.info(
            f"[AUX] multi-trait MTL enabled: targets={AUX_TARGETS} weight={AUX_LOSS_WEIGHT} "
            f"pheno={AUX_PHENO_PATH} rows={0 if AUX_SOURCE_DF is None else len(AUX_SOURCE_DF)}"
        )
    DOSAGE_OVERRIDE_SAMPLE_TO_SID = {str(k): str(v) for k, v in sample_key_to_sid.items()}
    DOSAGE_OVERRIDE_ENABLED = False
    DOSAGE_OVERRIDE_MAP = {}
    DOSAGE_OVERRIDE_DIM = 0
    _DOSAGE_OVERRIDE_MISSING_WARNED = False

    if not USE_CV:
        assert set(train_samples).isdisjoint(val_samples), "Train/Val overlap detected."
        assert set(train_samples).isdisjoint(test_samples), "Train/Test overlap detected."
        assert set(val_samples).isdisjoint(test_samples), "Val/Test overlap detected."
        logging.info(f"Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        analyze_splits(
            meta_df=meta.copy(),
            train_keys=train_samples,
            val_keys=val_samples,
            test_keys=test_samples,
            key_col=metadata_key_col,
            id_col='SampleID_str',
            loc_col='Location_Code',
            year_col='Year_Code'
        )
    else:
        logging.info(f"Using cross-validation on the full dataset ({len(unique_ids)} unique genotypes); skipping fixed train/val/test split analysis.")


    if not USE_CV:
        # Analyze target distributions
        target_analysis = analyze_target_distribution(
            meta_df=meta.copy(),
            train_keys=train_samples,
            val_keys=val_samples,
            test_keys=test_samples,
            key_col=metadata_key_col,
            target_col=TARGET_COL
        )
    # Re-standardize environment on train split to avoid leakage (overrides cache/global stats).
    if USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP:
        try:
            env_keys_for_temporal = env["EnvKey"] if MULTI_ENV else env["SampleID"].astype(str)
            fit_mask = env_keys_for_temporal.astype(str).isin(train_samples)
            env_wide_cols = [c for c in env.columns if c.startswith("E_")]
            env_wide = None
            env_wide_mean = None
            env_wide_std = None
            if env_wide_cols:
                env_wide = env[env_wide_cols].to_numpy(dtype=float)
                env_fit_mask = fit_mask.to_numpy()
                if env_wide.shape[0] == env_fit_mask.shape[0] and env_fit_mask.any():
                    fit_slice_wide = env_wide[env_fit_mask]
                else:
                    fit_slice_wide = env_wide
                env_wide_mean = fit_slice_wide.mean(axis=0, keepdims=True)
                env_wide_std = fit_slice_wide.std(axis=0, keepdims=True) + 1e-8
                env_wide = (env_wide - env_wide_mean) / env_wide_std
            env_temporal = None
            env_feature_names = None
            env_stats = (None, None)
            location_ids_array = None
            year_ids_array = None
            if USE_TEMPORAL_ENV_ENCODING:
                (
                    env_temporal,
                    location_ids_array,
                    year_ids_array,
                    location_map,
                    year_map,
                    env_feature_names,
                    env_stats,
                ) = preprocess_environmental_data(
                    env,
                    critical_features=CRITICAL_ENV_FEATURES,
                    strict=ENV_LOOKUP_FAIL_ON_MISSING,
                    n_steps=N_MONTHS,
                    engineer_features=ENV_ENGINEERED_FEATURES,
                    use_stage_summaries=ENV_STAGE_SUMMARIES,
                    return_feature_names=True,
                    fit_mask=fit_mask.to_numpy(),
                    return_stats=True,
                )
            env_keys_for_temporal = env_keys_for_temporal.astype(str)
            key_to_idx = {str(k): i for i, k in enumerate(env_keys_for_temporal)}
            env_data_dict = {
                "temporal": env_temporal,
                "location_ids": location_ids_array,
                "year_ids": year_ids_array,
                "key_to_idx": key_to_idx,
                "env_pair_ids": None,
                "env_feature_names": np.array(env_feature_names, dtype=object) if env_feature_names is not None else None,
                "env_mean": env_stats[0],
                "env_std": env_stats[1],
                "env_wide": env_wide,
                "env_wide_mean": env_wide_mean,
                "env_wide_std": env_wide_std,
                "env_wide_cols": env_wide_cols,
            }
            num_locations_env = int(location_ids_array.max()) + 1 if location_ids_array is not None and len(location_ids_array) > 0 else 0
            num_years_env = int(year_ids_array.max()) + 1 if year_ids_array is not None and len(year_ids_array) > 0 else 0
            if USE_ENV_MATRIX_AS_MLP and env_wide is not None and env_wide.ndim == 2:
                N_ENV_FEATURES_PER_MONTH = int(env_wide.shape[1])
            elif env_temporal is not None:
                N_ENV_FEATURES_PER_MONTH = int(env_temporal.shape[2])
            logging.info(f"Env feature dim (per step) -> {N_ENV_FEATURES_PER_MONTH}")
        except Exception as e:
            logging.warning(f"Env re-standardization on train split failed; falling back to existing env tensor. Error: {e}")
    use_env_residuals = (
        USE_ENV_RESIDUAL_TRAINING
        and USE_TEMPORAL_ENV_ENCODING
        and env_data_dict is not None
    )
    if USE_RESIDUAL_FOCUS_ARCH and use_env_residuals:
        logging.warning(
            "USE_RESIDUAL_FOCUS_ARCH is enabled; disabling USE_ENV_RESIDUAL_TRAINING to avoid double residualization."
        )
        use_env_residuals = False
    # Fit env-only main effect and build residual targets (single-split path only; CV handles per-fold)
    baseline_lookup = {}
    residual_targets = {}
    nonCV_aux_target_lookup = None
    if not USE_CV and USE_AUX:
        nonCV_aux_target_lookup = _build_aux_target_lookup(
            AUX_SOURCE_DF,
            train_keys=train_samples,
            all_keys=list(dict.fromkeys(train_samples + val_samples + test_samples)),
            key_col=metadata_key_col,
            aux_cols=AUX_TARGETS,
        )
    if not USE_CV and use_env_residuals:
        all_keys_for_baseline = list(dict.fromkeys(train_samples + val_samples + test_samples))
        baseline_lookup = fit_env_main_effects(
            meta_df=meta,
            train_keys=train_samples,
            all_keys=all_keys_for_baseline,
            key_col=metadata_key_col,
            env_data_dict=env_data_dict
        )
        target_raw = pd.to_numeric(meta[TARGET_COL], errors="coerce")
        key_series = meta[metadata_key_col].astype(str)
        for k, t in zip(key_series, target_raw):
            base = baseline_lookup.get(str(k), 0.0)
            if pd.notna(t):
                residual_targets[str(k)] = float(t - base)

        def _residual_stats(keys: List[str]) -> Tuple[float, float]:
            vals = [residual_targets.get(str(k)) for k in keys if str(k) in residual_targets]
            vals = [v for v in vals if v is not None]
            if not vals:
                return 0.0, 1.0
            arr = np.array(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            return mean, std if std > 1e-8 else 1.0

        target_mean, target_std = _residual_stats(train_samples)
        if STANDARDIZE_TARGET:
            set_target_scaler(target_mean, target_std)
        else:
            set_target_scaler(None, None)
        logging.info(
            "Env residual training active: targets are residuals; eval metrics add env baselines to report raw traits."
        )
    else:
        # Fallback to original scaling on raw targets
        target_mean, target_std = _compute_target_scaler(meta, train_samples, metadata_key_col)
        if STANDARDIZE_TARGET:
            set_target_scaler(target_mean, target_std)
        else:
            set_target_scaler(None, None)

    use_env_zscore = (
        USE_ENV_ZSCORE
        and USE_TEMPORAL_ENV_ENCODING
        and env_data_dict is not None
        and "Location_Code" in meta.columns
        and "Year_Code" in meta.columns
    )
    env_target_stats: Dict[int, Tuple[float, float]] = {}
    if use_env_zscore:
        target_lookup = residual_targets if residual_targets else None
        env_target_stats = _build_env_target_stats(
            meta,
            train_samples,
            metadata_key_col,
            target_lookup=target_lookup,
            env_data_dict=env_data_dict,
            sample_key_to_sid=sample_key_to_sid
        )
        if not env_target_stats:
            logging.warning("Env z-score requested but no per-env stats were computed; disabling env z-score.")
            use_env_zscore = False
    standardize_target = STANDARDIZE_TARGET and not use_env_zscore
    if not standardize_target:
        set_target_scaler(None, None)

    # Discover TE subtype channels if requested (tile pipeline removed, so skip discovery)
    te_subtype_keys: List[str] = []
    if USE_TE_SUBTYPE_FEATURES:
        logging.warning("USE_TE_SUBTYPE_FEATURES is enabled but tile-based subtype discovery is disabled; proceeding without subtype channels.")

    IS_SIMPLE = str(TRAIT_MODE).lower() == "simple"
    geno_matrix = None
    geno_ids = None
    geno_map_global = {}
    geno_dim_global = 0
    if IS_SIMPLE:
        logging.info("Loading raw genotypes for simple trait mode...")
        train_sid_set = {sample_key_to_sid.get(k, k) for k in train_samples}
        all_simple_ids = list(dict.fromkeys(train_samples + val_samples + test_samples))
        if GENO_SOURCE == "plink":
            geno_matrix, geno_ids = load_genotypes_plink(PLINK_PREFIX)
            geno_map_global, geno_dim_global = build_geno_transform(
                geno_matrix,
                geno_ids,
                list(train_sid_set),
                use_pca=SIMPLE_GENO_PCA,
                n_pcs=SIMPLE_GENO_N_PCS,
                standardize=SIMPLE_GENO_STANDARDIZE
            )
        elif GENO_SOURCE.lower() == "dosage feature":
            logging.info("Building genotype vectors from tensor dosage feature for simple mode...")
            dosage_matrix, dosage_ids = _build_dosage_matrix_from_tensors(
                all_simple_ids,
                sample_key_to_sid,
                TENSOR_DIR
            )
            geno_map_global, geno_dim_global, _ = build_geno_transform(
                dosage_matrix,
                dosage_ids,
                list(train_sid_set),
                use_pca=SIMPLE_GENO_PCA,
                n_pcs=SIMPLE_GENO_N_PCS,
                standardize=SIMPLE_GENO_STANDARDIZE,
                return_assets=False
            )
        else:
            raise RuntimeError(f"Unsupported GENO_SOURCE={GENO_SOURCE} (expected 'plink' or 'dosage feature').")
    elif USE_DOSAGE_BRANCH and str(DOSAGE_SOURCE).lower() == "plink":
        logging.info("Loading PLINK dosage matrix for dosage branch override (complex/tensor path).")
        try:
            if geno_matrix is None or geno_ids is None:
                geno_matrix, geno_ids = load_genotypes_plink(PLINK_PREFIX)
            geno_matrix = np.nan_to_num(np.asarray(geno_matrix), nan=0.0).astype(np.float32, copy=False)
            DOSAGE_OVERRIDE_MAP = {str(iid): geno_matrix[i] for i, iid in enumerate(geno_ids)}
            DOSAGE_OVERRIDE_DIM = int(geno_matrix.shape[1]) if geno_matrix.ndim == 2 else 0
            DOSAGE_OVERRIDE_ENABLED = bool(DOSAGE_OVERRIDE_MAP) and DOSAGE_OVERRIDE_DIM > 0
            if DOSAGE_OVERRIDE_ENABLED:
                logging.info(
                    "Enabled PLINK dosage override for dosage branch (samples=%d, snps=%d).",
                    len(DOSAGE_OVERRIDE_MAP),
                    DOSAGE_OVERRIDE_DIM
                )
            else:
                logging.warning("PLINK dosage override requested but matrix is empty; continuing with tensor dosage.")
        except Exception as e:
            logging.warning("Failed to load PLINK dosage override; continuing with tensor dosage. Error: %s", e)
            DOSAGE_OVERRIDE_ENABLED = False
            DOSAGE_OVERRIDE_MAP = {}
            DOSAGE_OVERRIDE_DIM = 0

    # Shared dataset kwargs
    base_ds_kwargs = dict(
        metadata=meta,
        environment_data=env,
        env_data_dict=env_data_dict if (USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP) else None,
        target_col=TARGET_COL,
        tensor_dir=TENSOR_DIR,
        sample_key_to_sid=sample_key_to_sid,
        metadata_key_col=metadata_key_col,
        standardize_target=standardize_target,
        target_mean=target_mean,
        target_std=target_std,
        residual_targets=residual_targets,
        env_target_stats=env_target_stats
    )
    if IS_SIMPLE:
        base_ds_kwargs["geno_map"] = geno_map_global
        base_ds_kwargs.pop("tensor_dir", None)
        DS_CLASS = RawGenoEnvDataset
        collate_fn_main = raw_geno_collate_fn
    else:
        DS_CLASS = ChromomapTensorDataset
        collate_fn_main = chromomap_collate_fn

    # Build datasets
    train_ds = DS_CLASS(sample_ids=train_samples, **base_ds_kwargs)
    val_ds = DS_CLASS(sample_ids=val_samples, **base_ds_kwargs)
    test_ds = DS_CLASS(sample_ids=test_samples, **base_ds_kwargs)
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn_main, pin_memory=False)
    train_eval_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_main,
    pin_memory=False
)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn_main, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn_main, pin_memory=False)

    # Infer model category sizes from dataset
    num_env_features = N_ENV_FEATURES_PER_MONTH
    num_locations = N_LOCATIONS
    num_years = N_YEARS
    num_pops = int(meta["Pop_Code"].max()) + 1
    if IS_SIMPLE:
        geno_dim_main = geno_dim_global
        logging.info(f"Model sizes → env={num_env_features}, geno_vec_dim={geno_dim_main}, "
                     f"locations={num_locations}, years={num_years}, pops={num_pops}")
    else:
        num_chromosomes = int(train_ds.n_chr)
        genomic_feature_dim = int(getattr(train_ds, "feature_dim", 0)) if USE_GENOMIC_TENSORS else None
        max_haplotype_blocks = int(getattr(train_ds, "max_block_id_est", 1) or 1)
        if USE_GENOMIC_TENSORS:
            logging.info(f"Model sizes â†’â€™ env={num_env_features}, genomic_feat={genomic_feature_dim}, "
                         f"chrom={num_chromosomes}, locations={num_locations}, years={num_years}, pops={num_pops}")
            logging.info(
                "Tensor feature indices: block_id_raw_idx=%s, te_hotspot_idx=%s, is_te_idx=%s, "
                "is_genic_idx=%s, is_promoter_idx=%s, block_gene_count_idx=%s, "
                "block_snp_density_idx=%s, block_mean_maf_idx=%s, dosage_idx=%s, "
                "token_rank_idx=%s, dosage_local_mean_idx=%s, dosage_local_std_idx=%s, "
                "te_dist_idx=%s, gene_dist_idx=%s",
                getattr(train_ds, "block_id_raw_idx", None),
                getattr(train_ds, "te_hotspot_idx", None),
                getattr(train_ds, "is_te_idx", None),
                getattr(train_ds, "is_genic_idx", None),
                getattr(train_ds, "is_promoter_idx", None),
                getattr(train_ds, "block_gene_count_idx", None),
                getattr(train_ds, "block_snp_density_idx", None),
                getattr(train_ds, "block_mean_maf_idx", None),
                getattr(train_ds, "dosage_idx", None),
                getattr(train_ds, "token_rank_idx", None),
                getattr(train_ds, "dosage_local_mean_idx", None),
                getattr(train_ds, "dosage_local_std_idx", None),
                getattr(train_ds, "te_dist_idx", None),
                getattr(train_ds, "gene_dist_idx", None),
            )
        else:
            logging.info(f"Model sizes â†’â€™ env={num_env_features}, chrom={num_chromosomes}, blocksÃ¢â€°Â¤{max_haplotype_blocks}, "
                         f"locations={num_locations}, years={num_years}, pops={num_pops}")
    logging.info("Using model type: simple_rawgeno" if IS_SIMPLE else "Using model type: gxetensor")

    # Optional CV branch (leaves the original single-split path untouched)
    dosage_pca_components = None
    dosage_pca_mean = None
    dosage_pca_std = None
    if USE_DOSAGE_BRANCH:
        if USE_DOSAGE_PCA:
            dosage_pca_components, dosage_pca_mean, dosage_pca_std = _load_dosage_pca_assets()
            # If assets were not provided, compute from PLINK (train IDs) and optionally persist.
            if (dosage_pca_components is None or dosage_pca_mean is None or dosage_pca_std is None) and not IS_SIMPLE:
                assets_found = False
                # Option B: approximate PCA directly from the tensor dosage channel (no PLINK dependency).
                if (
                    str(DOSAGE_SOURCE).lower() != "plink"
                    and USE_GENOMIC_TENSORS
                    and hasattr(train_ds, "dosage_idx")
                    and train_ds.dosage_idx is not None
                ):
                    comps, mean_arr, std_arr = compute_tensor_dosage_pca(
                        train_ds,
                        max_samples=4000,
                        n_components=SIMPLE_GENO_N_PCS
                    )
                    if comps is not None and mean_arr is not None and std_arr is not None:
                        dosage_pca_components = comps
                        dosage_pca_mean = mean_arr
                        dosage_pca_std = std_arr
                        assets_found = True
                        logging.info(
                            "Computed dosage PCA assets from Chromomap tensors (components=%s, mean=%s, std=%s).",
                            comps.shape,
                            mean_arr.shape,
                            std_arr.shape
                        )
                        _save_dosage_pca_assets(dosage_pca_components, dosage_pca_mean, dosage_pca_std)
                # Option A fallback: reuse PLINK genotypes when available.
                if not assets_found and GENO_SOURCE == "plink":
                    try:
                        geno_matrix, geno_ids = load_genotypes_plink(PLINK_PREFIX)
                        train_sid_set = {sample_key_to_sid.get(k, k) for k in train_samples}
                        _, _, assets = build_geno_transform(
                            geno_matrix,
                            geno_ids,
                            list(train_sid_set),
                            use_pca=SIMPLE_GENO_PCA,
                            n_pcs=SIMPLE_GENO_N_PCS,
                            standardize=SIMPLE_GENO_STANDARDIZE,
                            return_assets=True
                        )
                        if assets:
                            comps, mean_arr, std_arr = assets
                            if comps is not None:
                                dosage_pca_components = comps
                            if mean_arr is not None:
                                dosage_pca_mean = mean_arr
                            if std_arr is not None:
                                dosage_pca_std = std_arr
                            assets_found = any(x is not None for x in (comps, mean_arr, std_arr))
                            logging.info(
                                "Computed dosage PCA assets from PLINK for complex path (components=%s, mean=%s, std=%s).",
                                None if comps is None else comps.shape,
                                None if mean_arr is None else mean_arr.shape,
                                None if std_arr is None else std_arr.shape
                            )
                            _save_dosage_pca_assets(dosage_pca_components, dosage_pca_mean, dosage_pca_std)
                    except Exception as e:
                        logging.warning("Failed to compute dosage PCA assets from PLINK for complex path: %s", e)
                if not assets_found:
                    logging.warning("Dosage PCA assets unavailable; dosage branch will fall back to LazyLinear.")
        else:
            logging.info(
                "USE_DOSAGE_PCA=False: dosage branch will use flattened dosage (all SNP positions) without PCA projection."
            )
    if USE_CV:
        genomic_feature_dim_cv: Optional[int] = None
        num_chromosomes_cv: Optional[int] = None
        geno_dim_cv: Optional[int] = None

        env_encoder_type_local = "mlp" if USE_ENV_MATRIX_AS_MLP else ENV_ENCODER_TYPE
        env_input_dim_cv = N_ENV_FEATURES_PER_MONTH

        def _build_model_cv():
            if IS_SIMPLE:
                if geno_dim_cv is None:
                    raise RuntimeError("Genotype dimension not set for simple-mode CV model.")
                logging.info("Creating GxE_RawGenotypeModel model (CV path)...")
                base_model = GxE_RawGenotypeModel(
                    geno_input_dim=int(geno_dim_cv),
                    embed_dim=EMBED_DIM,
                    env_embed_dim=32,
                    env_encoder_type=env_encoder_type_local,
                    env_hidden_dim=ENV_HIDDEN_DIM,
                    env_lstm_layers=ENV_LSTM_LAYERS,
                    env_conv_channels=ENV_CONV_CHANNELS,
                    env_conv_layers=ENV_CONV_LAYERS,
                    env_conv_kernel=ENV_CONV_KERNEL,
                    env_pyramid_scales=ENV_PYRAMID_SCALES,
                    env_pyramid_layers=ENV_PYRAMID_LAYERS,
                    n_env_features_per_month=env_input_dim_cv,
                    n_months=1 if env_encoder_type_local == "mlp" else N_MONTHS,
                    n_locations=num_locations,
                    n_years=num_years,
                    location_embed_dim=LOCATION_EMBED_DIM,
                    year_embed_dim=YEAR_EMBED_DIM,
                    n_populations=num_pops,
                    pop_embed_dim=POP_EMBED_DIM,
                    dropout=model_dropout,
                    main_head_dropout=MAIN_HEAD_DROPOUT,
                    interaction_head_dropout=INTERACTION_HEAD_DROPOUT,
                    interaction_dim=64,
                    low_rank_bilinear_rank=LRBI_RANK,
                    interaction_reg_lambda=INTERACTION_REG_LAMBDA,
                ).to(device)
                return base_model
            if genomic_feature_dim_cv is None or num_chromosomes_cv is None:
                raise RuntimeError("Genomic tensor dims not set before CV model creation.")
            logging.info("Creating GxE_Transformer_Tensor model (CV path)...")
            feat_names_cv = getattr(model_ds_for_cv, "feature_names", []) or []
            def _idx_or_none(names: List[str], name: str) -> Optional[int]:
                if not names:
                    return None
                try:
                    return names.index(name)
                except ValueError:
                    return None
            def _idx_any(names: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
                for cand in candidates:
                    idx = _idx_or_none(names, cand)
                    if idx is not None:
                        return idx
                return None
            block_id_idx_cv = _idx_or_none(feat_names_cv, "block_id_raw")
            if block_id_idx_cv is None:
                block_id_idx_cv = getattr(model_ds_for_cv, "block_id_raw_idx", None)
            te_hotspot_idx_cv = _idx_any(
                feat_names_cv,
                ("te_hotspot_flag", "te_hotspot_mask", "is_te_hotspot", "te_hotspot")
            )
            if te_hotspot_idx_cv is None:
                te_hotspot_idx_cv = getattr(model_ds_for_cv, "te_hotspot_idx", None)
            te_idx_cv = _idx_or_none(feat_names_cv, "is_te")
            if te_idx_cv is None:
                te_idx_cv = getattr(model_ds_for_cv, "is_te_idx", None)
            genic_idx_cv = _idx_or_none(feat_names_cv, "is_genic")
            if genic_idx_cv is None:
                genic_idx_cv = getattr(model_ds_for_cv, "is_genic_idx", None)
            promoter_idx_cv = _idx_any(
                feat_names_cv,
                ("is_promoter", "is_gene_promoter", "gene_promoter")
            )
            if promoter_idx_cv is None:
                promoter_idx_cv = getattr(model_ds_for_cv, "is_promoter_idx", None)
            dosage_idx_cv = _idx_any(
                feat_names_cv,
                ("dosage", "dosage_norm", "dosage_raw", "dosage_scaled", "dosage_float", "dosage_prior")
            )
            if dosage_idx_cv is None:
                dosage_idx_cv = getattr(model_ds_for_cv, "dosage_idx", None)
            te_dist_idx_cv = _idx_any(
                feat_names_cv,
                ("te_dist", "te_distance", "te_dist_bp", "dist_te", "te_dist_norm")
            )
            if te_dist_idx_cv is None:
                te_dist_idx_cv = getattr(model_ds_for_cv, "te_dist_idx", None)
            gene_dist_idx_cv = _idx_any(
                feat_names_cv,
                ("gene_dist", "gene_distance", "gene_dist_bp", "dist_gene", "genic_dist", "genic_distance", "gene_dist_norm")
            )
            if gene_dist_idx_cv is None:
                gene_dist_idx_cv = getattr(model_ds_for_cv, "gene_dist_idx", None)
            block_gene_idx_cv = _idx_or_none(feat_names_cv, "block_gene_count_norm")
            if block_gene_idx_cv is None:
                block_gene_idx_cv = getattr(model_ds_for_cv, "block_gene_count_idx", None)
            block_density_idx_cv = _idx_or_none(feat_names_cv, "block_snp_density_norm")
            if block_density_idx_cv is None:
                block_density_idx_cv = getattr(model_ds_for_cv, "block_snp_density_idx", None)
            block_maf_idx_cv = _idx_or_none(feat_names_cv, "block_mean_maf_norm")
            if block_maf_idx_cv is None:
                block_maf_idx_cv = getattr(model_ds_for_cv, "block_mean_maf_idx", None)
            logging.info(
                "Functional channel idx (CV): block_id_raw=%s, te_hotspot=%s, is_te=%s, is_genic=%s, "
                "is_promoter=%s, block_gene=%s, block_density=%s, block_maf=%s, dosage=%s, te_dist=%s, gene_dist=%s",
                block_id_idx_cv,
                te_hotspot_idx_cv,
                te_idx_cv,
                genic_idx_cv,
                promoter_idx_cv,
                block_gene_idx_cv,
                block_density_idx_cv,
                block_maf_idx_cv,
                dosage_idx_cv,
                te_dist_idx_cv,
                gene_dist_idx_cv,
            )
            base_model = GxE_Transformer_Tensor(
                genomic_feature_dim=genomic_feature_dim_cv,
                num_chromosomes=num_chromosomes_cv,
                embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS,
                num_intra_layers=NUM_TRANSFORMER_LAYERS,
                num_cross_layers=max(1, NUM_TRANSFORMER_LAYERS // 2),
                ff_dim=FF_DIM,
                n_env_features_per_month=N_ENV_FEATURES_PER_MONTH,
                n_months=N_MONTHS,
                env_hidden_dim=ENV_HIDDEN_DIM,
                env_lstm_layers=ENV_LSTM_LAYERS,
                env_embed_dim=32,
                env_encoder_type=env_encoder_type_local,
                env_conv_channels=ENV_CONV_CHANNELS,
                env_conv_layers=ENV_CONV_LAYERS,
                env_conv_kernel=ENV_CONV_KERNEL,
                env_pyramid_scales=ENV_PYRAMID_SCALES,
                env_pyramid_layers=ENV_PYRAMID_LAYERS,
                n_locations=num_locations,
                n_years=num_years,
                location_embed_dim=LOCATION_EMBED_DIM,
                year_embed_dim=YEAR_EMBED_DIM,
                n_populations=num_pops,
                pop_embed_dim=POP_EMBED_DIM,
                dropout=model_dropout,
                main_head_dropout=MAIN_HEAD_DROPOUT,
                interaction_head_dropout=INTERACTION_HEAD_DROPOUT,
                residual_gate_init=RESIDUAL_GATE_INIT,
                distance_log1p=DISTANCE_LOG1P,
                use_env_anomalies=USE_ENV_ANOMALIES,
                env_anomaly_mean=env_anomaly_mean,
                add_row_embeddings=True,
                row_embed_dim=32,
                chr_downsample_stride=max(1, int(CHR_DOWNSAMPLE_STRIDE)),
                chr_downsample_kernel=CHR_DOWNSAMPLE_KERNEL if CHR_DOWNSAMPLE_KERNEL and CHR_DOWNSAMPLE_KERNEL > 0 else None,
                block_id_channel_idx=block_id_idx_cv,
                te_hotspot_channel_idx=te_hotspot_idx_cv,
                te_channel_idx=te_idx_cv,
                genic_channel_idx=genic_idx_cv,
                promoter_channel_idx=promoter_idx_cv,
                block_gene_count_channel_idx=block_gene_idx_cv,
                block_snp_density_channel_idx=block_density_idx_cv,
                block_mean_maf_channel_idx=block_maf_idx_cv,
                dosage_channel_idx=dosage_idx_cv,
                te_distance_channel_idx=te_dist_idx_cv,
                gene_distance_channel_idx=gene_dist_idx_cv,
                use_dosage_branch=USE_DOSAGE_BRANCH,
                dosage_branch_hidden=DOSAGE_BRANCH_HIDDEN,
                dosage_gate_hidden=DOSAGE_GATE_HIDDEN,
                dosage_gate_dropout=DOSAGE_GATE_DROPOUT,
                dosage_pca_components=dosage_pca_components,
                dosage_pca_mean=dosage_pca_mean,
                dosage_pca_std=dosage_pca_std,
                dosage_blend_prior=DOSAGE_BLEND_PRIOR,
                dosage_fixed_weight=DOSAGE_FIXED_WEIGHT,
                dosage_center=DOSAGE_PCA_CENTER,
                dosage_scale=DOSAGE_PCA_SCALE,
                use_biological_aware_embedding=USE_BIOLOGICAL_AWARE_EMBEDDING,
                use_habe=USE_HABE,
                max_sparse_tokens=MAX_SPARSE_TOKENS,
                hotspot_focus_bias=HOTSPOT_FOCUS_BIAS,
                use_env_film=USE_ENV_FILM,
                use_env_pool_bias=USE_ENV_POOL_BIAS,
                use_meta_film=USE_META_FILM,
                meta_film_scale=META_FILM_SCALE,
                low_rank_bilinear_rank=LRBI_RANK,
                use_gxe_moe=USE_GXE_MOE,
                gxe_moe_num_experts=GXE_MOE_NUM_EXPERTS,
                gxe_moe_hidden_dim=GXE_MOE_HIDDEN_DIM,
                gxe_moe_temperature=GXE_MOE_TEMPERATURE,
                interaction_reg_lambda=INTERACTION_REG_LAMBDA,
                use_env_cross_attention=False,
                n_aux_targets=(len(AUX_TARGETS) if USE_AUX else 0)
            ).to(device)
            base_model.strict_hotspots = True
            base_model.hotspot_focus_bias = 0.0
            if hasattr(base_model, "hotspot_focus_bias_param"):
                base_model.hotspot_focus_bias_param.data.zero_()
            base_model.use_functional_pool_bias = False

            base_model.max_sparse_tokens = 128
            if hasattr(base_model, "habe") and hasattr(base_model.habe, "sparse_injector"):
                base_model.habe.sparse_injector.max_sparse_tokens = 128
            if USE_DUAL_BRANCH_MODEL and not IS_SIMPLE:
                base_model = DualBranchGxE(
                    base_model,
                    additive_hidden_dim=ADDITIVE_BRANCH_HIDDEN,
                    gate_hidden_dim=DUAL_GATE_HIDDEN,
                    gate_dropout=DUAL_GATE_DROPOUT
                ).to(device)
            return base_model


        base_ds_kwargs = dict(
            metadata=meta,
            environment_data=env,
            env_data_dict=env_data_dict if (USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP) else None,
            target_col=TARGET_COL,
            tensor_dir=TENSOR_DIR,
            sample_key_to_sid=sample_key_to_sid,
            metadata_key_col=metadata_key_col,
            standardize_target=STANDARDIZE_TARGET,
            target_mean=None,
            target_std=None,
            residual_targets=residual_targets
        )
        if IS_SIMPLE:
            base_ds_kwargs.pop("tensor_dir", None)
            base_ds_kwargs["geno_map"] = geno_map_global
        collate_fn_cv = raw_geno_collate_fn if IS_SIMPLE else chromomap_collate_fn

        all_ids = sorted(set(train_samples + val_samples + test_samples))
        diagnose_cv_leakage(meta, metadata_key_col, sample_key_to_sid, all_ids, CV_FOLDS, SEED)
        cv_jobs: List[Tuple[str, str, List[Tuple[List[str], List[str], List[str]]]]] = []
        if EVAL_TIER_MODE in ("geno_cv", "both"):
            if USE_GENOTYPE_CV:
                fold_triplets_geno = run_genotype_cv(all_ids, sample_key_to_sid, CV_FOLDS, SEED)
                check_cv_fold_overlap(fold_triplets_geno, context="genotype-cv")
                cv_jobs.append(("tier1_geno", "Genotype-CV (unseen genotypes)", fold_triplets_geno))
            else:
                logging.warning("EVAL_TIER_MODE includes genotype CV but USE_GENOTYPE_CV=False; skipping tier1.")
        if EVAL_TIER_MODE in ("within_genotype_env_holdout", "both"):
            if metadata_key_col != "SplitKey" or metadata_key_col not in meta.columns:
                logging.warning("Within-genotype env holdout requires metadata_key_col='SplitKey' and column present; skipping tier2.")
            else:
                sid_col = "SampleID_str" if "SampleID_str" in meta.columns else ("SampleID" if "SampleID" in meta.columns else None)
                if sid_col is None:
                    logging.warning("No SampleID column found for within-genotype env holdout; skipping tier2.")
                else:
                    fold_triplets_within = run_within_genotype_env_holdout_cv(
                        meta=meta,
                        split_key_col=metadata_key_col,
                        sampleid_col=sid_col,
                        n_folds=CV_FOLDS,
                        test_frac=WITHIN_GENO_TEST_FRAC,
                        val_frac=WITHIN_GENO_VAL_FRAC,
                        seed=SEED + WITHIN_GENO_SEED_OFFSET,
                        min_test=WITHIN_GENO_MIN_TEST,
                        min_train=WITHIN_GENO_MIN_TRAIN
                    )
                    check_within_genotype_split(
                        meta,
                        fold_triplets_within,
                        split_key_col=metadata_key_col,
                        sampleid_col=sid_col
                    )
                    cv_jobs.append(("tier2_env", "Within-genotype env holdout (sparse trial completion; environments shared with training)", fold_triplets_within))

        if EVAL_TIER_MODE in ("env_blocked", "all"):
            try:
                from eval_env_blocked import run_environment_blocked_cv
                _sid_eb = "SampleID_str" if "SampleID_str" in meta.columns else ("SampleID" if "SampleID" in meta.columns else None)
                if _sid_eb is None or metadata_key_col not in meta.columns:
                    logging.warning("Environment-blocked CV needs SplitKey + SampleID columns; skipping.")
                else:
                    for _eb_mode in ENV_BLOCKED_MODES:
                        try:
                            _eb_tr = run_environment_blocked_cv(meta=meta, split_key_col=metadata_key_col, sampleid_col=_sid_eb, block_by=_eb_mode, location_col="Location", year_col="Year", val_frac=WITHIN_GENO_VAL_FRAC, seed=SEED)
                            cv_jobs.append((f"tier3_{_eb_mode}", f"Environment-blocked ({_eb_mode} held out; truly unseen environment)", _eb_tr))
                        except Exception as _e:
                            logging.warning("env-blocked mode %s skipped: %s", _eb_mode, _e)
            except Exception as _e2:
                logging.warning("Environment-blocked CV setup failed: %s", _e2)

        if not cv_jobs:
            raise RuntimeError("No CV jobs scheduled. Check EVAL_TIER_MODE / USE_GENOTYPE_CV / metadata columns.")

        set_target_scaler(None, None)
        for job_tag, job_desc, fold_triplets in cv_jobs:
            fold_metrics, fold_preds, fold_emb_rows = [], [], []
            fold_emb_rows_by_view: Dict[str, List[Dict[str, Any]]] = {}
            total_folds = len(fold_triplets) if fold_triplets else CV_FOLDS
            logging.info("[%s] Starting CV: %s (%d folds)", job_tag, job_desc, total_folds)

            for fold, (tr_ids, va_ids, te_ids) in enumerate(fold_triplets, start=1):
                fold_analysis = analyze_target_distribution(
                    meta_df=meta.copy(),
                    train_keys=tr_ids,
                    val_keys=va_ids,
                    test_keys=te_ids,
                    key_col=metadata_key_col,
                    target_col=TARGET_COL
                )

                if not tr_ids or not te_ids:
                    logging.warning(
                        "[%s Fold %d] Skipping fold because train/test splits are empty (train=%d, val=%d, test=%d).",
                        job_tag,
                        fold,
                        len(tr_ids),
                        len(va_ids),
                        len(te_ids),
                    )
                    continue
                if not va_ids:
                    fallback_n = max(1, min(len(tr_ids), max(1, len(tr_ids) // 10)))
                    va_ids = tr_ids[:fallback_n]
                    logging.warning(
                        "[%s Fold %d] Validation split was empty; recycling %d train samples as val.",
                        job_tag,
                        fold,
                        fallback_n,
                    )

                fold_env_data_dict = env_data_dict if (USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP) else None
                if USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP:
                    try:
                        fold_env_data_dict = build_env_data_dict_fold(env, tr_ids)
                    except Exception as e:
                        logging.warning(
                            "[%s Fold %d] Fold-specific env preprocessing failed; falling back to global env stats. Error: %s",
                            job_tag,
                            fold,
                            e,
                        )
                if USE_ENV_MATRIX_AS_MLP and fold_env_data_dict is not None and fold_env_data_dict.get("env_wide") is not None:
                    env_input_dim_cv = int(fold_env_data_dict["env_wide"].shape[1])
                else:
                    env_input_dim_cv = N_ENV_FEATURES_PER_MONTH

                fold_baseline_lookup = None
                fold_residual_targets = residual_targets
                fold_target_mean, fold_target_std = _compute_target_scaler(
                    meta,
                    tr_ids,
                    metadata_key_col
                )
                if use_env_residuals:
                    all_keys_for_baseline = list(dict.fromkeys(tr_ids + va_ids + te_ids))
                    fold_baseline_lookup = fit_env_main_effects(
                        meta_df=meta,
                        train_keys=tr_ids,
                        all_keys=all_keys_for_baseline,
                        key_col=metadata_key_col,
                        env_data_dict=fold_env_data_dict
                    )
                    fold_residual_targets = {}
                    key_set = {str(k) for k in all_keys_for_baseline}
                    target_raw = pd.to_numeric(meta[TARGET_COL], errors="coerce")
                    key_series = meta[metadata_key_col].astype(str)
                    for k, t in zip(key_series, target_raw):
                        k_str = str(k)
                        if k_str not in key_set or pd.isna(t):
                            continue
                        base = fold_baseline_lookup.get(k_str, 0.0)
                        fold_residual_targets[k_str] = float(t - base)
                    vals = [fold_residual_targets.get(str(k)) for k in tr_ids if str(k) in fold_residual_targets]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        arr = np.array(vals, dtype=float)
                        fold_target_mean = float(arr.mean())
                        fold_target_std = float(arr.std(ddof=0))
                        if fold_target_std <= 1e-8:
                            fold_target_std = 1.0
                    else:
                        fold_target_mean, fold_target_std = 0.0, 1.0
                fold_env_target_stats: Dict[int, Tuple[float, float]] = {}
                if use_env_zscore:
                    fold_env_target_stats = _build_env_target_stats(
                        meta,
                        tr_ids,
                        metadata_key_col,
                        target_lookup=fold_residual_targets if fold_residual_targets else None,
                        env_data_dict=fold_env_data_dict,
                        sample_key_to_sid=sample_key_to_sid
                    )
                    if not fold_env_target_stats:
                        logging.warning(f"[{job_tag} Fold {fold}] Env z-score enabled but no per-env stats were computed.")
                fold_aux_target_lookup = None
                if USE_AUX:
                    fold_aux_target_lookup = _build_aux_target_lookup(
                        AUX_SOURCE_DF,
                        train_keys=tr_ids,
                        all_keys=list(dict.fromkeys(tr_ids + va_ids + te_ids)),
                        key_col=metadata_key_col,
                        aux_cols=AUX_TARGETS,
                    )
                if standardize_target:
                    set_target_scaler(fold_target_mean, fold_target_std)
                else:
                    set_target_scaler(None, None)

                ds_kwargs_fold_v1 = dict(
                    base_ds_kwargs,
                    env_data_dict=fold_env_data_dict if (USE_TEMPORAL_ENV_ENCODING or USE_ENV_MATRIX_AS_MLP) else None,
                    standardize_target=standardize_target,
                    target_mean=fold_target_mean,
                    target_std=fold_target_std,
                    residual_targets=fold_residual_targets,
                    env_target_stats=fold_env_target_stats if use_env_zscore else {}
                )
                if IS_SIMPLE:
                    ds_kwargs_fold_v1.pop("tensor_dir", None)
                    # build fold-specific genotype map using train genotypes
                    geno_map_fold, geno_dim_cv = build_geno_transform(
                        geno_matrix,
                        geno_ids,
                        list({sample_key_to_sid.get(k, k) for k in tr_ids}),
                        use_pca=SIMPLE_GENO_PCA,
                        n_pcs=SIMPLE_GENO_N_PCS,
                        standardize=SIMPLE_GENO_STANDARDIZE
                    )
                    ds_kwargs_fold_v1["geno_map"] = geno_map_fold
                    env_input_dim_cv = env_input_dim_cv  # already set above

                fold_train_ds = DS_CLASS(sample_ids=tr_ids, **ds_kwargs_fold_v1)
                fold_val_ds = DS_CLASS(sample_ids=va_ids, **ds_kwargs_fold_v1)
                fold_test_ds = DS_CLASS(sample_ids=te_ids, **ds_kwargs_fold_v1)
                model_ds_for_cv = fold_train_ds

                if USE_GENOMIC_TENSORS and not IS_SIMPLE:
                    genomic_feature_dim_cv = int(getattr(fold_train_ds, "feature_dim", 0))
                    num_chromosomes_cv = int(getattr(fold_train_ds, "n_chr", getattr(fold_train_ds, "num_chromosomes", 0)))
                    logging.info(
                        f"[{job_tag} Fold {fold}/{total_folds}] Tensor dims →’ genomic_feat={genomic_feature_dim_cv}, chrom={num_chromosomes_cv}, "
                        f"locations={num_locations}, years={num_years}, pops={num_pops}"
                    )
                    logging.info(
                        "[%s Fold %d/%d] Tensor feature indices: block_id_raw_idx=%s, te_hotspot_idx=%s, "
                        "is_te_idx=%s, is_genic_idx=%s, is_promoter_idx=%s, block_gene_count_idx=%s, "
                        "block_snp_density_idx=%s, block_mean_maf_idx=%s, dosage_idx=%s, "
                        "token_rank_idx=%s, dosage_local_mean_idx=%s, dosage_local_std_idx=%s, "
                        "te_dist_idx=%s, gene_dist_idx=%s",
                        job_tag,
                        fold,
                        total_folds,
                        getattr(fold_train_ds, "block_id_raw_idx", None),
                        getattr(fold_train_ds, "te_hotspot_idx", None),
                        getattr(fold_train_ds, "is_te_idx", None),
                        getattr(fold_train_ds, "is_genic_idx", None),
                        getattr(fold_train_ds, "is_promoter_idx", None),
                        getattr(fold_train_ds, "block_gene_count_idx", None),
                        getattr(fold_train_ds, "block_snp_density_idx", None),
                        getattr(fold_train_ds, "block_mean_maf_idx", None),
                        getattr(fold_train_ds, "dosage_idx", None),
                        getattr(fold_train_ds, "token_rank_idx", None),
                        getattr(fold_train_ds, "dosage_local_mean_idx", None),
                        getattr(fold_train_ds, "dosage_local_std_idx", None),
                        getattr(fold_train_ds, "te_dist_idx", None),
                        getattr(fold_train_ds, "gene_dist_idx", None),
                    )

                fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS, collate_fn=collate_fn_cv, pin_memory=False)
                fold_train_eval_loader = DataLoader(
                    fold_train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    collate_fn=collate_fn_cv,
                    pin_memory=False
                )
                fold_val_loader = DataLoader(fold_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=NUM_WORKERS, collate_fn=collate_fn_cv, pin_memory=False)
                fold_test_loader = DataLoader(fold_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=NUM_WORKERS, collate_fn=collate_fn_cv, pin_memory=False)

                if fold == 3:
                    try:
                        for b_idx, batch in enumerate(fold_val_loader):
                            if len(batch) == 9:
                                loc_ids = batch[4]
                                year_ids = batch[5]
                            elif len(batch) == 7:
                                loc_ids = batch[2]
                                year_ids = batch[3]
                            else:
                                loc_ids = None
                                year_ids = None
                            if loc_ids is not None and year_ids is not None:
                                logging.info(
                                    f"[{job_tag} Fold {fold}] Diagnostic loc_ids min={loc_ids.min().item()} max={loc_ids.max().item()} "
                                    f"(expected 0–{num_locations - 1}), "
                                    f"year_ids min={year_ids.min().item()} max={year_ids.max().item()} "
                                    f"(expected 0–{num_years - 1})"
                                )
                            break
                    except Exception as e:
                        logging.warning(f"[{job_tag} Fold {fold}] Diagnostic on validation loader failed: {e}")

                model = _build_model_cv()
                logging.info(f"[{job_tag} Fold {fold}/{total_folds}] Model parameters: {_safe_param_count(model):,}")
                do_ssl = PRETRAIN_GENOMIC_SIMCLR and job_tag != "tier2_env" and not IS_SIMPLE
                if do_ssl:
                    pretrain_sids = sorted({sample_key_to_sid.get(k, k) for k in tr_ids})
                    ssl_ds = GenomicOnlyTensorDataset(
                        pretrain_sids,
                        tensor_dir=TENSOR_DIR,
                        feature_dim=genomic_feature_dim_cv,
                        drop_feature_indices=getattr(fold_train_ds, "drop_feature_indices", []),
                        feature_names=getattr(fold_train_ds, "feature_names", [])
                    )
                    ssl_loader = DataLoader(
                        ssl_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_genomic_only,
                        pin_memory=False
                    )
                    logging.info(f"[{job_tag} Fold {fold}/{total_folds}] Running genomic SSL pretraining for %d epochs...", SIMCLR_EPOCHS)
                    run_genomic_simclr_pretraining(
                        model,
                        ssl_loader,
                        device,
                        epochs=SIMCLR_EPOCHS,
                        lr=SIMCLR_LR,
                        temp=SIMCLR_TEMP,
                        token_drop_p=SIMCLR_TOKEN_DROP,
                        feature_noise=SIMCLR_FEATURE_NOISE
                    )
                criterion = build_loss_fn(LOSS_FUNCTION)
                snapshot_prefix = f"{job_tag}_cv_fold{fold}"
                if USE_RESIDUAL_FOCUS_ARCH and _supports_gxe_stage(model):
                    m = train_and_eval_two_stage(
                        model=model,
                        train_eval_loader=fold_train_eval_loader,
                        train_loader=fold_train_loader,
                        val_loader=fold_val_loader,
                        test_loader=fold_test_loader,
                        device=device,
                        criterion=criterion,
                        baseline_lookup=fold_baseline_lookup if use_env_residuals else None,
                        env_target_stats=fold_env_target_stats if use_env_zscore else None,
                        aux_target_lookup=fold_aux_target_lookup,
                        aux_loss_weight=(AUX_LOSS_WEIGHT if USE_AUX else 0.0),
                        snapshot_cycle_length=SNAPSHOT_CYCLE_LENGTH,
                        snapshot_prefix=snapshot_prefix,
                        monitor_test=False
                    )
                else:
                    optimizer = create_gxe_optimizer(
                        model,
                        lr=LEARNING_RATE,
                        weight_decay=model_weight_decay,
                        pop_weight_decay=POP_EMBED_WEIGHT_DECAY,
                        metadata_weight_decay=METADATA_WEIGHT_DECAY
                    )
                    scheduler = build_scheduler(optimizer, max_epochs=NUM_EPOCHS)

                    m = train_and_eval_once(
                        model, fold_train_eval_loader, fold_train_loader, fold_val_loader, fold_test_loader, device,
                        optimizer, scheduler, criterion,
                        max_epochs=NUM_EPOCHS,
                        early_stop_patience=EARLY_STOP_PATIENCE,
                        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
                        baseline_lookup=fold_baseline_lookup if use_env_residuals else None,
                        env_target_stats=fold_env_target_stats if use_env_zscore else None,
                        aux_target_lookup=fold_aux_target_lookup,
                        aux_loss_weight=(AUX_LOSS_WEIGHT if USE_AUX else 0.0),
                        snapshot_cycle_length=SNAPSHOT_CYCLE_LENGTH,
                        snapshot_prefix=snapshot_prefix,
                        monitor_test=False
                    )
                fold_metrics.append(m)
                preds = predict_with_ids(
                    model,
                    fold_test_loader,
                    device,
                    baseline_lookup=fold_baseline_lookup if use_env_residuals else None,
                    env_target_stats=fold_env_target_stats if use_env_zscore else None
                )
                for row in preds:
                    row["fold"] = fold
                if SAVE_FOLD_PREDICTIONS:
                    fold_preds.extend(preds)
                if EXPORT_EMBEDDINGS:
                    for view in EMBEDDING_VIEWS:
                        vecs, targ_vec, sid_list = collect_embeddings(model, fold_test_loader, device, view=view)
                        for sid, v, t in zip(sid_list, vecs, targ_vec):
                            row = {"fold": fold, "SampleID": sid, "target": float(t)}
                            for i, x in enumerate(v):
                                row[f"repr_{i}"] = float(x)
                            if view == "fused":
                                fold_emb_rows.append(row)
                            else:
                                fold_emb_rows_by_view.setdefault(view, []).append(row)
                print(f"\n{'='*60}")
                print(f"{job_desc} | FOLD {fold} DETAILED ANALYSIS")
                print(f"{'='*60}")
                val_preds = predict_with_ids(
                    model,
                    fold_val_loader,
                    device,
                    baseline_lookup=fold_baseline_lookup if use_env_residuals else None,
                    env_target_stats=fold_env_target_stats if use_env_zscore else None
                )
                test_preds = preds
                print("\nValidation (seen environments):")
                analyze_env_performance(val_preds, meta, metadata_key_col)
                print("\nTest (UNSEEN environment):")
                analyze_env_performance(test_preds, meta, metadata_key_col)

            if not fold_metrics:
                logging.warning("[%s] No completed folds (likely due to empty splits); skipping metrics/logging.", job_tag)
                continue

            val_r2s = [f["val"][1] for f in fold_metrics]
            test_r2s = [f["test"][1] for f in fold_metrics]
            val_mean, val_std = np.nanmean(val_r2s), np.nanstd(val_r2s)
            test_mean, test_std = np.nanmean(test_r2s), np.nanstd(test_r2s)
            val_rmse = [f["val"][2] for f in fold_metrics]
            val_mae = [f["val"][3] for f in fold_metrics]
            val_ccc = [f["val"][4] for f in fold_metrics]
            test_rmse = [f["test"][2] for f in fold_metrics]
            test_mae = [f["test"][3] for f in fold_metrics]
            test_ccc = [f["test"][4] for f in fold_metrics]
            val_rmse_mean, val_rmse_std = np.nanmean(val_rmse), np.nanstd(val_rmse)
            val_mae_mean, val_mae_std = np.nanmean(val_mae), np.nanstd(val_mae)
            test_rmse_mean, test_rmse_std = np.nanmean(test_rmse), np.nanstd(test_rmse)
            test_mae_mean, test_mae_std = np.nanmean(test_mae), np.nanstd(test_mae)
            val_ccc_mean, val_ccc_std = np.nanmean(val_ccc), np.nanstd(val_ccc)
            test_ccc_mean, test_ccc_std = np.nanmean(test_ccc), np.nanstd(test_ccc)
            logging.info(
                "[%s] CV complete. Metrics (mean +/- std across folds): "
                f"Val R2 {val_mean:.4f} +/- {val_std:.4f} | "
                f"Test R2 {test_mean:.4f} +/- {test_std:.4f} | "
                f"Val RMSE {val_rmse_mean:.4f} +/- {val_rmse_std:.4f} | "
                f"Test RMSE {test_rmse_mean:.4f} +/- {test_rmse_std:.4f} | "
                f"Val MAE {val_mae_mean:.4f} +/- {val_mae_std:.4f} | "
                f"Test MAE {test_mae_mean:.4f} +/- {test_mae_std:.4f} | "
                f"Val CCC {val_ccc_mean:.4f} +/- {val_ccc_std:.4f} | "
                f"Test CCC {test_ccc_mean:.4f} +/- {test_ccc_std:.4f}",
                job_tag
            )
            if SAVE_FOLD_PREDICTIONS and fold_preds:
                preds_path = f"{job_tag}_cv_fold_predictions.csv"
                pd.DataFrame(fold_preds).to_csv(preds_path, index=False)
                logging.info("Saved CV fold predictions to %s", preds_path)
                y_true = np.array([r["true"] for r in fold_preds], dtype=float)
                y_pred = np.array([r["pred"] for r in fold_preds], dtype=float)
                plt.figure(figsize=(5, 5))
                plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
                lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
                plt.plot(lims, lims, 'k--', linewidth=1)
                plt.xlim(lims); plt.ylim(lims)
                plt.xlabel("True"); plt.ylabel("Predicted")
                y_true, y_pred = _filter_finite_pairs(y_true, y_pred)
                if len(y_true) > 1:
                    r2cv = r2_score(y_true, y_pred)
                    rmsecv = math.sqrt(mean_squared_error(y_true, y_pred))
                    maecv = mean_absolute_error(y_true, y_pred)
                    ccc_cv = concordance_correlation_coefficient(y_true, y_pred)
                    ceiling_cv = residual_variance_ceiling(y_true, y_pred)
                    logging.info(
                        "[%s] CV pooled trait ceiling ~= 1 - Var(residual)/Var(y): %.4f",
                        job_tag, ceiling_cv
                    )
                else:
                    r2cv = rmsecv = maecv = ccc_cv = float("nan")
                plt.title(f"CV scatter R2={r2cv:.3f} RMSE={rmsecv:.2f} MAE={maecv:.2f} CCC={ccc_cv:.3f}")
                plt.tight_layout()
                scatter_path = f"{job_tag}_cv_scatter.png"
                plt.savefig(scatter_path, dpi=200)
                plt.close()
                logging.info("Saved CV scatter plot to %s", scatter_path)
                plot_calibration(fold_preds, bins=10, fname=f"{job_tag}_cv_calibration.png")
            if EXPORT_EMBEDDINGS and fold_emb_rows:
                emb_path = f"{job_tag}_penultimate_embeddings_cv.csv"
                pd.DataFrame(fold_emb_rows).to_csv(emb_path, index=False)
                logging.info(f"Saved CV embeddings to {emb_path}")
                if RUN_EMBED_PLOTS and "fused" in TSNE_EMBEDDING_VIEWS:
                    plot_embeddings_tsne(emb_path, meta, out_prefix=f"{job_tag}_emb_tsne_cv")
                    emb_df = pd.DataFrame(fold_emb_rows)
                    repr_cols = [c for c in emb_df.columns if c.startswith("repr_")]
                    if repr_cols:
                        vecs = emb_df[repr_cols].values
                        targ_vec = emb_df["target"].values
                        plot_tsne_basic(vecs, targ_vec, out_prefix=f"{job_tag}_tsne_penultimate_cv")
                        plot_pairwise_dist_vs_target(vecs, targ_vec, fname=f"{job_tag}_pairwise_dist_vs_target_diff_cv.png")
                    fold_vals = sorted(
                        {int(f) for f in emb_df.get("fold", pd.Series(dtype=int)).dropna().unique()}
                    )
                    for fold_val in fold_vals:
                        fold_df = emb_df[emb_df["fold"] == fold_val]
                        if len(fold_df) < 2:
                            continue
                        fold_csv = f"{job_tag}_penultimate_embeddings_cv_fold{fold_val}.csv"
                        fold_df.to_csv(fold_csv, index=False)
                        plot_embeddings_tsne(fold_csv, meta, out_prefix=f"{job_tag}_emb_tsne_cv_fold{fold_val}")
                        repr_cols = [c for c in fold_df.columns if c.startswith("repr_")]
                        if repr_cols:
                            vecs = fold_df[repr_cols].values
                            targ_vec = fold_df["target"].values
                            plot_tsne_basic(vecs, targ_vec, out_prefix=f"{job_tag}_tsne_penultimate_cv_fold{fold_val}")
            if EXPORT_EMBEDDINGS and fold_emb_rows_by_view:
                for view, rows in sorted(fold_emb_rows_by_view.items()):
                    if not rows:
                        continue
                    emb_path = f"{job_tag}_penultimate_embeddings_cv_{view}.csv"
                    pd.DataFrame(rows).to_csv(emb_path, index=False)
                    logging.info(f"Saved CV embeddings ({view}) to {emb_path}")
                    if RUN_EMBED_PLOTS and view in TSNE_EMBEDDING_VIEWS:
                        plot_embeddings_tsne(emb_path, meta, out_prefix=f"{job_tag}_emb_tsne_cv_{view}")
                        emb_df = pd.DataFrame(rows)
                        repr_cols = [c for c in emb_df.columns if c.startswith("repr_")]
                        if repr_cols:
                            vecs = emb_df[repr_cols].values
                            targ_vec = emb_df["target"].values
                            plot_tsne_basic(vecs, targ_vec, out_prefix=f"{job_tag}_tsne_penultimate_cv_{view}")
                            plot_pairwise_dist_vs_target(vecs, targ_vec, fname=f"{job_tag}_pairwise_dist_vs_target_diff_cv_{view}.png")
                        fold_vals = sorted(
                            {int(f) for f in emb_df.get("fold", pd.Series(dtype=int)).dropna().unique()}
                        )
                        for fold_val in fold_vals:
                            fold_df = emb_df[emb_df["fold"] == fold_val]
                            if len(fold_df) < 2:
                                continue
                            fold_csv = f"{job_tag}_penultimate_embeddings_cv_{view}_fold{fold_val}.csv"
                            fold_df.to_csv(fold_csv, index=False)
                            plot_embeddings_tsne(fold_csv, meta, out_prefix=f"{job_tag}_emb_tsne_cv_{view}_fold{fold_val}")
                            repr_cols = [c for c in fold_df.columns if c.startswith("repr_")]
                            if repr_cols:
                                vecs = fold_df[repr_cols].values
                                targ_vec = fold_df["target"].values
                                plot_tsne_basic(vecs, targ_vec, out_prefix=f"{job_tag}_tsne_penultimate_cv_{view}_fold{fold_val}")
            if EXPORT_EMBEDDINGS and fold_emb_rows and val_r2s and not np.all(np.isnan(val_r2s)):
                best_fold_idx = int(np.nanargmax(val_r2s))
                best_fold_num = best_fold_idx + 1
                best_rows = [r for r in fold_emb_rows if int(r.get("fold", -1)) == best_fold_num]
                if best_rows:
                    best_emb_df = pd.DataFrame(best_rows)
                    best_emb_path = f"{job_tag}_penultimate_embeddings_cv_best_fold.csv"
                    best_emb_df.to_csv(best_emb_path, index=False)
                    logging.info(f"Saved best-fold (fold {best_fold_num}) embeddings to {best_emb_path}")
                    if RUN_EMBED_PLOTS:
                        plot_latent_space_summary_from_embeddings(best_emb_df, meta, tag=f"{job_tag}_fold{best_fold_num}")
        return

    # Model
    if USE_GENOMIC_TENSORS:
        genomic_feature_dim_main = int(getattr(train_ds, "feature_dim", 0))
        num_chromosomes_main = int(getattr(train_ds, "n_chr", getattr(train_ds, "num_chromosomes", 0)))
        logging.info(
            f"Tensor model dims â†’â€™ genomic_feat={genomic_feature_dim_main}, chrom={num_chromosomes_main}, "
            f"locations={num_locations}, years={num_years}, pops={num_pops}"
        )
        block_id_idx_main = getattr(train_ds, "block_id_raw_idx", None)
        feat_names_main = getattr(train_ds, "feature_names", []) or []
        def _idx_or_none(names: List[str], name: str) -> Optional[int]:
            if not names:
                return None
            try:
                return names.index(name)
            except ValueError:
                return None
        def _idx_any(names: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
            for cand in candidates:
                idx = _idx_or_none(names, cand)
                if idx is not None:
                    return idx
            return None
        if block_id_idx_main is None:
            block_id_idx_main = _idx_or_none(feat_names_main, "block_id_raw")
        te_hotspot_idx_main = _idx_any(
            feat_names_main,
            ("te_hotspot_flag", "te_hotspot_mask", "is_te_hotspot", "te_hotspot")
        )
        if te_hotspot_idx_main is None:
            te_hotspot_idx_main = getattr(train_ds, "te_hotspot_idx", None)
        te_idx_main = _idx_or_none(feat_names_main, "is_te")
        if te_idx_main is None:
            te_idx_main = getattr(train_ds, "is_te_idx", None)
        genic_idx_main = _idx_or_none(feat_names_main, "is_genic")
        if genic_idx_main is None:
            genic_idx_main = getattr(train_ds, "is_genic_idx", None)
        promoter_idx_main = _idx_any(
            feat_names_main,
            ("is_promoter", "is_gene_promoter", "gene_promoter")
        )
        if promoter_idx_main is None:
            promoter_idx_main = getattr(train_ds, "is_promoter_idx", None)
        dosage_idx_main = _idx_any(
            feat_names_main,
            ("dosage", "dosage_norm", "dosage_raw", "dosage_scaled", "dosage_float", "dosage_prior")
        )
        if dosage_idx_main is None:
            dosage_idx_main = getattr(train_ds, "dosage_idx", None)
        te_dist_idx_main = _idx_any(
            feat_names_main,
            ("te_dist", "te_distance", "te_dist_bp", "dist_te", "te_dist_norm")
        )
        if te_dist_idx_main is None:
            te_dist_idx_main = getattr(train_ds, "te_dist_idx", None)
        gene_dist_idx_main = _idx_any(
            feat_names_main,
            ("gene_dist", "gene_distance", "gene_dist_bp", "dist_gene", "genic_dist", "genic_distance", "gene_dist_norm")
        )
        if gene_dist_idx_main is None:
            gene_dist_idx_main = getattr(train_ds, "gene_dist_idx", None)
        block_gene_idx_main = _idx_or_none(feat_names_main, "block_gene_count_norm")
        if block_gene_idx_main is None:
            block_gene_idx_main = getattr(train_ds, "block_gene_count_idx", None)
        block_density_idx_main = _idx_or_none(feat_names_main, "block_snp_density_norm")
        if block_density_idx_main is None:
            block_density_idx_main = getattr(train_ds, "block_snp_density_idx", None)
        block_maf_idx_main = _idx_or_none(feat_names_main, "block_mean_maf_norm")
        if block_maf_idx_main is None:
            block_maf_idx_main = getattr(train_ds, "block_mean_maf_idx", None)
        logging.info(
            "Functional channel idx (train): block_id_raw=%s, te_hotspot=%s, is_te=%s, is_genic=%s, "
            "is_promoter=%s, block_gene=%s, block_density=%s, block_maf=%s, dosage=%s, te_dist=%s, gene_dist=%s",
            block_id_idx_main,
            te_hotspot_idx_main,
            te_idx_main,
            genic_idx_main,
            promoter_idx_main,
            block_gene_idx_main,
            block_density_idx_main,
            block_maf_idx_main,
            dosage_idx_main,
            te_dist_idx_main,
            gene_dist_idx_main,
        )
        env_encoder_type_local = "mlp" if USE_ENV_MATRIX_AS_MLP else ENV_ENCODER_TYPE
        env_input_dim_main = N_ENV_FEATURES_PER_MONTH
        model = GxE_Transformer_Tensor(
            genomic_feature_dim=genomic_feature_dim_main,
            num_chromosomes=num_chromosomes_main,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_intra_layers=NUM_TRANSFORMER_LAYERS,
            num_cross_layers=max(1, NUM_TRANSFORMER_LAYERS // 2),
            ff_dim=FF_DIM,
            n_env_features_per_month=env_input_dim_main,
            n_months=N_MONTHS,
            env_hidden_dim=ENV_HIDDEN_DIM,
            env_lstm_layers=ENV_LSTM_LAYERS,
            env_embed_dim=32,
            env_encoder_type=env_encoder_type_local,
            n_locations=num_locations,
            n_years=num_years,
            location_embed_dim=LOCATION_EMBED_DIM,
            year_embed_dim=YEAR_EMBED_DIM,
            n_populations=num_pops,
            pop_embed_dim=POP_EMBED_DIM,
            dropout=model_dropout,
            main_head_dropout=MAIN_HEAD_DROPOUT,
            interaction_head_dropout=INTERACTION_HEAD_DROPOUT,
            residual_gate_init=RESIDUAL_GATE_INIT,
            distance_log1p=DISTANCE_LOG1P,
            use_env_anomalies=USE_ENV_ANOMALIES,
            env_anomaly_mean=env_anomaly_mean,
            add_row_embeddings=True,
            row_embed_dim=32,
            chr_downsample_stride=max(1, int(CHR_DOWNSAMPLE_STRIDE)),
            chr_downsample_kernel=(
                CHR_DOWNSAMPLE_KERNEL if CHR_DOWNSAMPLE_KERNEL and CHR_DOWNSAMPLE_KERNEL > 0 else None
            ),
            block_id_channel_idx=block_id_idx_main,
            te_hotspot_channel_idx=te_hotspot_idx_main,
            te_channel_idx=te_idx_main,
            genic_channel_idx=genic_idx_main,
            promoter_channel_idx=promoter_idx_main,
            block_gene_count_channel_idx=block_gene_idx_main,
            block_snp_density_channel_idx=block_density_idx_main,
            block_mean_maf_channel_idx=block_maf_idx_main,
            dosage_channel_idx=dosage_idx_main,
            te_distance_channel_idx=te_dist_idx_main,
            gene_distance_channel_idx=gene_dist_idx_main,
            use_dosage_branch=USE_DOSAGE_BRANCH,
            dosage_branch_hidden=DOSAGE_BRANCH_HIDDEN,
            dosage_gate_hidden=DOSAGE_GATE_HIDDEN,
            dosage_gate_dropout=DOSAGE_GATE_DROPOUT,
            dosage_pca_components=dosage_pca_components,
            dosage_pca_mean=dosage_pca_mean,
            dosage_pca_std=dosage_pca_std,
            dosage_blend_prior=DOSAGE_BLEND_PRIOR,
            dosage_fixed_weight=DOSAGE_FIXED_WEIGHT,
            dosage_center=DOSAGE_PCA_CENTER,
            dosage_scale=DOSAGE_PCA_SCALE,
            use_biological_aware_embedding=USE_BIOLOGICAL_AWARE_EMBEDDING,
            use_habe=USE_HABE,
            max_sparse_tokens=MAX_SPARSE_TOKENS,
            hotspot_focus_bias=HOTSPOT_FOCUS_BIAS,
            use_env_film=USE_ENV_FILM,
            use_env_pool_bias=USE_ENV_POOL_BIAS,
            use_meta_film=USE_META_FILM,
            meta_film_scale=META_FILM_SCALE,
            low_rank_bilinear_rank=LRBI_RANK,
            use_gxe_moe=USE_GXE_MOE,
            gxe_moe_num_experts=GXE_MOE_NUM_EXPERTS,
            gxe_moe_hidden_dim=GXE_MOE_HIDDEN_DIM,
            gxe_moe_temperature=GXE_MOE_TEMPERATURE,
            interaction_reg_lambda=INTERACTION_REG_LAMBDA,
            n_aux_targets=(len(AUX_TARGETS) if USE_AUX else 0)
        ).to(device)
        model.strict_hotspots = True
        model.hotspot_focus_bias = 0.0
        if hasattr(model, "hotspot_focus_bias_param"):
            model.hotspot_focus_bias_param.data.zero_()
        model.use_functional_pool_bias = False

        # keep sparse branch smaller
        model.max_sparse_tokens = 128
        if hasattr(model, "habe") and hasattr(model.habe, "sparse_injector"):
            model.habe.sparse_injector.max_sparse_tokens = 128
    if USE_DUAL_BRANCH_MODEL:
        model = DualBranchGxE(
            model,
            additive_hidden_dim=ADDITIVE_BRANCH_HIDDEN,
            gate_hidden_dim=DUAL_GATE_HIDDEN,
            gate_dropout=DUAL_GATE_DROPOUT
        ).to(device)
    logging.info(f"Model parameters: {_safe_param_count(model):,}")
    if PRETRAIN_GENOMIC_SIMCLR:
        pretrain_sids = sorted({sample_key_to_sid.get(k, k) for k in train_samples})
        ssl_ds = GenomicOnlyTensorDataset(
            pretrain_sids,
            tensor_dir=TENSOR_DIR,
            feature_dim=genomic_feature_dim_main,
            drop_feature_indices=getattr(train_ds, "drop_feature_indices", []),
            feature_names=getattr(train_ds, "feature_names", [])
        )
        ssl_loader = DataLoader(
            ssl_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_genomic_only,
            pin_memory=False
        )
        logging.info("Running genomic SSL pretraining for %d epochs...", SIMCLR_EPOCHS)
        run_genomic_simclr_pretraining(
            model,
            ssl_loader,
            device,
            epochs=SIMCLR_EPOCHS,
            lr=SIMCLR_LR,
            temp=SIMCLR_TEMP,
            token_drop_p=SIMCLR_TOKEN_DROP,
            feature_noise=SIMCLR_FEATURE_NOISE
        )

    criterion = build_loss_fn(LOSS_FUNCTION)

    optimizer = create_gxe_optimizer(
        model,
        lr=LEARNING_RATE,
        weight_decay=model_weight_decay,
        pop_weight_decay=POP_EMBED_WEIGHT_DECAY,
        metadata_weight_decay=METADATA_WEIGHT_DECAY
    )
    scheduler = build_scheduler(optimizer, max_epochs=NUM_EPOCHS)

    # Train
    best_val_r2 = -1e9
    epochs_since_improve = 0
    history = []
    snapshot_paths = []
    snapshot_val_metrics = None
    snapshot_test_metrics = None

    if USE_RESIDUAL_FOCUS_ARCH and _supports_gxe_stage(model):
        metrics = train_and_eval_two_stage(
            model=model,
            train_eval_loader=train_eval_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            criterion=criterion,
            baseline_lookup=baseline_lookup if not USE_CV else None,
            env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None,
            aux_target_lookup=nonCV_aux_target_lookup if not USE_CV else None,
            aux_loss_weight=(AUX_LOSS_WEIGHT if (USE_AUX and not USE_CV) else 0.0),
            snapshot_cycle_length=SNAPSHOT_CYCLE_LENGTH,
            snapshot_prefix="final",
            monitor_test=False
        )
        best_val_r2 = metrics["val"][1]
        torch.save(model.state_dict(), "best_model_chromomap.pt")
        snapshot_test_metrics = metrics["test"]
        logging.info(f"Two-stage training complete. Best Val RÂ² = {best_val_r2:.4f}")
    else:
        if USE_SNAPSHOT_ENSEMBLE and SNAPSHOT_CYCLE_LENGTH > 0:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        for epoch in range(1, NUM_EPOCHS+1):
            adv_alpha = 0.0
            if USE_ENV_ADVERSARY:
                adv_alpha = min(ENV_ADVERSARY_MAX_ALPHA, epoch / float(max(1, ENV_ADVERSARY_WARMUP_EPOCHS)))
            tr_loss, tr_r2 = train_epoch_regularized(
                model, train_loader, criterion, optimizer, device,
                l1_weight=0.0, mixup_alpha=MIXUP_ALPHA, adv_alpha=adv_alpha,
                aux_target_lookup=nonCV_aux_target_lookup,
                aux_loss_weight=(AUX_LOSS_WEIGHT if USE_AUX else 0.0)
            )
            
            tr_eval_loss, tr_eval_r2, tr_eval_rmse, tr_eval_mae, tr_eval_ccc = evaluate(
            model, train_eval_loader, criterion, device,
            baseline_lookup=baseline_lookup if not USE_CV else None,
            env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
        )
            va_loss, va_r2, va_rmse, va_mae, va_ccc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                baseline_lookup=baseline_lookup if not USE_CV else None,
                env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
            )
            step_scheduler(scheduler, epoch - 1)
            history.append((tr_loss, tr_r2, va_loss, va_r2))
            logging.info(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss {tr_loss:.4f} R2 {tr_r2:.4f} | " f"TrainEval R2 {tr_eval_r2:.4f} RMSE {tr_eval_rmse:.4f} MAE {tr_eval_mae:.4f} CCC {tr_eval_ccc:.4f} |"
                         f"| Val Loss {va_loss:.4f} R2 {va_r2:.4f} RMSE {va_rmse:.4f} MAE {va_mae:.4f} CCC {va_ccc:.4f}")
            if va_r2 > best_val_r2 + EARLY_STOP_MIN_DELTA:
                best_val_r2 = va_r2
                torch.save(model.state_dict(), "best_model_chromomap.pt")
                logging.info(f" âœ“ New best model saved (RÂ² = {va_r2:.4f})")
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if USE_SNAPSHOT_ENSEMBLE and SNAPSHOT_CYCLE_LENGTH > 0 and epoch % SNAPSHOT_CYCLE_LENGTH == 0:
                snapshot_path = os.path.join(SNAPSHOT_DIR, f"final_snapshot_epoch{epoch}.pt")
                torch.save(model.state_dict(), snapshot_path)
                snapshot_paths.append(snapshot_path)

            if epochs_since_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"Early stopping triggered (no val RÂ² improvement in {EARLY_STOP_PATIENCE} epochs).")
                break

        logging.info(f"Done. Best Val RÂ² = {best_val_r2:.4f}")
        if snapshot_paths:
            snapshot_val_metrics = _aggregate_snapshot_metrics(
                model, val_loader, criterion, device, snapshot_paths,
                baseline_lookup=baseline_lookup if not USE_CV else None,
                env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
            )
            snapshot_test_metrics = _aggregate_snapshot_metrics(
                model, test_loader, criterion, device, snapshot_paths,
                baseline_lookup=baseline_lookup if not USE_CV else None,
                env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
            )
            if snapshot_val_metrics:
                logging.info(
                    "Snapshot ensemble val | "
                    f"Loss {snapshot_val_metrics[0]:.4f} R2 {snapshot_val_metrics[1]:.4f} "
                    f"RMSE {snapshot_val_metrics[2]:.4f} MAE {snapshot_val_metrics[3]:.4f} "
                    f"CCC {snapshot_val_metrics[4]:.4f}"
                )
            if snapshot_test_metrics:
                logging.info(
                    "Snapshot ensemble test | "
                    f"Loss {snapshot_test_metrics[0]:.4f} R2 {snapshot_test_metrics[1]:.4f} "
                    f"RMSE {snapshot_test_metrics[2]:.4f} MAE {snapshot_test_metrics[3]:.4f} "
                    f"CCC {snapshot_test_metrics[4]:.4f}"
                )
        if history:
            ep = np.arange(1, len(history)+1)
            tr_l = [h[0] for h in history]; tr_r = [h[1] for h in history]
            va_l = [h[2] for h in history]; va_r = [h[3] for h in history]
            plt.figure(figsize=(6,4))
            plt.plot(ep, tr_l, label="Train Loss")
            plt.plot(ep, va_l, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig("learning_curve_loss.png", dpi=200); plt.close()
            plt.figure(figsize=(6,4))
            plt.plot(ep, tr_r, label="Train R2")
            plt.plot(ep, va_r, label="Val R2")
            plt.xlabel("Epoch")
            plt.ylabel("R2")
            plt.legend()
            plt.tight_layout()
            plt.savefig("learning_curve_r2.png", dpi=200); plt.close()
            logging.info("Saved learning curves (loss and R2).")

    # Ensure best model is loaded for predictions
    if os.path.exists("best_model_chromomap.pt"):
        load_checkpoint_safely(model, "best_model_chromomap.pt", device, allow_shape_mismatch=False)
        if snapshot_test_metrics is None:
            te_loss, te_r2, te_rmse, te_mae, te_ccc = evaluate(
                model, test_loader, criterion, device,
                baseline_lookup=baseline_lookup if not USE_CV else None,
                env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
            )
            logging.info(f"Test | Loss {te_loss:.4f} R2 {te_r2:.4f} RMSE {te_rmse:.4f} MAE {te_mae:.4f} CCC {te_ccc:.4f}")
        else:
            te_loss, te_r2, te_rmse, te_mae, te_ccc = snapshot_test_metrics
            logging.info("Using snapshot ensemble metrics for final test reporting.")
    # Scatter plot of true vs predicted on test set
    preds_rows = predict_with_ids(
        model,
        test_loader,
        device,
        baseline_lookup=baseline_lookup if not USE_CV else None,
        env_target_stats=env_target_stats if use_env_zscore and not USE_CV else None
    )
    log_population_r2(preds_rows, sample_key_to_pop, context="[Final predictions] ")
    if preds_rows:
        y_true = np.array([r["true"] for r in preds_rows], dtype=float)
        y_pred = np.array([r["pred"] for r in preds_rows], dtype=float)
        trait_ceiling = residual_variance_ceiling(y_true, y_pred)
        logging.info(
            "Final test trait ceiling ~= 1 - Var(residual)/Var(y): %.4f",
            trait_ceiling
        )
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
        lims = [
            min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred))
        ]
        plt.plot(lims, lims, 'k--', linewidth=1)
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel("True"); plt.ylabel("Predicted")
        plt.title(f"Test scatter R2={te_r2:.3f} RMSE={te_rmse:.2f} MAE={te_mae:.2f} CCC={te_ccc:.3f}")
        plt.tight_layout()
        plt.savefig("test_scatter.png", dpi=200)
        plt.close()
        logging.info("Saved test scatter plot to test_scatter.png")
        plot_calibration(preds_rows, bins=10, fname="calibration_plot.png")
    # Optional embedding export
    if EXPORT_EMBEDDINGS:
        for view in EMBEDDING_VIEWS:
            if view == "fused":
                out_path = "penultimate_embeddings.csv"
            else:
                out_path = f"penultimate_embeddings_{view}.csv"
            export_penultimate_embeddings(model, test_loader, device, path=out_path, view=view)
            if RUN_EMBED_PLOTS and view in TSNE_EMBEDDING_VIEWS:
                plot_embeddings_tsne(out_path, meta, out_prefix=f"emb_tsne_{view}")
                vecs, targ_vec, _ = collect_embeddings(model, test_loader, device, view=view)
                if vecs.size and targ_vec.size:
                    plot_tsne_basic(vecs, targ_vec, out_prefix=f"tsne_penultimate_{view}")
                    # Pairwise embedding distance vs phenotype difference
                    plot_pairwise_dist_vs_target(
                        vecs,
                        targ_vec,
                        fname=f"pairwise_dist_vs_target_diff_{view}.png"
                    )

    # Analyze model predictions on test set
    pred_analysis = analyze_model_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        target_mean=target_mean if STANDARDIZE_TARGET else None,
        target_std=target_std if STANDARDIZE_TARGET else None
    ) if not USE_TEMPORAL_ENV_ENCODING else {}
if __name__ == "__main__":
    main()
