"""
Environment-blocked (leave-one-environment-out) cross-validation splitters.

Addresses the reviewer request for a STRICTER test of generalization to truly
unseen environments than the within-genotype environment-holdout (env-CV) used
in the submitted version. Here a whole LOCATION, a whole YEAR, or a whole
LOCATION x YEAR is held out for testing, so that environment is entirely absent
from training.

Returns the same (train_keys, val_keys, test_keys) triplet format as
run_genotype_cv / run_within_genotype_env_holdout_cv, so it plugs directly into
the existing fold loop.

Note on validation: the test fold is a clean, fully held-out environment. The
validation split is carved as a random fraction of the remaining (training)
observations and is used only for early stopping / model selection. This mirrors
how val is treated in the existing geno-CV (val overlaps train groups; only the
TEST group is guaranteed unseen).
"""

from typing import List, Tuple
import logging
import random

import pandas as pd


_POP_COL_CANDIDATES = ("PopID", "Pop_Code", "Pop", "Assigned_Cluster", "population")


def _resolve_pop_col(meta, pop_col):
    if pop_col:
        return pop_col
    for c in _POP_COL_CANDIDATES:
        if c in meta.columns:
            return c
    raise KeyError(
        "run_environment_blocked_cv(block_by='population') could not find a population column; "
        f"looked for {list(_POP_COL_CANDIDATES)}. Pass pop_col explicitly."
    )


def _env_label(row, block_by: str, location_col: str, year_col: str, pop_col=None) -> str:
    if block_by == "population":
        return str(row[pop_col])
    loc = str(row[location_col])
    yr = str(row[year_col])
    if block_by == "location":
        return loc
    if block_by == "year":
        return yr
    if block_by == "loc_year":
        return f"{loc}||{yr}"
    raise ValueError(f"block_by must be 'location'|'year'|'loc_year'|'population', got: {block_by}")


def run_environment_blocked_cv(
    meta: pd.DataFrame,
    split_key_col: str,
    sampleid_col: str,
    block_by: str = "location",
    location_col: str = "Location",
    year_col: str = "Year",
    pop_col: str = None,
    val_frac: float = 0.15,
    seed: int = 20,
    min_test_obs: int = 1,
    min_train_obs: int = 1,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Leave-one-environment-out CV.

    One fold per unique environment (location, year, or location x year). All
    observations belonging to the held-out environment become the test set; the
    genotype MAY still appear in training through OTHER environments, but the
    held-out environment itself is never seen in training.

    Args:
        meta: metadata DataFrame with one row per genotype x environment observation.
        split_key_col: column with the per-observation key used by the datasets
                       (the same keys returned to the fold loop).
        sampleid_col: genotype/line ID column (kept for logging / sanity checks).
        block_by: 'location' | 'year' | 'loc_year'.
        location_col, year_col: environment-defining columns in meta.
        val_frac: fraction of the remaining (non-test) observations used for validation.
        seed: base RNG seed (offset per fold for reproducibility).
        min_test_obs, min_train_obs: skip a fold if it is degenerate.

    Returns:
        List of (train_keys, val_keys, test_keys), one entry per usable environment.
    """
    if block_by == "population":
        pop_col = _resolve_pop_col(meta, pop_col)
        needed = (split_key_col, sampleid_col, pop_col)
    else:
        needed = (split_key_col, sampleid_col, location_col, year_col)
    for col in needed:
        if col not in meta.columns:
            raise KeyError(f"run_environment_blocked_cv: column '{col}' not in metadata.")

    meta2 = meta.copy()
    for col in needed:
        meta2[col] = meta2[col].astype(str)

    if meta2[split_key_col].duplicated().any():
        n_dup = int(meta2[split_key_col].duplicated().sum())
        logging.warning(
            "run_environment_blocked_cv: %d duplicate split keys; keeping first occurrence.",
            n_dup,
        )
        meta2 = meta2.drop_duplicates(subset=[split_key_col], keep="first").reset_index(drop=True)

    meta2["_env"] = meta2.apply(
        lambda r: _env_label(r, block_by, location_col, year_col, pop_col), axis=1
    )
    unique_envs = sorted(meta2["_env"].unique())
    if len(unique_envs) < 2:
        raise RuntimeError(
            f"Environment-blocked CV needs >=2 distinct environments for block_by='{block_by}', "
            f"found {len(unique_envs)}."
        )

    fold_triplets: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_idx, env in enumerate(unique_envs, start=1):
        is_test = meta2["_env"] == env
        test_keys = meta2.loc[is_test, split_key_col].tolist()
        trainval_keys = meta2.loc[~is_test, split_key_col].tolist()

        if len(test_keys) < min_test_obs or len(trainval_keys) < min_train_obs:
            logging.warning(
                "Skipping environment '%s' (test=%d, trainval=%d): below minimum sizes.",
                env, len(test_keys), len(trainval_keys),
            )
            continue

        rng = random.Random(seed + fold_idx)
        shuffled = list(trainval_keys)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_frac * len(shuffled)))) if val_frac > 0 else 0
        n_val = min(n_val, len(shuffled) - 1)  # always leave >=1 for training
        val_keys = shuffled[:n_val]
        train_keys = shuffled[n_val:]

        n_test_genos = meta2.loc[is_test, sampleid_col].nunique()
        logging.info(
            "[env-blocked:%s] fold %d/%d held-out env='%s' | train=%d val=%d test=%d (test genotypes=%d)",
            block_by, fold_idx, len(unique_envs), env,
            len(train_keys), len(val_keys), len(test_keys), n_test_genos,
        )
        fold_triplets.append((train_keys, val_keys, test_keys))

    if not fold_triplets:
        raise RuntimeError(
            f"Environment-blocked CV produced no usable folds for block_by='{block_by}'."
        )
    logging.info(
        "Environment-blocked CV (%s): %d folds over %d environments.",
        block_by, len(fold_triplets), len(unique_envs),
    )
    return fold_triplets
