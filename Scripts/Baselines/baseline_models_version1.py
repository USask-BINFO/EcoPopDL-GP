import os
import random
import logging
import time

import numpy as np
import pandas as pd
from pandas_plink import read_plink
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""
Baseline models evaluated with genotype-grouped cross-validation (no genotype leakage).
Inputs (D1 Yield):
  Genotype: plink set at Demo/input_files/Genotype/Axiom_genotype/D1/Genotype_files/imp.qc.all.withdc.clean.*
  Phenotype: yield_mean.txt (FID/IID/Location/Year/SD/Pop/Yield)
Covariates: Location/Year/SD/Pop are present in phenotype but excluded from model features during CV
            to avoid environment leakage; the 120 env features from the D1 matrix are used instead.
Models: GBLUP, RR-BLUP (Ridge) variants, Random Forest, XGBoost (geno and geno+env)
"""

GENO_PREFIX = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Genotype_files/imp.qc.all.withdc.clean.fixed"
PHENO_PATH = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Genotype/Axiom_genotype/D4/Phenotype/Phenotype_files/dtf_mean.txt"
ENV_PATH = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Demo/input_files/Environment/D4/d4_env_matrix_dtf.csv"
N_PCA_COMPONENTS = 200
N_FOLDS = 4
RANDOM_STATE = 20
# For apples-to-apples with ChromoMap CV mode, keep this False (no outer test split).
USE_GROUP_TEST_SPLIT = False
TEST_SIZE = 0.2
REPORT_POOLED_CV_METRICS = True  # pooled (micro) metrics across all CV folds
ADD_GBLUP_WITH_ENV = True  # include env features as fixed effects in GBLUP for fairer comparison
TIER2_TEST_FRAC = 0.2
TIER2_VAL_FRAC = 0.2
TIER2_SEED_OFFSET = 1000
LOG_PATH = "baseline_models_runtime.log"
RUNTIME_LOG_CSV = "baseline_models_runtime_details.csv"
RUNTIME_SUMMARY_CSV = "baseline_models_runtime_summary.csv"


LOGGER = logging.getLogger("baseline_models")


def configure_logging(log_path: str = LOG_PATH) -> None:
    """Log to both console and file so long runs can be inspected later."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def log_info(message: str) -> None:
    if LOGGER.handlers:
        LOGGER.info(message)
    else:
        print(message)


def record_runtime(
    runtime_rows: list,
    stage: str,
    model: str,
    seconds: float,
    fold_label: str = "",
    train_size: int | None = None,
    test_size: int | None = None,
) -> None:
    runtime_rows.append(
        {
            "stage": stage,
            "fold": fold_label,
            "model": model,
            "seconds": seconds,
            "train_size": train_size,
            "test_size": test_size,
        }
    )
    prefix = f"[{fold_label}] " if fold_label else ""
    size_bits = []
    if train_size is not None:
        size_bits.append(f"train={train_size}")
    if test_size is not None:
        size_bits.append(f"test={test_size}")
    size_suffix = f" ({', '.join(size_bits)})" if size_bits else ""
    log_info(f"{prefix}{model} runtime: {seconds:.2f}s{size_suffix}")


def write_runtime_reports(runtime_rows: list) -> None:
    if not runtime_rows:
        log_info("No runtime rows collected; skipping runtime reports.")
        return

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(RUNTIME_LOG_CSV, index=False)

    summary = (
        runtime_df.groupby(["stage", "model"], dropna=False)["seconds"]
        .agg(runs="count", mean_seconds="mean", max_seconds="max", total_seconds="sum")
        .reset_index()
        .sort_values(["stage", "total_seconds"], ascending=[True, False])
    )
    summary.to_csv(RUNTIME_SUMMARY_CSV, index=False)

    log_info(f"Saved runtime details to {RUNTIME_LOG_CSV}")
    log_info(f"Saved runtime summary to {RUNTIME_SUMMARY_CSV}")
    log_info("\n" + summary.to_string(index=False))


def seed_everything(seed: int) -> None:
    """Set RNG seeds for reproducibility where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance correlation coefficient (CCC); more stable than R2 on small samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))
    cov = float(np.cov(y_true, y_pred)[0, 1])
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * cov / denom)


def load_data():
    # Genotypes
    bim, fam, bed = read_plink(GENO_PREFIX, verbose=False)
    X_geno = bed.compute().T  # [samples, snps]
    geno_ids = fam["iid"].astype(str).values

    # Phenotypes
    pheno = pd.read_csv(PHENO_PATH, sep=None, engine="python")
    pheno["IID"] = pheno.get("IID", pheno.get("SampleID", pheno.iloc[:, 1])).astype(str)
    pheno["DTF"] = pd.to_numeric(pheno["DTF"], errors="coerce")
    pheno = pheno.dropna(subset=["DTF"])

    if not {"Location", "Year"}.issubset(pheno.columns):
        raise RuntimeError("Phenotype data must include Location and Year columns for env merge.")

    # Environment matrix
    env_df = pd.read_csv(ENV_PATH)
    env_cols = [c for c in env_df.columns if c.startswith("E_")]
    if not env_cols:
        raise RuntimeError("No environment features found in the D1 environment matrix.")
    if not {"Location", "Year"}.issubset(set(env_df.columns)):
        raise RuntimeError("Environment matrix must contain Location and Year columns for merging.")
    env_df = env_df.drop_duplicates(subset=["Location", "Year"])
    env_df = env_df[["Location", "Year"] + env_cols]

    pheno = pheno.merge(env_df, on=["Location", "Year"], how="left")
    pheno = pheno.dropna(subset=env_cols)

    rows = []
    for _, row in pheno.iterrows():
        iid = str(row["IID"])
        idx = np.where(geno_ids == iid)[0]
        if len(idx) == 0:
            continue
        gvec = X_geno[idx[0]].astype(float)
        env_vec = row[env_cols].values.astype(float)
        rows.append((iid, gvec, env_vec, float(row["DTF"]), str(row["Location"]), str(row["Year"])))

    if not rows:
        raise RuntimeError("No overlapping genotype/phenotype samples found after merging environment data.")

    sample_ids = np.array([r[0] for r in rows])
    X_geno_mat = np.vstack([r[1] for r in rows])
    X_env_mat = np.vstack([r[2] for r in rows])
    y_vec = np.array([r[3] for r in rows], dtype=float)
    loc_arr = np.array([r[4] for r in rows])
    year_arr = np.array([r[5] for r in rows])
    split_keys = np.array([f"{iid}|{loc}|{year}" for iid, loc, year in zip(sample_ids, loc_arr, year_arr)])
    return sample_ids, X_geno_mat, X_env_mat, y_vec, loc_arr, year_arr, split_keys


def gblup_predict(X_train, y_train, X_test, cov_train=None, cov_test=None, lambda_g=1e-3):
    """GBLUP with explicit ridge regularization and optional fixed covariates."""
    mu = np.mean(X_train, axis=0, keepdims=True)
    M_train = X_train - mu
    M_test = X_test - mu
    n_snps = max(1, M_train.shape[1])

    G_train = M_train @ M_train.T / n_snps
    K_test_train = M_test @ M_train.T / n_snps

    G_reg = G_train + lambda_g * np.eye(len(y_train))
    V = G_reg + np.eye(len(y_train))
    V_inv = np.linalg.pinv(V)

    if cov_train is not None:
        Xf_tr = np.column_stack([np.ones((len(y_train), 1)), cov_train])
        Xf_te = np.column_stack([np.ones((len(X_test), 1)), cov_test])
    else:
        Xf_tr = np.ones((len(y_train), 1))
        Xf_te = np.ones((len(X_test), 1))

    beta = np.linalg.pinv(Xf_tr.T @ V_inv @ Xf_tr) @ (Xf_tr.T @ V_inv @ y_train)
    resid = y_train - Xf_tr @ beta
    u = G_reg @ V_inv @ resid
    y_hat = Xf_te @ beta + K_test_train @ V_inv @ u
    return y_hat.ravel()


def pca_ridge_predict(X_g_tr_s, X_g_te_s, X_e_tr_s, X_e_te_s, y_train_z, n_pcs):
    """Fit PCA on standardized genotypes, then Ridge on PCs + env covariates."""
    pca = PCA(n_components=n_pcs)
    pcs_tr = pca.fit_transform(X_g_tr_s)
    pcs_te = pca.transform(X_g_te_s)
    X_pca_tr = np.hstack([pcs_tr, X_e_tr_s])
    X_pca_te = np.hstack([pcs_te, X_e_te_s])
    ridge_pca = Ridge(alpha=1.0)
    ridge_pca.fit(X_pca_tr, y_train_z)
    return ridge_pca.predict(X_pca_te)


def _standardize_target(y: np.ndarray):
    mean = float(np.mean(y))
    std = float(np.std(y))
    std = std if std > 1e-8 else 1.0
    return (y - mean) / std, mean, std


def build_within_genotype_env_folds(
    genotypes: np.ndarray,
    n_folds: int = N_FOLDS,
    test_frac: float = TIER2_TEST_FRAC,
    val_frac: float = TIER2_VAL_FRAC,
    seed: int = RANDOM_STATE + TIER2_SEED_OFFSET,
):
    """
    Split rows within each genotype into train/val/test for env-holdout CV.
    """
    genotypes = np.asarray(genotypes)
    unique_genos = np.unique(genotypes)
    folds = []
    for fold in range(n_folds):
        rng = np.random.RandomState(seed + fold * 101)
        tr_idx, va_idx, te_idx = [], [], []
        for gid in unique_genos:
            idxs = np.where(genotypes == gid)[0]
            if idxs.size == 0:
                continue
            perm = idxs.copy()
            rng.shuffle(perm)
            n = len(perm)
            if n < 2:
                tr_idx.extend(perm.tolist())
                continue
            n_test = max(1, int(np.ceil(test_frac * n)))
            n_val = max(0, int(np.ceil(val_frac * n)))
            if n_test + n_val >= n:
                n_val = min(n_val, n - n_test - 1)
                if n_val < 0:
                    n_val = 0
                if n_test + n_val >= n:
                    n_test = max(1, n - n_val - 1)
            test_i = perm[:n_test]
            val_i = perm[n_test:n_test + n_val]
            train_i = perm[n_test + n_val:]
            if train_i.size == 0:
                train_i = test_i[:1]
                test_i = test_i[1:]
            tr_idx.extend(train_i.tolist())
            va_idx.extend(val_i.tolist())
            te_idx.extend(test_i.tolist())
        folds.append(
            (
                np.array(tr_idx, dtype=int),
                np.array(va_idx, dtype=int),
                np.array(te_idx, dtype=int),
            )
        )
    return folds


def _run_fold_models(
    X_g: np.ndarray,
    X_e: np.ndarray,
    y_vec: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    results: dict,
    pooled: dict,
    fold_label: str,
    runtime_rows: list,
    runtime_stage: str,
):
    fold_start = time.perf_counter()
    if train_idx.size == 0 or test_idx.size == 0:
        log_info(f"  [{fold_label}] Skipped (empty train or test split).")
        return

    X_g_tr, X_g_te = X_g[train_idx], X_g[test_idx]
    X_e_tr, X_e_te = X_e[train_idx], X_e[test_idx]
    y_tr, y_te = y_vec[train_idx], y_vec[test_idx]
    if y_tr.size == 0 or y_te.size == 0:
        log_info(f"  [{fold_label}] Skipped (empty targets).")
        return

    y_tr_z, y_mean, y_std = _standardize_target(y_tr)

    scaler_g = StandardScaler()
    scaler_e = StandardScaler()

    X_g_tr_s = scaler_g.fit_transform(X_g_tr)
    X_g_te_s = scaler_g.transform(X_g_te)
    X_e_tr_s = scaler_e.fit_transform(X_e_tr)
    X_e_te_s = scaler_e.transform(X_e_te)

    def record(name, y_pred):
        r2 = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        ccc = concordance_correlation_coefficient(y_te, y_pred)
        results.setdefault(name, {"r2": [], "rmse": [], "mae": [], "ccc": []})
        results[name]["r2"].append(r2)
        results[name]["rmse"].append(rmse)
        results[name]["mae"].append(mae)
        results[name]["ccc"].append(ccc)
        pooled.setdefault(name, {"y_true": [], "y_pred": []})
        pooled[name]["y_true"].append(y_te)
        pooled[name]["y_pred"].append(y_pred)
        log_info(f"  [{fold_label}] {name}: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, CCC={ccc:.4f}")

    def destandardize(pred_z):
        return pred_z * y_std + y_mean

    def run_model(name, fn):
        model_start = time.perf_counter()
        y_pred = fn()
        record_runtime(
            runtime_rows,
            stage=runtime_stage,
            model=name,
            seconds=time.perf_counter() - model_start,
            fold_label=fold_label,
            train_size=len(train_idx),
            test_size=len(test_idx),
        )
        record(name, y_pred)

    # GBLUP (geno)
    run_model("GBLUP (geno)", lambda: destandardize(gblup_predict(X_g_tr_s, y_tr_z, X_g_te_s)))
    if ADD_GBLUP_WITH_ENV:
        run_model(
            "GBLUP (geno + env)",
            lambda: destandardize(gblup_predict(X_g_tr_s, y_tr_z, X_g_te_s, X_e_tr_s, X_e_te_s)),
        )

    # Ridge: geno only
    ridge_g = Ridge(alpha=1.0)
    run_model(
        "RR-BLUP (geno)",
        lambda: (
            ridge_g.fit(X_g_tr_s, y_tr_z),
            destandardize(ridge_g.predict(X_g_te_s)),
        )[1],
    )

    # Ridge: geno + env
    X_ge_tr = np.hstack([X_g_tr_s, X_e_tr_s])
    X_ge_te = np.hstack([X_g_te_s, X_e_te_s])
    ridge_ge = Ridge(alpha=1.0)
    run_model(
        "Ridge (geno + env)",
        lambda: (
            ridge_ge.fit(X_ge_tr, y_tr_z),
            destandardize(ridge_ge.predict(X_ge_te)),
        )[1],
    )

    # Random Forest: geno only (no feature scaling for tree models)
    rf_g = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    run_model(
        "RF (geno)",
        lambda: (
            rf_g.fit(X_g_tr, y_tr_z),
            destandardize(rf_g.predict(X_g_te)),
        )[1],
    )

    # Random Forest: geno + env
    X_ge_tr_raw = np.hstack([X_g_tr, X_e_tr])
    X_ge_te_raw = np.hstack([X_g_te, X_e_te])
    rf_ge = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    run_model(
        "RF (geno + env)",
        lambda: (
            rf_ge.fit(X_ge_tr_raw, y_tr_z),
            destandardize(rf_ge.predict(X_ge_te_raw)),
        )[1],
    )

    # XGBoost: geno only
    xgb_g = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    run_model(
        "XGB (geno)",
        lambda: (
            xgb_g.fit(X_g_tr, y_tr_z),
            destandardize(xgb_g.predict(X_g_te)),
        )[1],
    )

    # XGBoost: geno + env
    xgb_ge = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    run_model(
        "XGB (geno + env)",
        lambda: (
            xgb_ge.fit(X_ge_tr_raw, y_tr_z),
            destandardize(xgb_ge.predict(X_ge_te_raw)),
        )[1],
    )

    # Ridge on genomic PCs + env
    n_pcs = min(N_PCA_COMPONENTS, X_g_tr_s.shape[0], X_g_tr_s.shape[1])
    if n_pcs < 1:
        log_info(f"  [{fold_label}] Skipped PCA ridge (insufficient samples/features).")
    else:
        run_model(
            f"Ridge ({n_pcs} PCs + env)",
            lambda: destandardize(
                pca_ridge_predict(X_g_tr_s, X_g_te_s, X_e_tr_s, X_e_te_s, y_tr_z, n_pcs)
            ),
        )

    record_runtime(
        runtime_rows,
        stage=f"{runtime_stage}_total",
        model="All models",
        seconds=time.perf_counter() - fold_start,
        fold_label=fold_label,
        train_size=len(train_idx),
        test_size=len(test_idx),
    )


def _summarize_results(results: dict) -> pd.DataFrame:
    summary_rows = []
    for name, vals in results.items():
        summary_rows.append({
            "model": name,
            "cv_r2_mean": np.mean(vals["r2"]),
            "cv_r2_std": np.std(vals["r2"]),
            "cv_rmse_mean": np.mean(vals["rmse"]),
            "cv_rmse_std": np.std(vals["rmse"]),
            "cv_mae_mean": np.mean(vals["mae"]),
            "cv_mae_std": np.std(vals["mae"]),
            "cv_ccc_mean": np.mean(vals["ccc"]),
            "cv_ccc_std": np.std(vals["ccc"]),
        })
    if not summary_rows:
        return pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return summary.sort_values("cv_r2_mean", ascending=False)


def _write_pooled_metrics(pooled: dict, filename: str):
    pooled_rows = []
    for name, vals in pooled.items():
        y_true_all = np.concatenate(vals["y_true"]) if vals["y_true"] else np.array([])
        y_pred_all = np.concatenate(vals["y_pred"]) if vals["y_pred"] else np.array([])
        if y_true_all.size == 0:
            continue
        pooled_rows.append({
            "model": name,
            "cv_r2_pooled": r2_score(y_true_all, y_pred_all),
            "cv_rmse_pooled": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
            "cv_mae_pooled": mean_absolute_error(y_true_all, y_pred_all),
            "cv_ccc_pooled": concordance_correlation_coefficient(y_true_all, y_pred_all),
        })
    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows).sort_values("cv_r2_pooled", ascending=False)
        pooled_df.to_csv(filename, index=False)
        log_info("\n" + "=" * 80)
        log_info(f"POOLED CV RESULTS ({filename})")
        log_info("=" * 80)
        log_info(pooled_df.to_string(index=False))
        log_info("=" * 80)


def eval_folds(sample_ids, X_geno, X_env, y_vec, runtime_rows):
    """Evaluate models with genotype-grouped CV using geno/env features only (Tier-1)."""
    unique_genos, genotype_groups = np.unique(sample_ids, return_inverse=True)
    results = {}
    pooled = {}

    # Optional genotype-disjoint test split
    if USE_GROUP_TEST_SPLIT:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X_geno, y_vec, groups=genotype_groups))
        X_g_tr_full, X_g_te_hold = X_geno[train_idx], X_geno[test_idx]
        X_e_tr_full, X_e_te_hold = X_env[train_idx], X_env[test_idx]
        y_tr_full, y_te_hold = y_vec[train_idx], y_vec[test_idx]
        genotypes_tr_full = genotype_groups[train_idx]
        log_info(
            f"\nHeld-out test split: train {len(train_idx)} samples / test {len(test_idx)} samples "
            f"({len(np.unique(genotypes_tr_full))} train genotypes, {len(np.unique(genotype_groups[test_idx]))} test genotypes)"
        )
    else:
        X_g_tr_full, X_g_te_hold = X_geno, None
        X_e_tr_full, X_e_te_hold = X_env, None
        y_tr_full, y_te_hold = y_vec, None
        genotypes_tr_full = genotype_groups

    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_g_tr_full, y_tr_full, groups=genotypes_tr_full), start=1):
        log_info(f"\n=== Tier-1 Fold {fold}/{N_FOLDS} (genotype CV) ===")
        _run_fold_models(
            X_g_tr_full,
            X_e_tr_full,
            y_tr_full,
            tr_idx,
            te_idx,
            results,
            pooled,
            fold_label=f"Tier-1 Fold {fold}",
            runtime_rows=runtime_rows,
            runtime_stage="tier1_cv_model",
        )

    summary = _summarize_results(results)
    if summary.empty:
        log_info("No Tier-1 CV results to summarize (empty splits).")
        return
    summary.to_csv("baseline_tier1_cv_results.csv", index=False)

    def _fmt(mean_val: float, std_val: float, decimals: int) -> str:
        return f"{mean_val:.{decimals}f} +/- {std_val:.{decimals}f}"

    summary_pretty = pd.DataFrame({
        "model": summary["model"],
        "cv_r2": [
            _fmt(m, s, 4) for m, s in zip(summary["cv_r2_mean"], summary["cv_r2_std"])
        ],
        "cv_rmse": [
            _fmt(m, s, 2) for m, s in zip(summary["cv_rmse_mean"], summary["cv_rmse_std"])
        ],
        "cv_mae": [
            _fmt(m, s, 2) for m, s in zip(summary["cv_mae_mean"], summary["cv_mae_std"])
        ],
        "cv_ccc": [
            _fmt(m, s, 4) for m, s in zip(summary["cv_ccc_mean"], summary["cv_ccc_std"])
        ],
    })

    log_info("\n" + "=" * 80)
    log_info("TIER-1 CV RESULTS (GENOTYPE-BASED, mean +/- std)")
    log_info("=" * 80)
    log_info(summary_pretty.to_string(index=False))
    log_info("=" * 80)

    if REPORT_POOLED_CV_METRICS and pooled:
        _write_pooled_metrics(pooled, "baseline_tier1_cv_pooled_results.csv")

    # Held-out test evaluation (genotype-disjoint) for apples-to-apples reporting
    if USE_GROUP_TEST_SPLIT and X_g_te_hold is not None:
        log_info("\n" + "=" * 80)
        log_info("HELD-OUT TEST RESULTS (GENOTYPE-DISJOINT)")
        log_info("=" * 80)
        holdout_start = time.perf_counter()
        y_tr_z_full, y_mean_full, y_std_full = _standardize_target(y_tr_full)
        scaler_g = StandardScaler().fit(X_g_tr_full)
        scaler_e = StandardScaler().fit(X_e_tr_full)

        X_g_tr_s = scaler_g.transform(X_g_tr_full)
        X_g_te_s = scaler_g.transform(X_g_te_hold)
        X_e_tr_s = scaler_e.transform(X_e_tr_full)
        X_e_te_s = scaler_e.transform(X_e_te_hold)

        test_rows = []

        def destandardize_test(pred_z):
            return pred_z * y_std_full + y_mean_full

        def record_test(name, y_pred):
            r2 = r2_score(y_te_hold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te_hold, y_pred))
            mae = mean_absolute_error(y_te_hold, y_pred)
            ccc = concordance_correlation_coefficient(y_te_hold, y_pred)
            test_rows.append({"model": name, "test_r2": r2, "test_rmse": rmse, "test_mae": mae, "test_ccc": ccc})
            log_info(f"  {name}: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, CCC={ccc:.4f}")

        def run_holdout_model(name, fn):
            model_start = time.perf_counter()
            y_pred = fn()
            record_runtime(
                runtime_rows,
                stage="tier1_holdout_model",
                model=name,
                seconds=time.perf_counter() - model_start,
                fold_label="Tier-1 Holdout",
                train_size=len(y_tr_full),
                test_size=len(y_te_hold),
            )
            record_test(name, y_pred)

        # GBLUP (geno)
        run_holdout_model("GBLUP (geno)", lambda: destandardize_test(gblup_predict(X_g_tr_s, y_tr_z_full, X_g_te_s)))
        if ADD_GBLUP_WITH_ENV:
            run_holdout_model(
                "GBLUP (geno + env)",
                lambda: destandardize_test(gblup_predict(X_g_tr_s, y_tr_z_full, X_g_te_s, X_e_tr_s, X_e_te_s)),
            )

        # Ridge: geno only
        ridge_g = Ridge(alpha=1.0)
        run_holdout_model(
            "RR-BLUP (geno)",
            lambda: (
                ridge_g.fit(X_g_tr_s, y_tr_z_full),
                destandardize_test(ridge_g.predict(X_g_te_s)),
            )[1],
        )

        # Ridge: geno + env
        X_ge_tr = np.hstack([X_g_tr_s, X_e_tr_s])
        X_ge_te = np.hstack([X_g_te_s, X_e_te_s])
        ridge_ge = Ridge(alpha=1.0)
        run_holdout_model(
            "Ridge (geno + env)",
            lambda: (
                ridge_ge.fit(X_ge_tr, y_tr_z_full),
                destandardize_test(ridge_ge.predict(X_ge_te)),
            )[1],
        )

        # Random Forest: geno only (no feature scaling for tree models)
        rf_g = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        run_holdout_model(
            "RF (geno)",
            lambda: (
                rf_g.fit(X_g_tr_full, y_tr_z_full),
                destandardize_test(rf_g.predict(X_g_te_hold)),
            )[1],
        )

        # Random Forest: geno + env
        X_ge_tr_raw = np.hstack([X_g_tr_full, X_e_tr_full])
        X_ge_te_raw = np.hstack([X_g_te_hold, X_e_te_hold])
        rf_ge = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        run_holdout_model(
            "RF (geno + env)",
            lambda: (
                rf_ge.fit(X_ge_tr_raw, y_tr_z_full),
                destandardize_test(rf_ge.predict(X_ge_te_raw)),
            )[1],
        )

        # XGBoost: geno only
        xgb_g = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        run_holdout_model(
            "XGB (geno)",
            lambda: (
                xgb_g.fit(X_g_tr_full, y_tr_z_full),
                destandardize_test(xgb_g.predict(X_g_te_hold)),
            )[1],
        )

        # XGBoost: geno + env
        xgb_ge = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        run_holdout_model(
            "XGB (geno + env)",
            lambda: (
                xgb_ge.fit(X_ge_tr_raw, y_tr_z_full),
                destandardize_test(xgb_ge.predict(X_ge_te_raw)),
            )[1],
        )

        # Ridge on genomic PCs + env
        n_pcs = min(N_PCA_COMPONENTS, X_g_tr_s.shape[0], X_g_tr_s.shape[1])
        if n_pcs >= 1:
            run_holdout_model(
                f"Ridge ({n_pcs} PCs + env)",
                lambda: destandardize_test(
                    pca_ridge_predict(X_g_tr_s, X_g_te_s, X_e_tr_s, X_e_te_s, y_tr_z_full, n_pcs)
                ),
            )

        if test_rows:
            pd.DataFrame(test_rows).to_csv("baseline_tier1_test_results.csv", index=False)
            log_info("\nSaved held-out test metrics to baseline_tier1_test_results.csv")
        record_runtime(
            runtime_rows,
            stage="tier1_holdout_total",
            model="All models",
            seconds=time.perf_counter() - holdout_start,
            fold_label="Tier-1 Holdout",
            train_size=len(y_tr_full),
            test_size=len(y_te_hold),
        )


def eval_within_genotype_env_folds(sample_ids, X_geno, X_env, y_vec, runtime_rows, split_keys=None):
    """Tier-2: within-genotype environment holdout (seen genotypes, unseen env rows)."""
    results = {}
    pooled = {}
    if split_keys is not None and len(split_keys) != len(sample_ids):
        log_info("Warning: split_keys length mismatch; ignoring split_keys for Tier-2 CV.")
        split_keys = None
    if split_keys is not None:
        unique_split_keys = len(np.unique(split_keys))
        log_info(f"\nTier-2 SplitKey rows: {unique_split_keys} unique IID|Location|Year combinations")
    folds = build_within_genotype_env_folds(
        genotypes=sample_ids,
        n_folds=N_FOLDS,
        test_frac=TIER2_TEST_FRAC,
        val_frac=TIER2_VAL_FRAC,
        seed=RANDOM_STATE + TIER2_SEED_OFFSET,
    )
    for fold_idx, (tr_idx, va_idx, te_idx) in enumerate(folds, start=1):
        train_count, val_count, test_count = len(tr_idx), len(va_idx), len(te_idx)
        log_info(f"\n=== Tier-2 Fold {fold_idx}/{N_FOLDS} (within-genotype env holdout) ===")
        log_info(f"  Split sizes -> train={train_count}, val={val_count}, test={test_count}")
        _run_fold_models(
            X_geno,
            X_env,
            y_vec,
            tr_idx,
            te_idx,
            results,
            pooled,
            fold_label=f"Tier-2 Fold {fold_idx}",
            runtime_rows=runtime_rows,
            runtime_stage="tier2_cv_model",
        )

    summary = _summarize_results(results)
    if summary.empty:
        log_info("No Tier-2 CV results to summarize (empty splits).")
        return
    summary.to_csv("baseline_tier2_cv_results.csv", index=False)

    def _fmt(mean_val: float, std_val: float, decimals: int) -> str:
        return f"{mean_val:.{decimals}f} +/- {std_val:.{decimals}f}"

    summary_pretty = pd.DataFrame({
        "model": summary["model"],
        "cv_r2": [
            _fmt(m, s, 4) for m, s in zip(summary["cv_r2_mean"], summary["cv_r2_std"])
        ],
        "cv_rmse": [
            _fmt(m, s, 2) for m, s in zip(summary["cv_rmse_mean"], summary["cv_rmse_std"])
        ],
        "cv_mae": [
            _fmt(m, s, 2) for m, s in zip(summary["cv_mae_mean"], summary["cv_mae_std"])
        ],
        "cv_ccc": [
            _fmt(m, s, 4) for m, s in zip(summary["cv_ccc_mean"], summary["cv_ccc_std"])
        ],
    })

    log_info("\n" + "=" * 80)
    log_info("TIER-2 CV RESULTS (WITHIN-GENOTYPE ENV HOLDOUT, mean +/- std)")
    log_info("=" * 80)
    log_info(summary_pretty.to_string(index=False))
    log_info("=" * 80)

    if REPORT_POOLED_CV_METRICS and pooled:
        _write_pooled_metrics(pooled, "baseline_tier2_cv_pooled_results.csv")


if __name__ == "__main__":
    configure_logging()
    seed_everything(RANDOM_STATE)
    runtime_rows = []
    script_start = time.perf_counter()

    load_start = time.perf_counter()
    sample_ids, X_geno, X_env, y_vec, loc_arr, year_arr, split_keys = load_data()
    record_runtime(runtime_rows, stage="setup", model="load_data", seconds=time.perf_counter() - load_start)
    log_info(f"Loaded {len(y_vec)} phenotype rows with genotype + environment overlap.")
    log_info(f"Genotypes: {X_geno.shape[1]} markers, Envs: {X_env.shape[1]} features")

    tier1_start = time.perf_counter()
    eval_folds(sample_ids, X_geno, X_env, y_vec, runtime_rows=runtime_rows)
    record_runtime(runtime_rows, stage="section_total", model="Tier-1 CV", seconds=time.perf_counter() - tier1_start)

    tier2_start = time.perf_counter()
    eval_within_genotype_env_folds(sample_ids, X_geno, X_env, y_vec, runtime_rows=runtime_rows, split_keys=split_keys)
    record_runtime(runtime_rows, stage="section_total", model="Tier-2 CV", seconds=time.perf_counter() - tier2_start)

    record_runtime(runtime_rows, stage="section_total", model="Entire script", seconds=time.perf_counter() - script_start)
    write_runtime_reports(runtime_rows)
