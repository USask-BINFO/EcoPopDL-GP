#!/usr/bin/env python3
"""
Build manuscript-ready tables and figures from EcoPopDLGP signal-analysis outputs.

What it creates
---------------
Figure 1: Temporal window importance (aggregated across folds/runs, separated by tier)
Figure 2: Stage x feature heatmap (aggregated across folds/runs, separated by tier)
Figure 3: Module/input sensitivity bar plot (aggregated across folds/runs, separated by tier)
Figure 4: Homeolog / subgenome sensitivity bar plot for polyploid runs
          (aggregated across folds/runs, separated by tier; optional)
Table 3: Per-population performance table (aggregated across folds/runs, separated by tier)
Table 2: Full model vs -HABE / -BAE / -Population from retrained ablation runs
         using *_cv_fold_predictions.csv files.

Expected inputs
---------------
1) --signal-root should point to the results_signal directory produced by
   integrated_training_chromomap_D1_signal.py.
   Example contents:
       results_signal/
         tier1_geno_fold1/
           env_window_importance.csv
           env_stage_feature_importance.csv
           module_input_sensitivity.csv
           homeolog_feature_importance.csv   # optional, polyploid runs
           population_metrics.csv
         tier1_geno_fold2/
           ...
         tier2_env_fold1/
           ...

2) For Table 2, pass one directory per experiment run. Each directory should contain
   the prediction files produced by the training script, e.g.:
       tier1_geno_cv_fold_predictions.csv
       tier2_env_cv_fold_predictions.csv

Usage examples
--------------
python paper_results_builder.py \
  --signal-root results_signal \
  --out paper_results

python paper_results_builder.py \
  --signal-root results_signal \
  --full /path/to/full_run \
  --no-habe /path/to/no_habe_run \
  --no-bae /path/to/no_bae_run \
  --no-pop /path/to/no_pop_run \
  --out paper_results
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Small utilities
# -----------------------------

def _safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return text.strip("_") or "unnamed"


def _mean_std_str(mean_val: float, std_val: float, digits: int = 4) -> str:
    if pd.isna(mean_val):
        return "NA"
    if pd.isna(std_val):
        std_val = 0.0
    return f"{mean_val:.{digits}f} +/- {std_val:.{digits}f}"


def _std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1))


def _collect_csvs(root: Path, filename: str) -> List[Path]:
    if root is None:
        return []
    root = Path(root)
    if not root.exists():
        return []
    direct = root / filename
    paths: List[Path] = []
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted(p for p in root.rglob(filename) if p != direct))
    return paths


def _infer_tier_from_parent(parent_name: str) -> str:
    if parent_name in ("", "."):
        return "root"
    m = re.match(r"(.+)_fold\d+$", parent_name)
    if m:
        return m.group(1)
    return parent_name


def _evaluation_label_from_stem(stem: str) -> str:
    clean = stem.replace("_cv_fold_predictions", "")
    mapping = {
        "tier1_geno": "Genotype-CV",
        "tier2_env": "Within-genotype env holdout",
        "root": "Final test split",
    }
    return mapping.get(clean, clean)


# -----------------------------
# Regression metrics for Table 2
# -----------------------------

def _filter_finite_pairs(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(yt) & np.isfinite(yp)
    return yt[keep], yp[keep]


def _r2_score_local(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _filter_finite_pairs(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _rmse_local(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _filter_finite_pairs(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _mae_local(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _filter_finite_pairs(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def _ccc_local(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _filter_finite_pairs(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    mean_t = float(np.mean(yt))
    mean_p = float(np.mean(yp))
    var_t = float(np.var(yt))
    var_p = float(np.var(yp))
    cov = float(np.mean((yt - mean_t) * (yp - mean_p)))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom <= 0:
        return float("nan")
    return float((2.0 * cov) / denom)


def _metric_dict(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    return {
        "r2": _r2_score_local(y_true, y_pred),
        "rmse": _rmse_local(y_true, y_pred),
        "mae": _mae_local(y_true, y_pred),
        "ccc": _ccc_local(y_true, y_pred),
    }


# -----------------------------
# Plotting helpers
# -----------------------------

def _save_window_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    req = {"window_idx", "delta_r2_mean"}
    if df.empty or not req.issubset(df.columns):
        return
    plot_df = df.sort_values("window_idx").copy()
    x = plot_df["window_idx"].astype(int).values + 1
    y = pd.to_numeric(plot_df["delta_r2_mean"], errors="coerce").values
    yerr = pd.to_numeric(plot_df.get("delta_r2_std", 0.0), errors="coerce").fillna(0.0).values if isinstance(plot_df.get("delta_r2_std", 0.0), pd.Series) else np.zeros(len(plot_df))

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(x)
    plt.xlabel("Temporal window")
    plt.ylabel("Mean delta R2 after permutation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_barh(df: pd.DataFrame, label_col: str, value_col: str, err_col: Optional[str], out_path: Path, title: str, xlabel: str, top_k: Optional[int] = None) -> None:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return
    plot_df = df.copy()
    if top_k is not None:
        plot_df = plot_df.head(top_k)
    y = np.arange(len(plot_df))
    vals = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(np.nan).values
    if err_col and err_col in plot_df.columns:
        errs = pd.to_numeric(plot_df[err_col], errors="coerce").fillna(0.0).values
    else:
        errs = np.zeros(len(plot_df))

    plt.figure(figsize=(8, max(4, 0.38 * len(plot_df) + 1.5)))
    plt.barh(y, vals, xerr=errs)
    plt.yticks(y, plot_df[label_col].astype(str).values)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_heatmap(df: pd.DataFrame, out_path: Path, title: str, value_col: str = "delta_r2_mean") -> None:
    req = {"stage", "feature", value_col}
    if df.empty or not req.issubset(df.columns):
        return

    pivot = df.pivot(index="stage", columns="feature", values=value_col)
    if pivot.empty:
        return

    stage_order = [s for s in ["early", "mid", "late"] if s in pivot.index]
    stage_order += [s for s in pivot.index if s not in stage_order]
    preferred_feats = [
        "daylength_h",
        "tmax_C",
        "tmin_C",
        "gdd",
        "vpd_kPa",
        "heat_hdd",
        "cold_cdd",
        "drought_vpd",
        "photo_temp",
        "cum_ptu",
    ]
    feat_order = [f for f in preferred_feats if f in pivot.columns]
    feat_order += [f for f in pivot.columns if f not in feat_order]

    pivot = pivot.loc[stage_order, feat_order]
    arr = pivot.values.astype(float)

    plt.figure(figsize=(max(7, 0.7 * arr.shape[1] + 2), max(3, 0.8 * arr.shape[0] + 1.5)))
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index.astype(str))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean delta R2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Aggregation by tier for Figures 1-3 and Table 3
# -----------------------------

def _read_with_tier(root: Path, filename: str) -> pd.DataFrame:
    rows = []
    for path in _collect_csvs(root, filename):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["_tier"] = _infer_tier_from_parent(path.parent.name)
        df["_source_dir"] = path.parent.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_by_tier(
    df: pd.DataFrame,
    group_cols: List[str],
    numeric_mean_cols: Optional[List[str]] = None,
    sum_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    tiers = list(pd.Series(df["_tier"]).dropna().astype(str).unique())
    for tier in tiers:
        sub = df[df["_tier"] == tier].copy()
        if sub.empty:
            continue

        numeric_cols = [
            c for c in sub.columns
            if c not in set(group_cols + ["_tier", "_source_dir"])
            and pd.api.types.is_numeric_dtype(sub[c])
        ]
        if numeric_mean_cols is not None:
            numeric_cols = [c for c in numeric_cols if c in set(numeric_mean_cols)]

        group = sub.groupby(group_cols, dropna=False)
        agg_frames: List[pd.DataFrame] = []

        if numeric_cols:
            agg = group[numeric_cols].agg(["mean", _std])
            agg.columns = [f"{col}_{stat if stat != '_std' else 'std'}" for col, stat in agg.columns]
            agg = agg.reset_index()
            agg_frames.append(agg)
        else:
            agg_frames.append(group.size().reset_index(name="n_runs"))

        merged = agg_frames[0]
        if "n_runs" not in merged.columns:
            merged = merged.merge(group.size().reset_index(name="n_runs"), on=group_cols, how="left")

        if sum_cols:
            existing_sum_cols = [c for c in sum_cols if c in sub.columns and pd.api.types.is_numeric_dtype(sub[c])]
            if existing_sum_cols:
                sum_df = group[existing_sum_cols].sum().reset_index()
                sum_df = sum_df.rename(columns={c: f"{c}_total" for c in existing_sum_cols})
                merged = merged.merge(sum_df, on=group_cols, how="left")

        out[tier] = merged
    return out


# -----------------------------
# Table 2 from retrained ablations
# -----------------------------

def _summarize_prediction_file(pred_path: Path) -> Optional[Tuple[pd.DataFrame, Dict[str, float]]]:
    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        print(f"Warning: failed to read predictions file {pred_path}: {exc}")
        return None

    needed = {"true", "pred"}
    if df.empty or not needed.issubset(df.columns):
        return None

    fold_col = "fold" if "fold" in df.columns else None
    if fold_col is None:
        df = df.copy()
        df["fold"] = 1
        fold_col = "fold"

    fold_rows = []
    for fold_id, g in df.groupby(fold_col, dropna=False):
        metrics = _metric_dict(g["true"].values, g["pred"].values)
        fold_rows.append({
            "fold": fold_id,
            "n": int(len(g)),
            **metrics,
        })
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return None

    summary = {
        "n_folds": int(len(fold_df)),
        "n_samples_total": int(len(df)),
        "n_samples_mean": float(fold_df["n"].mean()),
        "n_samples_std": _std(fold_df["n"]),
    }
    for metric in ["r2", "rmse", "mae", "ccc"]:
        summary[f"{metric}_mean"] = float(pd.to_numeric(fold_df[metric], errors="coerce").mean())
        summary[f"{metric}_std"] = _std(fold_df[metric])

    return fold_df, summary


def _find_prediction_files(exp_dir: Path) -> List[Path]:
    if exp_dir is None or not exp_dir.exists():
        return []
    patterns = ["*_cv_fold_predictions.csv", "*fold_predictions.csv"]
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(exp_dir.rglob(pat)))
    # unique while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _build_table2(ablation_dirs: Dict[str, Optional[Path]], out_dir: Path) -> Optional[pd.DataFrame]:
    records = []
    per_fold_rows = []

    for exp_label, exp_dir in ablation_dirs.items():
        if exp_dir is None:
            continue
        pred_files = _find_prediction_files(exp_dir)
        if not pred_files:
            print(f"Warning: no prediction files found in {exp_dir} for experiment {exp_label}")
            continue

        for pred_path in pred_files:
            res = _summarize_prediction_file(pred_path)
            if res is None:
                continue
            fold_df, summary = res
            stem = pred_path.stem
            evaluation_key = stem.replace("_cv_fold_predictions", "")
            evaluation = _evaluation_label_from_stem(evaluation_key)

            summary_row = {
                "experiment": exp_label,
                "experiment_dir": str(exp_dir),
                "evaluation_key": evaluation_key,
                "evaluation": evaluation,
                "predictions_file": str(pred_path),
                **summary,
            }
            records.append(summary_row)

            fold_df = fold_df.copy()
            fold_df["experiment"] = exp_label
            fold_df["evaluation_key"] = evaluation_key
            fold_df["evaluation"] = evaluation
            fold_df["predictions_file"] = str(pred_path)
            per_fold_rows.append(fold_df)

    if not records:
        return None

    long_df = pd.DataFrame(records)
    long_df = long_df.sort_values(["evaluation", "experiment"]).reset_index(drop=True)
    long_df.to_csv(out_dir / "table2_ablation_long.csv", index=False)

    if per_fold_rows:
        pd.concat(per_fold_rows, ignore_index=True).to_csv(out_dir / "table2_ablation_per_fold.csv", index=False)

    formatted = long_df.copy()
    formatted["R2"] = formatted.apply(lambda r: _mean_std_str(r["r2_mean"], r["r2_std"]), axis=1)
    formatted["RMSE"] = formatted.apply(lambda r: _mean_std_str(r["rmse_mean"], r["rmse_std"]), axis=1)
    formatted["MAE"] = formatted.apply(lambda r: _mean_std_str(r["mae_mean"], r["mae_std"]), axis=1)
    formatted["CCC"] = formatted.apply(lambda r: _mean_std_str(r["ccc_mean"], r["ccc_std"]), axis=1)
    formatted["n_folds"] = formatted["n_folds"].astype(int)
    formatted["n_samples_total"] = formatted["n_samples_total"].astype(int)
    formatted = formatted[[
        "evaluation",
        "experiment",
        "R2",
        "RMSE",
        "MAE",
        "CCC",
        "n_folds",
        "n_samples_total",
    ]]
    formatted.to_csv(out_dir / "table2_ablation_formatted.csv", index=False)

    for eval_key, sub in long_df.groupby("evaluation_key", dropna=False):
        eval_slug = _safe_slug(str(eval_key))
        sub_fmt = sub.copy()
        sub_fmt["R2"] = sub_fmt.apply(lambda r: _mean_std_str(r["r2_mean"], r["r2_std"]), axis=1)
        sub_fmt["RMSE"] = sub_fmt.apply(lambda r: _mean_std_str(r["rmse_mean"], r["rmse_std"]), axis=1)
        sub_fmt["MAE"] = sub_fmt.apply(lambda r: _mean_std_str(r["mae_mean"], r["mae_std"]), axis=1)
        sub_fmt["CCC"] = sub_fmt.apply(lambda r: _mean_std_str(r["ccc_mean"], r["ccc_std"]), axis=1)
        sub_fmt = sub_fmt[["experiment", "R2", "RMSE", "MAE", "CCC", "n_folds", "n_samples_total"]]
        # preferred row order
        order = {"Full model": 0, "-HABE": 1, "-BAE": 2, "-Population": 3}
        sub_fmt["_order"] = sub_fmt["experiment"].map(order).fillna(99)
        sub_fmt = sub_fmt.sort_values(["_order", "experiment"]).drop(columns="_order")
        sub_fmt.to_csv(out_dir / f"table2_{eval_slug}.csv", index=False)

    return long_df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-ready figures/tables from signal-analysis outputs and ablation runs.")
    ap.add_argument("--signal-root", default=None, help="results_signal directory from the updated training script")
    ap.add_argument("--full", default=None, help="Directory for the full-model run (for Table 2)")
    ap.add_argument("--no-habe", default=None, help="Directory for the -HABE retraining run (for Table 2)")
    ap.add_argument("--no-bae", default=None, help="Directory for the -BAE retraining run (for Table 2)")
    ap.add_argument("--no-pop", default=None, help="Directory for the -Population retraining run (for Table 2)")
    ap.add_argument("--out", default="paper_results", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Figures 1-4 + Table 3 from signal-root -----
    if args.signal_root:
        signal_root = Path(args.signal_root)
        if not signal_root.exists():
            print(f"Warning: signal root does not exist: {signal_root}")
        else:
            # Figure 1: temporal window importance
            window_df = _read_with_tier(signal_root, "env_window_importance.csv")
            for tier, agg in _aggregate_by_tier(window_df, group_cols=["window_idx"], numeric_mean_cols=["delta_r2", "delta_rmse", "delta_mae", "delta_ccc"]).items():
                agg = agg.sort_values("window_idx").reset_index(drop=True)
                tier_slug = _safe_slug(tier)
                agg.to_csv(out_dir / f"figure1_{tier_slug}_temporal_window_importance.csv", index=False)
                _save_window_plot(
                    agg,
                    out_dir / f"figure1_{tier_slug}_temporal_window_importance.png",
                    f"Temporal window importance ({tier})",
                )

            # Figure 2: stage x feature heatmap
            stage_df = _read_with_tier(signal_root, "env_stage_feature_importance.csv")
            for tier, agg in _aggregate_by_tier(stage_df, group_cols=["stage", "feature"], numeric_mean_cols=["delta_r2", "delta_rmse", "delta_mae", "delta_ccc"]).items():
                tier_slug = _safe_slug(tier)
                agg = agg.sort_values(["stage", "feature"]).reset_index(drop=True)
                agg.to_csv(out_dir / f"figure2_{tier_slug}_stage_feature_heatmap.csv", index=False)
                _save_heatmap(
                    agg,
                    out_dir / f"figure2_{tier_slug}_stage_feature_heatmap.png",
                    f"Stage x feature importance ({tier})",
                    value_col="delta_r2_mean",
                )

            # Figure 3: module/input sensitivity
            module_df = _read_with_tier(signal_root, "module_input_sensitivity.csv")
            for tier, agg in _aggregate_by_tier(module_df, group_cols=["component"], numeric_mean_cols=["delta_r2", "delta_rmse", "delta_mae", "delta_ccc"]).items():
                tier_slug = _safe_slug(tier)
                agg = agg.sort_values("delta_r2_mean", ascending=False).reset_index(drop=True)
                agg.to_csv(out_dir / f"figure3_{tier_slug}_module_input_sensitivity.csv", index=False)
                _save_barh(
                    agg,
                    label_col="component",
                    value_col="delta_r2_mean",
                    err_col="delta_r2_std",
                    out_path=out_dir / f"figure3_{tier_slug}_module_input_sensitivity.png",
                    title=f"Module / input sensitivity ({tier})",
                    xlabel="Mean delta R2 after perturbation",
                )

            # Figure 4 (polyploid only): dedicated homeolog / subgenome sensitivity
            homeolog_df = _read_with_tier(signal_root, "homeolog_feature_importance.csv")
            if homeolog_df.empty and (not module_df.empty) and ("component" in module_df.columns):
                homeolog_mask = module_df["component"].astype(str).str.startswith("Homeolog_") | module_df["component"].astype(str).str.startswith("Subgenome_")
                homeolog_df = module_df.loc[homeolog_mask].copy()
            for tier, agg in _aggregate_by_tier(homeolog_df, group_cols=["component"], numeric_mean_cols=["delta_r2", "delta_rmse", "delta_mae", "delta_ccc"]).items():
                tier_slug = _safe_slug(tier)
                agg = agg.sort_values("delta_r2_mean", ascending=False).reset_index(drop=True)
                agg.to_csv(out_dir / f"figure4_{tier_slug}_homeolog_feature_importance.csv", index=False)
                _save_barh(
                    agg,
                    label_col="component",
                    value_col="delta_r2_mean",
                    err_col="delta_r2_std",
                    out_path=out_dir / f"figure4_{tier_slug}_homeolog_feature_importance.png",
                    title=f"Homeolog / subgenome sensitivity ({tier})",
                    xlabel="Mean delta R2 after perturbation",
                )

            # Table 3: per-population performance
            pop_df = _read_with_tier(signal_root, "population_metrics.csv")
            for tier, agg in _aggregate_by_tier(pop_df, group_cols=["population"], numeric_mean_cols=["r2", "rmse", "mae", "ccc", "n"], sum_cols=["n"]).items():
                tier_slug = _safe_slug(tier)
                agg = agg.sort_values("r2_mean", ascending=False).reset_index(drop=True)
                agg.to_csv(out_dir / f"table3_{tier_slug}_per_population_long.csv", index=False)

                fmt = agg.copy()
                fmt["R2"] = fmt.apply(lambda r: _mean_std_str(r["r2_mean"], r["r2_std"]), axis=1)
                fmt["RMSE"] = fmt.apply(lambda r: _mean_std_str(r["rmse_mean"], r["rmse_std"]), axis=1)
                fmt["MAE"] = fmt.apply(lambda r: _mean_std_str(r["mae_mean"], r["mae_std"]), axis=1)
                fmt["CCC"] = fmt.apply(lambda r: _mean_std_str(r["ccc_mean"], r["ccc_std"]), axis=1)
                keep_cols = ["population"]
                if "n_total" in fmt.columns:
                    keep_cols.append("n_total")
                elif "n_mean" in fmt.columns:
                    keep_cols.append("n_mean")
                keep_cols += ["R2", "RMSE", "MAE", "CCC", "n_runs"]
                fmt = fmt[keep_cols]
                fmt.to_csv(out_dir / f"table3_{tier_slug}_per_population_formatted.csv", index=False)

                # optional companion plot
                _save_barh(
                    agg,
                    label_col="population",
                    value_col="r2_mean",
                    err_col="r2_std",
                    out_path=out_dir / f"table3_{tier_slug}_per_population_r2.png",
                    title=f"Per-population predictive performance ({tier})",
                    xlabel="Mean R2",
                )

    # ----- Table 2 from ablation directories -----
    ablation_dirs = {
        "Full model": Path(args.full) if args.full else None,
        "-HABE": Path(args.no_habe) if args.no_habe else None,
        "-BAE": Path(args.no_bae) if args.no_bae else None,
        "-Population": Path(args.no_pop) if args.no_pop else None,
    }
    if any(v is not None for v in ablation_dirs.values()):
        _build_table2(ablation_dirs, out_dir)

    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
