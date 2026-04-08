import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _collect_csvs(root: Path, filename: str) -> List[Path]:
    paths = []
    direct = root / filename
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted(p for p in root.rglob(filename) if p != direct))
    return paths


def _aggregate_table(root: Path, filename: str, group_cols: List[str], sort_col: Optional[str] = None) -> pd.DataFrame:
    files = _collect_csvs(root, filename)
    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["source_dir"] = path.parent.name
            dfs.append(df)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
    if not dfs:
        return pd.DataFrame()
    full = pd.concat(dfs, ignore_index=True)
    numeric_cols = [c for c in full.columns if c not in set(group_cols + ["source_dir"]) and pd.api.types.is_numeric_dtype(full[c])]
    agg_spec: Dict[str, List[str]] = {c: ["mean", "std"] for c in numeric_cols}
    out = full.groupby(group_cols, dropna=False).agg(agg_spec)
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out = out.reset_index()
    out["n_runs"] = full.groupby(group_cols, dropna=False).size().values
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out


def _aggregate_homeolog_table(root: Path, sort_col: str = "delta_r2_mean") -> pd.DataFrame:
    homeolog_files = _collect_csvs(root, "homeolog_feature_importance.csv")
    dfs = []

    if homeolog_files:
        candidate_files = homeolog_files
        filter_homeolog = False
    else:
        candidate_files = _collect_csvs(root, "module_input_sensitivity.csv")
        filter_homeolog = True

    for path in candidate_files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            if filter_homeolog:
                if "component" not in df.columns:
                    continue
                mask = df["component"].astype(str).str.startswith("Homeolog_") | df["component"].astype(str).str.startswith("Subgenome_")
                df = df.loc[mask].copy()
                if df.empty:
                    continue
            df["source_dir"] = path.parent.name
            dfs.append(df)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")

    if not dfs:
        return pd.DataFrame()

    full = pd.concat(dfs, ignore_index=True)
    group_cols = ["component"]
    numeric_cols = [c for c in full.columns if c not in set(group_cols + ["source_dir"]) and pd.api.types.is_numeric_dtype(full[c])]
    agg_spec: Dict[str, List[str]] = {c: ["mean", "std"] for c in numeric_cols}
    out = full.groupby(group_cols, dropna=False).agg(agg_spec)
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out = out.reset_index()
    out["n_runs"] = full.groupby(group_cols, dropna=False).size().values
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out


def _save_barplot(df: pd.DataFrame, label_col: str, value_col: str, out_path: Path, title: str, top_k: int = 15):
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return
    plot_df = df.copy().head(top_k)
    plt.figure(figsize=(8, max(4, 0.35 * len(plot_df) + 1.5)))
    y = np.arange(len(plot_df))
    vals = plot_df[value_col].astype(float).values
    errs = plot_df.get(value_col.replace("_mean", "_std"), pd.Series(np.zeros(len(plot_df)))).astype(float).values
    plt.barh(y, vals, xerr=errs)
    plt.yticks(y, plot_df[label_col].astype(str).values)
    plt.gca().invert_yaxis()
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Aggregate signal-analysis CSVs across CV folds / runs.")
    ap.add_argument("--root", default="results_signal", help="Root directory containing signal-analysis outputs.")
    ap.add_argument("--out", default="results_signal_aggregated", help="Directory to write aggregated outputs.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("env_feature_importance.csv", ["feature"], "delta_r2_mean"),
        ("env_window_importance.csv", ["window_idx"], "delta_r2_mean"),
        ("env_stage_feature_importance.csv", ["stage", "feature"], "delta_r2_mean"),
        ("module_input_sensitivity.csv", ["component"], "delta_r2_mean"),
        ("population_metrics.csv", ["population"], "r2_mean"),
    ]

    for filename, group_cols, sort_col in specs:
        agg = _aggregate_table(root, filename, group_cols=group_cols, sort_col=sort_col)
        if agg.empty:
            continue
        out_csv = out_dir / f"aggregated_{filename}"
        agg.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
        if filename == "env_feature_importance.csv":
            _save_barplot(agg, "feature", "delta_r2_mean", out_dir / "aggregated_env_feature_importance.png", "Mean env feature importance")
        elif filename == "module_input_sensitivity.csv":
            _save_barplot(agg, "component", "delta_r2_mean", out_dir / "aggregated_module_input_sensitivity.png", "Mean module sensitivity")
        elif filename == "population_metrics.csv":
            _save_barplot(agg.sort_values("r2_mean", ascending=False), "population", "r2_mean", out_dir / "aggregated_population_r2.png", "Mean per-population R2")

    # Optional polyploid output: dedicated homeolog / subgenome sensitivity
    homeolog_agg = _aggregate_homeolog_table(root)
    if not homeolog_agg.empty:
        out_csv = out_dir / "aggregated_homeolog_feature_importance.csv"
        homeolog_agg.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
        _save_barplot(
            homeolog_agg,
            "component",
            "delta_r2_mean",
            out_dir / "aggregated_homeolog_feature_importance.png",
            "Mean homeolog / subgenome sensitivity",
            top_k=20,
        )

    print(f"Done. Aggregated results are in {out_dir}")


if __name__ == "__main__":
    main()
