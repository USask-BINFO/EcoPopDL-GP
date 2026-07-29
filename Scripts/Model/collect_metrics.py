#!/usr/bin/env python3
"""

Usage:
  python collect_metrics.py --runs-dir runs --out-prefix metrics
  # add existing (paper) D3 predictions for CCC recompute:
  python collect_metrics.py --runs-dir runs \
      --extra "d3_yield:full:/path/to/D3_Yield/CV/tier2_env_cv_fold_predictions.csv"
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

OUR = "EcoPopDL-GP"

# task -> (dataset, trait)
TASK_MAP = {
    "d1_yield": ("D1", "Yield"), "d1_dtf": ("D1", "DTF"), "d1_sw": ("D1", "SW"),
    "d2_ft": ("D2", "FT"),
    "d3_yield": ("D3", "Yield"), "d3_dtf": ("D3", "DTF"), "d3_sw": ("D3", "SW"),
    "d4_oil": ("D4", "Oil"), "d4_dtf": ("D4", "DTF"),
}


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    if y_true.size < 2:
        return np.nan
    vt, vp = np.var(y_true), np.var(y_pred)              # ddof=0
    cov = np.cov(y_true, y_pred, ddof=0)[0, 1]           # ddof=0 -> matches var (the fix)
    denom = vt + vp + (y_true.mean() - y_pred.mean()) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def metrics(df):
    yt, yp = df["true"].to_numpy(float), df["pred"].to_numpy(float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size < 2:
        return dict(n=int(yt.size), r2=np.nan, rmse=np.nan, mae=np.nan, ccc=np.nan)
    return dict(n=int(yt.size), r2=r2(yt, yp),
                rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
                mae=float(np.mean(np.abs(yt - yp))), ccc=ccc(yt, yp))


def parse_run_dir(name):
    """runs/<task>_<experiment>  ->  (task, experiment)."""
    for t in sorted(TASK_MAP, key=len, reverse=True):
        if name == t or name.startswith(t + "_"):
            return t, (name[len(t) + 1:] or "full")
    return None, name


def scheme_of(job_tag, experiment):
    jt = (job_tag or "").lower()
    if "tier1" in jt or "geno" in jt: return "geno_cv"
    if "tier3" in jt or "blocked" in jt or "population" in jt:
        return "pop_blocked" if "population" in jt else "env_blocked"
    if "tier2" in jt or "within" in jt or "env" in jt: return "env_cv"
    return {"genocv": "geno_cv", "envblocked": "env_blocked",
            "pop_blocked": "pop_blocked"}.get(experiment, "env_cv")


def variant_seed(experiment):
    m = re.fullmatch(r"seed(\d+)", experiment)
    if m: return "full", int(m.group(1))
    if experiment in ("full", "genocv", "envblocked", "pop_blocked"): return "full", 20
    return experiment, 20   # abl_*, season*


def collect(pred_files):
    rows = []
    for task, experiment, job_tag, path in pred_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  !! skip {path}: {e}"); continue
        if not {"pred", "true"}.issubset(df.columns) or df.empty:
            continue
        ds, tr = TASK_MAP.get(task, (task, "?"))
        sch = scheme_of(job_tag, experiment)
        var, seed = variant_seed(experiment)
        folds = sorted(df["fold"].unique()) if "fold" in df.columns else [0]
        # per fold
        for f in folds:
            sub = df[df["fold"] == f] if "fold" in df.columns else df
            rows.append(dict(dataset=ds, trait=tr, task=task, experiment=experiment,
                             scheme=sch, variant=var, seed=seed, model=OUR,
                             fold=int(f), pooled=False, **metrics(sub)))
        # pooled across folds
        rows.append(dict(dataset=ds, trait=tr, task=task, experiment=experiment,
                         scheme=sch, variant=var, seed=seed, model=OUR,
                         fold=-1, pooled=True, **metrics(df)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--extra", action="append", default=[],
                    help="task:experiment:/path/to/predictions.csv (repeatable)")
    ap.add_argument("--out-prefix", default="metrics")
    args = ap.parse_args()

    pred_files = []
    for path in glob.glob(os.path.join(args.runs_dir, "*", "**", "*cv_fold_predictions.csv"), recursive=True):
        run_dir = os.path.basename(os.path.dirname(path))
        # handle nested output dirs: take the runs/<dir>/ level
        rel = os.path.relpath(path, args.runs_dir)
        run_dir = rel.split(os.sep)[0]
        task, experiment = parse_run_dir(run_dir)
        if task is None:
            print(f"  ? unrecognized run dir '{run_dir}' ({path})"); continue
        job_tag = os.path.basename(path).replace("_cv_fold_predictions.csv", "")
        pred_files.append((task, experiment, job_tag, path))
    for spec in args.extra:
        task, experiment, path = spec.split(":", 2)
        pred_files.append((task, experiment, "", path))

    if not pred_files:
        print(f"No *cv_fold_predictions.csv found under {args.runs_dir}. (Runs may still be in progress.)")
        return
    print(f"Found {len(pred_files)} prediction file(s).")

    long = collect(pred_files)
    long.to_csv(f"{args.out_prefix}_long.csv", index=False)

    # per-experiment summary: mean +/- std across folds, and pooled
    per_fold = long[~long["pooled"]]
    summ = (per_fold.groupby(["dataset", "trait", "experiment", "scheme", "variant", "seed"])
            .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                 rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
                 ccc_mean=("ccc", "mean"), n_folds=("r2", "size"))
            .reset_index().sort_values(["dataset", "trait", "experiment"]))
    summ.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print(f"\nWrote {args.out_prefix}_long.csv  and  {args.out_prefix}_summary.csv")

    # ---- headline: EcoPop main results (env_cv 'full') ----
    main_full = summ[(summ["variant"] == "full") & (summ["scheme"] == "env_cv") & (summ["seed"] == 20)]
    if not main_full.empty:
        print("\n=== EcoPopDL-GP main (env-CV) — recomputed with fixed CCC ===")
        print(main_full[["dataset", "trait", "r2_mean", "rmse_mean", "mae_mean", "ccc_mean"]]
              .to_string(index=False))

    # ---- multi-seed stability (full + seed*) ----
    seeds = per_fold[(per_fold["variant"] == "full") & (per_fold["scheme"] == "env_cv")]
    if seeds["seed"].nunique() > 1:
        agg = (seeds.groupby(["dataset", "trait", "seed"])["r2"].mean().reset_index()
               .groupby(["dataset", "trait"])["r2"].agg(["mean", "std", "count"]).reset_index())
        print("\n=== Multi-seed stability (mean+/-std of per-seed R2) ===")
        print(agg.to_string(index=False))

    # ---- ablations (delta R2 vs full, per task) ----
    abls = summ[summ["variant"].str.startswith("abl_")]
    if not abls.empty:
        base = main_full.set_index(["dataset", "trait"])["r2_mean"]
        abls = abls.copy()
        abls["full_r2"] = abls.apply(lambda r: base.get((r["dataset"], r["trait"]), np.nan), axis=1)
        abls["delta_r2"] = abls["full_r2"] - abls["r2_mean"]
        print("\n=== Ablations (delta R2 = full - ablated; larger = more important) ===")
        print(abls[["dataset", "trait", "variant", "r2_mean", "delta_r2"]].to_string(index=False))

    # ---- environment-blocked ----
    eb = summ[summ["scheme"] == "env_blocked"]
    if not eb.empty:
        print("\n=== Environment-blocked CV (truly unseen environments) ===")
        print(eb[["dataset", "trait", "experiment", "r2_mean", "ccc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
