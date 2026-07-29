"""
Aggregate multi-seed / multi-fold results into mean +/- std and bootstrap CIs,
and test whether EcoPopDL-GP's margin over the best baseline is stable.

Directly supports the reviewer requests for (a) variance across repeated splits
on small datasets, and (b) interpretation of small geno-CV margins (are they real?).

Dependency-light: numpy + pandas only (no scipy).

Expected input: one long-format CSV with (at least) these columns:
    dataset, trait, scheme, model, seed, fold, r2, rmse, mae, ccc
where `scheme` is e.g. env_cv / geno_cv / loc_out / year_out / locyear_out,
and `model` includes "EcoPopDL-GP" plus each baseline.

If your training script writes per-fold metrics in a different shape, write a
small adapter to this schema; that is the only glue needed.
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd

OUR_MODEL_DEFAULT = "EcoPopDL-GP"
METRICS = ["r2", "rmse", "mae", "ccc"]
# For these, higher is better -> our_margin = ours - baseline; for error metrics, lower is better.
HIGHER_IS_BETTER = {"r2": True, "ccc": True, "rmse": False, "mae": False}


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


def summarize_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of each metric per (dataset, trait, scheme, model)."""
    rows = []
    keys = ["dataset", "trait", "scheme", "model"]
    for key_vals, g in df.groupby(keys):
        rec = dict(zip(keys, key_vals))
        rec["n_runs"] = len(g)
        for m in METRICS:
            if m in g:
                rec[f"{m}_mean"] = float(g[m].mean())
                rec[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "trait", "scheme", "r2_mean"], ascending=[True, True, True, False])


def margin_over_best_baseline(
    df: pd.DataFrame,
    our_model: str = OUR_MODEL_DEFAULT,
    metric: str = "r2",
    n_boot: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For each (dataset, trait, scheme): pick the best baseline by mean `metric`,
    then compute the paired margin (ours - baseline) across shared (seed, fold),
    with a bootstrap CI and a win-rate. If runs are not paired on (seed, fold),
    fall back to (mean_ours - mean_baseline) with an unpaired bootstrap.
    """
    higher = HIGHER_IS_BETTER[metric]
    out = []
    for (ds, tr, sch), g in df.groupby(["dataset", "trait", "scheme"]):
        ours = g[g["model"] == our_model]
        base = g[g["model"] != our_model]
        if ours.empty or base.empty:
            continue
        # best baseline by mean metric
        base_means = base.groupby("model")[metric].mean()
        best_baseline = (base_means.idxmax() if higher else base_means.idxmin())
        b = base[base["model"] == best_baseline]

        # try to pair on (seed, fold)
        merged = ours.merge(
            b, on=["seed", "fold"], suffixes=("_ours", "_base"), how="inner"
        )
        if len(merged) >= 2:
            diff = merged[f"{metric}_ours"].values - merged[f"{metric}_base"].values
            if not higher:
                diff = -diff  # for error metrics, positive diff = we are better
            paired = True
            win_rate = float((diff > 0).mean())
        else:
            # unpaired fallback
            a = ours[metric].values
            c = b[metric].values
            m = min(len(a), len(c))
            rng = np.random.default_rng(seed)
            a = rng.permutation(a)[:m]
            c = rng.permutation(c)[:m]
            diff = (a - c) if higher else (c - a)
            paired = False
            win_rate = float((diff > 0).mean()) if m else np.nan

        mean_d, lo, hi = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        out.append({
            "dataset": ds, "trait": tr, "scheme": sch,
            "metric": metric, "best_baseline": best_baseline,
            "delta_mean": round(mean_d, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "excludes_zero": bool(lo > 0),  # margin is stable/positive if True
            "win_rate": round(win_rate, 3) if win_rate == win_rate else np.nan,
            "paired": paired, "n_compared": int(len(diff)),
        })
    return pd.DataFrame(out).sort_values(["scheme", "dataset", "trait"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Aggregate multi-seed results and test margin stability.")
    ap.add_argument("--results", required=True, help="Long-format results CSV (see module docstring).")
    ap.add_argument("--our-model", default=OUR_MODEL_DEFAULT)
    ap.add_argument("--metric", default="r2", choices=METRICS)
    ap.add_argument("--out-prefix", default="multiseed")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    per_model = summarize_per_model(df)
    per_model.to_csv(f"{args.out_prefix}_per_model_summary.csv", index=False)

    margins = margin_over_best_baseline(
        df, our_model=args.our_model, metric=args.metric, n_boot=args.n_boot, seed=args.seed
    )
    margins.to_csv(f"{args.out_prefix}_margins.csv", index=False)

    print(f"\nPer-model summary -> {args.out_prefix}_per_model_summary.csv ({len(per_model)} rows)")
    print(f"Margin-vs-best-baseline ({args.metric}) -> {args.out_prefix}_margins.csv\n")
    if not margins.empty:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(margins.to_string(index=False))
        stable = int(margins["excludes_zero"].sum())
        print(f"\n{stable}/{len(margins)} task-schemes have a bootstrap 95% CI on delta-{args.metric} "
              f"that EXCLUDES ZERO (i.e., a statistically stable positive margin).")


if __name__ == "__main__":
    main()
