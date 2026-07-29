#!/usr/bin/env python
"""
collect_automodel.py -- gather run2_automodel/<D>/<trait>/ results into one table.

Headline automated-model number per trait = the VALID reg-ensemble: average of the
reg_low/reg_mid/reg_high test predictions (they share the CV partition, verified by
matching (SampleID,fold) keys), which gave a real +0.01 over any single reg on D1 yield.
Falls back to the single 'full' run when the reg runs are absent.

Also reports: full, geno-CV, per-seed spread + seed note, and ablation deltas vs headline.
Robust to missing experiments (reports what exists).
"""
import os, sys, glob, re
import numpy as np, pandas as pd

REV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(REV, "run2_automodel")
ENVCV = "tier2_env_cv_fold_predictions.csv"
GENOCV = "tier1_geno_cv_fold_predictions.csv"
REG_LEVELS = ["reg_low", "reg_mid", "reg_high"]
ABLATIONS = ["abl_bae", "abl_habe", "abl_add", "abl_weather", "abl_pop"]
SEEDS = ["seed07", "seed42", "seed88", "seed123"]

def fold_r2(df):
    return float(np.mean([1 - ((g.true-g.pred)**2).sum()/((g.true-g.true.mean())**2).sum()
                          for _, g in df.groupby("fold")]))

def load(expdir, fname=ENVCV):
    f = os.path.join(expdir, fname)
    return pd.read_csv(f) if os.path.isfile(f) else None

def keyed(df):
    df = df.copy()
    df["_k"] = df["SampleID"].astype(str) + "|" + df["fold"].astype(str)
    return df.set_index("_k")

def reg_ensemble(traitdir):
    """Average reg-level predictions IF they share the partition; else None."""
    runs = {}
    for lv in REG_LEVELS:
        d = load(os.path.join(traitdir, lv))
        if d is not None:
            runs[lv] = keyed(d)
    if len(runs) < 2:
        return None, list(runs)
    key_sets = [set(v.index) for v in runs.values()]
    common = set.intersection(*key_sets)
    if len(common) != max(len(s) for s in key_sets):   # partitions differ -> not a valid ensemble
        return None, list(runs)
    common = sorted(common)
    base = list(runs.values())[0].loc[common].copy()
    base["pred"] = np.column_stack([v.loc[common, "pred"].to_numpy(float) for v in runs.values()]).mean(1)
    return fold_r2(base.reset_index()), list(runs)

def main():
    if not os.path.isdir(ROOT):
        print(f"no {ROOT} yet"); return
    rows = []
    for traitdir in sorted(glob.glob(os.path.join(ROOT, "*", "*"))):
        if not os.path.isdir(traitdir):
            continue
        D, TR = traitdir.split(os.sep)[-2:]
        row = {"dataset": D, "trait": TR}
        # headline = reg-ensemble, else full
        ens, reg_present = reg_ensemble(traitdir)
        full = load(os.path.join(traitdir, "full"))
        row["headline_R2"] = round(ens, 3) if ens is not None else (round(fold_r2(full), 3) if full is not None else None)
        row["headline_src"] = f"reg-ens({len(reg_present)})" if ens is not None else ("full" if full is not None else "-")
        row["full_R2"] = round(fold_r2(full), 3) if full is not None else None
        # geno-CV
        gcv = load(os.path.join(traitdir, "genocv"), GENOCV)
        row["genoCV_R2"] = round(fold_r2(gcv), 3) if gcv is not None else None
        # seeds
        sv = [fold_r2(load(os.path.join(traitdir, s))) for s in SEEDS if load(os.path.join(traitdir, s)) is not None]
        row["seed_mean"] = round(np.mean(sv), 3) if sv else None
        row["n_seed"] = len(sv)
        # ablation deltas vs headline
        base = row["headline_R2"]
        for ab in ABLATIONS:
            d = load(os.path.join(traitdir, ab))
            if d is not None and base is not None:
                row[f"d_{ab}"] = round(fold_r2(d) - base, 3)
        rows.append(row)
    if not rows:
        print(f"{ROOT} exists but no completed experiments found yet."); return
    df = pd.DataFrame(rows)
    out = os.path.join(REV, "automodel_results.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    print("headline_R2 = reg-ensemble (avg reg_low/mid/high, same partition) or 'full' fallback.")

if __name__ == "__main__":
    main()
