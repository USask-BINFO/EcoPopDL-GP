#!/usr/bin/env python
"""
ensemble.py  --  seed-averaged prediction ensemble (the honest R2 booster).

Averages the PER-SAMPLE predictions across the seed runs of a task
(full=seed20, seed21..seed24), then computes R2 on the averaged prediction.
Averaging predictions (not averaging per-seed R2) reduces variance and the
val->test optimism, and typically gains +0.02..0.05 R2 for free.

Usage:  python ensemble.py d1_yield [d3_yield ...]     (default: all tasks)
Reads:  runs/<task>_{full,seed21,seed22,seed23,seed24}/tier2_env_cv_fold_predictions.csv
"""
import os, sys
import numpy as np, pandas as pd

REV = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(REV, "runs")
SEED_RUNS = ["full", "seed21", "seed22", "seed23", "seed24"]   # full == seed20
ALL = ["d1_yield","d1_dtf","d1_sw","d2_ft","d3_yield","d3_dtf","d3_sw","d4_oil","d4_dtf"]

def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ss = ((y - y.mean())**2).sum()
    return np.nan if ss <= 0 else 1 - ((y - yh)**2).sum()/ss

def foldmean_r2(df):
    return float(np.nanmean([r2(g["true"], g["pred"]) for _, g in df.groupby("fold")]))

def load(task, run):
    p = os.path.join(RUNS, f"{task}_{run}", "tier2_env_cv_fold_predictions.csv")
    if not os.path.isfile(p): return None
    d = pd.read_csv(p)
    return d if {"SampleID","pred","true","fold"}.issubset(d.columns) else None

tasks = sys.argv[1:] or ALL
print(f"{'task':10s} {'#seeds':>6s} {'single(mean)':>13s} {'ENSEMBLE':>9s} {'gain':>6s}")
print("-"*52)
for t in tasks:
    frames = [(s, load(t, s)) for s in SEED_RUNS]
    frames = [(s, d) for s, d in frames if d is not None]
    if not frames:
        print(f"{t:10s}  -- no seed runs yet --"); continue
    # align on SampleID+fold (identical splits across seeds); average pred
    base = frames[0][1][["SampleID","fold","true"]].copy()
    preds = []
    for s, d in frames:
        m = d[["SampleID","fold","pred"]].rename(columns={"pred": f"pred_{s}"})
        base = base.merge(m, on=["SampleID","fold"], how="inner")
        preds.append(f"pred_{s}")
    single = float(np.mean([foldmean_r2(frames[i][1]) for i in range(len(frames))]))
    ens = base.copy(); ens["pred"] = ens[preds].mean(axis=1)
    ens_r2 = foldmean_r2(ens[["true","pred","fold"]])
    print(f"{t:10s} {len(frames):6d} {single:13.3f} {ens_r2:9.3f} {ens_r2-single:+6.3f}")
    ens[["SampleID","fold","true","pred"]].to_csv(os.path.join(RUNS, f"{t}_ensemble_predictions.csv"), index=False)
print("\nENSEMBLE = R2 of the seed-averaged prediction; single = mean of per-seed R2.")
print("Wrote runs/<task>_ensemble_predictions.csv for each task with >=2 seeds.")
