#!/usr/bin/env python
"""Collect every run in runs/ into a comprehensive results table (long CSV + pivot + markdown)."""
import os, glob, re
import numpy as np, pandas as pd

REV = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REV, "runs")

CV_LABEL = {
    "tier2_env": "env-CV", "tier1_geno": "geno-CV",
    "tier3_location": "env-blocked(loc)", "tier3_year": "env-blocked(year)",
    "tier3_loc_year": "env-blocked(loc-year)", "tier3_population": "pop-blocked",
}
def cv_from_fname(fn):
    for k in CV_LABEL:
        if fn.startswith(k):
            return k
    return fn.replace("_cv_fold_predictions.csv", "")

def metrics(df):
    out = {}
    fr2, prd, prs = [], [], []
    for _, g in df.groupby("fold"):
        y, yh = g["true"].to_numpy(float), g["pred"].to_numpy(float)
        ss = ((y - y.mean())**2).sum()
        if ss > 0 and len(y) > 1:
            fr2.append(1 - ((y - yh)**2).sum()/ss)
            if np.std(yh) > 0:
                r = np.corrcoef(y, yh)[0, 1]; prd.append(r**2); prs.append(r)
    y, yh = df["true"].to_numpy(float), df["pred"].to_numpy(float)
    out["R2"] = np.mean(fr2) if fr2 else np.nan
    out["R2_std"] = np.std(fr2) if fr2 else np.nan
    out["pred_ability_r2"] = np.mean(prd) if prd else np.nan
    out["pearson_r"] = np.mean(prs) if prs else np.nan
    out["RMSE"] = np.sqrt(np.mean((y - yh)**2))
    out["MAE"] = np.mean(np.abs(y - yh))
    out["n"] = len(df); out["n_folds"] = df["fold"].nunique()
    return out

rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
    if not os.path.isdir(d): continue
    name = os.path.basename(d)
    m = re.match(r"(d[1-4])_([a-z]+)_(.+)", name)
    if not m: continue
    ds, trait, exp = m.group(1).upper(), m.group(2), m.group(3)
    for f in sorted(glob.glob(os.path.join(d, "*fold_predictions.csv"))):
        try:
            df = pd.read_csv(f)
            if not {"true", "pred", "fold"} <= set(df.columns): continue
            cv = cv_from_fname(os.path.basename(f))
            row = {"dataset": ds, "trait": trait, "experiment": exp,
                   "cv_scheme": CV_LABEL.get(cv, cv)}
            row.update(metrics(df))
            rows.append(row)
        except Exception as e:
            print(f"  skip {f}: {e}")

long = pd.DataFrame(rows)
for c in ["R2", "R2_std", "pred_ability_r2", "pearson_r", "RMSE", "MAE"]:
    long[c] = long[c].round(3)
long = long.sort_values(["dataset", "trait", "cv_scheme", "experiment"]).reset_index(drop=True)
long.to_csv(os.path.join(REV, "ALL_RESULTS_long.csv"), index=False)

# pivot: primary metric (R2) by task x experiment, using the main CV scheme per row
main = long[long["cv_scheme"].isin(["env-CV", "geno-CV"])].copy()
main["task"] = main["dataset"] + "_" + main["trait"]
pivot = main.pivot_table(index="task", columns="experiment", values="R2", aggfunc="max")
pivot.to_csv(os.path.join(REV, "ALL_RESULTS_pivot_R2.csv"))

# markdown
with open(os.path.join(REV, "RESULTS_TABLE.md"), "w") as fh:
    fh.write("# EcoPopDL-GP — full results (runs/)\n\n")
    fh.write(f"{len(long)} evaluations across {main['task'].nunique()} tasks.\n\n")
    fh.write("## R² by task × experiment (env-CV / geno-CV)\n\n")
    fh.write(pivot.round(3).to_markdown() + "\n\n")
    fh.write("## Env-blocked & pop-blocked (ranking-relevant Pearson r; absolute R² is expectedly negative)\n\n")
    eb = long[long["cv_scheme"].str.contains("blocked")][
        ["dataset", "trait", "experiment", "cv_scheme", "R2", "pearson_r", "n"]]
    fh.write(eb.to_markdown(index=False) + "\n\n")
    fh.write("## Full detail (every run, all metrics)\n\n")
    fh.write(long.to_markdown(index=False) + "\n")

print(f"WROTE:\n  RESULTS_TABLE.md\n  ALL_RESULTS_long.csv ({len(long)} rows)\n  ALL_RESULTS_pivot_R2.csv")
print("\n=== pivot preview (R2 by task x experiment) ===")
pd.set_option("display.width", 240, "display.max_columns", 40)
print(pivot.round(3).to_string())
