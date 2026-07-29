#!/usr/bin/env python
"""
emit_best_reg.py -- pick the automated model's regularization by VALIDATION, per trait.

"""
import os, re, sys, argparse

FOLD_RE = re.compile(r"Fold (\d+)/\d+")
BEST_RE = re.compile(r"New best model saved \(Val R.{0,4}=\s*([-\d.]+)\)")

# reg level name -> (dropout, weight_decay); must match run_experiments.sh EXPERIMENTS
LEVELS = {
    "reg_low":  (0.1,  0.001),
    "reg_mid":  (0.25, 0.005),
    "reg_high": (0.5,  0.05),
}

def val_r2(logpath):
    """Mean over folds of each fold's best validation R2, parsed from the log."""
    if not os.path.isfile(logpath):
        return None
    best = {}
    cur = 0
    with open(logpath, errors="ignore") as fh:
        for line in fh:
            m = FOLD_RE.search(line)
            if m:
                cur = int(m.group(1))
            b = BEST_RE.search(line)
            if b:
                best[cur] = max(best.get(cur, -9e9), float(b.group(1)))
    if not best:
        return None
    return sum(best.values()) / len(best)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run2_automodel/<D>/<trait> dir")
    ap.add_argument("--task", required=True, help="task name, e.g. d1_yield")
    ap.add_argument("--levels", default="reg_low,reg_mid,reg_high")
    ap.add_argument("--fallback", default="ECOPOP_GXE_DROPOUT=0.25 ECOPOP_WEIGHT_DECAY=0.005")
    a = ap.parse_args()
    scores = {}
    for lv in a.levels.split(","):
        s = val_r2(os.path.join(a.dir, f"{a.task}_{lv}.log"))
        print(f"# {lv}: val R2 = {s}", file=sys.stderr)
        if s is not None:
            scores[lv] = s
    if not scores:
        print(f"# no reg logs parsed; using fallback", file=sys.stderr)
        print(a.fallback)
        return
    best = max(scores, key=scores.get)
    d, w = LEVELS[best]
    print(f"# picked {best} (val R2 = {scores[best]:.4f})", file=sys.stderr)
    print(f"ECOPOP_GXE_DROPOUT={d} ECOPOP_WEIGHT_DECAY={w}")

if __name__ == "__main__":
    main()
