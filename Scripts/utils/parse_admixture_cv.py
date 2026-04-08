#!/usr/bin/env python3
"""Parse ADMIXTURE CV logs, summarize CV error by K, and choose a best K."""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RE_K_CV = re.compile(r"CV\s*error\s*\(K\s*=\s*(\d+)\)\s*:\s*([0-9]*\.?[0-9]+)")
RE_CV_ONLY = re.compile(r"CV\s*error\s*:\s*([0-9]*\.?[0-9]+)")
RE_K_ONLY = re.compile(r"\(K\s*=\s*(\d+)\)")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def parse_log(path: str):
    text = read_text(path)
    matches = RE_K_CV.findall(text)
    if matches:
        k, cv = matches[-1]
        return int(k), float(cv)
    cv_matches = RE_CV_ONLY.findall(text)
    k_matches = RE_K_ONLY.findall(text)
    if cv_matches and k_matches:
        return int(k_matches[-1]), float(cv_matches[-1])
    if cv_matches:
        fname = os.path.basename(path)
        m = re.search(r"[Kk]\s*[_-:]?\s*(\d+)", fname)
        if m:
            return int(m.group(1)), float(cv_matches[-1])
    return None, None


def add_elbow_scores(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values("K").reset_index(drop=True)
    if len(ordered) == 1:
        ordered["elbow_score"] = 0.0
        return ordered
    if len(ordered) == 2:
        ordered["elbow_score"] = 0.0
        return ordered

    x = ordered["K"].to_numpy(dtype=float)
    y = ordered["cv_mean"].to_numpy(dtype=float)
    p1 = np.array([x[0], y[0]], dtype=float)
    p2 = np.array([x[-1], y[-1]], dtype=float)
    line = p2 - p1
    denom = float(np.linalg.norm(line))
    if denom == 0.0:
        ordered["elbow_score"] = 0.0
        return ordered

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distances = np.abs(dy * x - dx * y + p2[0] * p1[1] - p2[1] * p1[0]) / denom
    ordered = ordered.copy()
    ordered["elbow_score"] = distances
    return ordered


def choose_elbow_k(summary: pd.DataFrame) -> tuple[int, pd.Series, pd.DataFrame]:
    scored = add_elbow_scores(summary)
    if len(scored) < 3:
        row = scored.loc[scored["cv_mean"].idxmin()].copy()
        return int(row["K"]), row, scored

    interior = scored.iloc[1:-1].copy()
    if interior.empty:
        row = scored.loc[scored["cv_mean"].idxmin()].copy()
        return int(row["K"]), row, scored

    best = interior.sort_values(
        ["elbow_score", "K"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return int(best["K"]), best, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pattern", action="append", default=["*.out", "*.log", "log*.out", "admixture*.out"])
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-plot", default=None)
    parser.add_argument(
        "--selection-method",
        default="elbow",
        choices=["elbow", "min_cv"],
        help="How to choose the best K from the CV summary.",
    )
    args = parser.parse_args()

    files = []
    for pattern in args.pattern:
        files.extend(glob.glob(os.path.join(args.log_dir, pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No log files found in {args.log_dir}")

    by_k = defaultdict(list)
    for path in files:
        k, cv = parse_log(path)
        if k is None or cv is None or math.isnan(cv):
            continue
        by_k[k].append(cv)

    if not by_k:
        raise RuntimeError("No parsable K/CV pairs found in the supplied logs.")

    ks = sorted(by_k)
    summary = pd.DataFrame(
        {
            "K": ks,
            "cv_mean": [float(np.mean(by_k[k])) for k in ks],
            "cv_sd": [float(np.std(by_k[k], ddof=1)) if len(by_k[k]) > 1 else 0.0 for k in ks],
            "n_runs": [len(by_k[k]) for k in ks],
        }
    )
    best_min_row = summary.loc[summary["cv_mean"].idxmin()].copy()
    best_elbow_k, best_elbow_row, summary = choose_elbow_k(summary)
    summary = summary.copy()
    summary["is_min_cv"] = summary["K"].astype(int) == int(best_min_row["K"])
    summary["is_elbow"] = summary["K"].astype(int) == int(best_elbow_k)

    if args.selection_method == "elbow":
        best_row = best_elbow_row
    else:
        best_row = best_min_row

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)

    if args.out_plot:
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.errorbar(summary["K"], summary["cv_mean"], yerr=summary["cv_sd"], marker="o", linestyle="-", capsize=3)
        plt.xlabel("K")
        plt.ylabel("CV error")
        plt.title("ADMIXTURE CV error by K")
        plt.axvline(int(best_elbow_k), color="tab:orange", linestyle="--", linewidth=1.5, label=f"Elbow K={int(best_elbow_k)}")
        plt.axvline(int(best_min_row["K"]), color="tab:green", linestyle=":", linewidth=1.5, label=f"Min-CV K={int(best_min_row['K'])}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_plot, dpi=300)
        plt.close()

    print(f"Selection_Method={args.selection_method}")
    print(f"Best_K={int(best_row['K'])}")
    print(f"Best_K_Elbow={int(best_elbow_k)}")
    print(f"Best_K_MinCV={int(best_min_row['K'])}")
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
