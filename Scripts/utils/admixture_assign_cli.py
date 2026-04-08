#!/usr/bin/env python3
"""Assign ADMIXTURE clusters to samples from a Q matrix and a PLINK FAM file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FAM_COLS = ["FID", "IID", "PID", "MID", "SEX", "PHENO"]


def clean_ids(series: pd.Series, mode: str) -> pd.Series:
    s = series.astype(str)
    if mode == "none":
        return s
    if mode == "strip_underscores":
        return s.str.replace("_", "", regex=False)
    if mode == "dedup_last_underscore":
        out = []
        for value in s:
            if "_" in value:
                left, right = value.rsplit("_", 1)
                out.append(left if left == right else value)
            else:
                out.append(value)
        return pd.Series(out, index=series.index)
    raise ValueError(f"Unsupported cleanup mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-file", required=True, help="ADMIXTURE Q matrix file (e.g., prefix.6.Q).")
    parser.add_argument("--plink-prefix", required=True, help="PLINK prefix matching the Q matrix sample order.")
    parser.add_argument("--id-source", default="iid", choices=["iid", "fid"], help="Use FID or IID from the FAM file.")
    parser.add_argument(
        "--cleanup",
        default="dedup_last_underscore",
        choices=["none", "strip_underscores", "dedup_last_underscore"],
        help="Normalize sample IDs before writing the output.",
    )
    parser.add_argument("--one-based", action="store_true", help="Write population labels as 1..K instead of 0..K-1.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    fam_path = Path(f"{args.plink_prefix}.fam")
    if not fam_path.exists():
        raise FileNotFoundError(f"Missing FAM file: {fam_path}")

    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, names=FAM_COLS)
    q = pd.read_csv(args.q_file, sep=r"\s+", header=None)

    if len(fam) != len(q):
        raise ValueError(
            f"Sample count mismatch: FAM has {len(fam)} rows, Q matrix has {len(q)} rows."
        )

    sample_series = fam["IID" if args.id_source == "iid" else "FID"]
    sample_series = clean_ids(sample_series, args.cleanup)

    cluster = q.idxmax(axis=1).astype(int)
    if args.one_based:
        cluster = cluster + 1

    out = pd.DataFrame({
        "SampleID": sample_series,
        "Pop": cluster,
    })
    for i in range(q.shape[1]):
        out[f"Q{i + 1}"] = q.iloc[:, i]

    out.to_csv(args.out, index=False)
    print(f"Wrote cluster assignments: {args.out}")


if __name__ == "__main__":
    main()
