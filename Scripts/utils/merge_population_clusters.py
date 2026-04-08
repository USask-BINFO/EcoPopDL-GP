#!/usr/bin/env python3
"""Merge ADMIXTURE-derived population clusters into model metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Metadata CSV created by make_pheno_files_cli.py")
    parser.add_argument("--clusters", required=True, help="Cluster CSV created by admixture_assign_cli.py")
    parser.add_argument("--metadata-sample-col", default="SampleID")
    parser.add_argument("--cluster-sample-col", default="SampleID")
    parser.add_argument("--cluster-col", default="Pop")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, sep=None, engine="python")
    clusters = pd.read_csv(args.clusters)

    if args.metadata_sample_col not in meta.columns:
        raise KeyError(f"Metadata sample column not found: {args.metadata_sample_col}")
    if args.cluster_sample_col not in clusters.columns:
        raise KeyError(f"Cluster sample column not found: {args.cluster_sample_col}")
    if args.cluster_col not in clusters.columns:
        raise KeyError(f"Cluster column not found: {args.cluster_col}")

    clusters = clusters[[args.cluster_sample_col, args.cluster_col]].copy()
    clusters.columns = [args.metadata_sample_col, "Cluster"]
    clusters[args.metadata_sample_col] = clusters[args.metadata_sample_col].astype(str).str.strip()
    meta[args.metadata_sample_col] = meta[args.metadata_sample_col].astype(str).str.strip()

    if "Population" in meta.columns and "Pop" in meta.columns:
        meta = meta.drop(columns=["Pop"])
    elif "Population" not in meta.columns and "Pop" in meta.columns:
        meta = meta.rename(columns={"Pop": "Population"})
    meta = meta.drop(columns=["Cluster"], errors="ignore")

    merged = meta.merge(
        clusters,
        on=args.metadata_sample_col,
        how="left",
    )
    missing = merged["Cluster"].isna().sum()
    if missing:
        print(f"Warning: {missing} metadata rows have no cluster assignment; filling with -1.")
        merged["Cluster"] = merged["Cluster"].fillna(-1)
    merged["Pop"] = merged["Cluster"]

    preferred = [
        "SampleID", "FID", "IID", "Location", "Year",
        "Population", "Cluster", "Pop",
    ]
    ordered = [col for col in preferred if col in merged.columns]
    ordered.extend(col for col in merged.columns if col not in ordered)
    merged = merged[ordered]

    out_path = Path(args.out)
    out_sep = "\t" if out_path.suffix.lower() in {".txt", ".tsv"} else ","
    merged.to_csv(out_path, index=False, sep=out_sep)
    print(f"Wrote merged metadata: {args.out}")


if __name__ == "__main__":
    main()
