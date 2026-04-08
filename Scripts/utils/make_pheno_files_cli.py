#!/usr/bin/env python3
"""Prepare phenotype metadata and per-environment phenotype files for EcoPopDL-GP.

This script turns a raw phenotype table into:
  1. a model-ready metadata file with SampleID/FID/IID, Location, Year, Pop, and target column
  2. a mean phenotype table that preserves environment/design columns and averages true replicates
  2. per-environment phenotype files (FID, IID, PHENO)
  3. optional covariate files per environment
  4. an overall phenotype file averaged across environments
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "trait"


def parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")


def infer_group_cols(
    df: pd.DataFrame,
    sample_col: str,
    location_col: str,
    year_col: str,
    pop_col: str,
    trait_col: str,
) -> Tuple[List[str], List[str]]:
    base_cols = [sample_col, location_col, year_col]
    if pop_col and pop_col in df.columns:
        base_cols.append(pop_col)

    candidate_cols = [
        col for col in df.columns
        if col not in set(base_cols + [trait_col])
    ]

    distinguishing_cols: List[str] = []
    if candidate_cols:
        grouped = df.groupby(base_cols, dropna=False, sort=False)
        for col in candidate_cols:
            nunique = grouped[col].nunique(dropna=False)
            if (nunique > 1).any():
                distinguishing_cols.append(col)

    return base_cols + distinguishing_cols, distinguishing_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw phenotype CSV/TSV file.")
    parser.add_argument("--sep", default=None, help="Delimiter. Default: auto-detect via pandas.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--sample-col", default="Name", help="Sample/genotype identifier column.")
    parser.add_argument("--location-col", default="Location", help="Environment location column.")
    parser.add_argument("--year-col", default="Year", help="Environment year column.")
    parser.add_argument("--pop-col", default="Pop", help="Population column. Leave empty to skip.")
    parser.add_argument("--trait-col", required=True, help="Raw trait column to aggregate.")
    parser.add_argument("--target-col", default=None, help="Target column name to use in model metadata. Default: same as --trait-col.")
    parser.add_argument(
        "--group-cols",
        default=None,
        help="Comma-separated grouping columns used before averaging replicates. Default: sample, location, year, and pop when present.",
    )
    parser.add_argument(
        "--covariate-cols",
        default=None,
        help="Comma-separated covariate columns to write into per-environment covariate files.",
    )
    parser.add_argument("--agg", default="mean", choices=["mean", "median"], help="Aggregation function for replicated trait records.")
    parser.add_argument("--metadata-name", default=None, help="Optional explicit metadata filename.")
    parser.add_argument("--mean-name", default=None, help="Optional explicit mean-table filename.")
    parser.add_argument("--write-overall", action="store_true", help="Also write an overall phenotype file across all environments.")
    parser.add_argument("--write-covariates", action="store_true", help="Write per-environment covariate files.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sep = args.sep
    if sep is None:
        df = pd.read_csv(args.input, sep=None, engine="python")
    else:
        df = pd.read_csv(args.input, sep=sep)

    target_col = args.target_col or args.trait_col
    pop_col = args.pop_col.strip() if args.pop_col else ""

    base_required = [args.sample_col, args.location_col, args.year_col, args.trait_col]
    if pop_col:
        base_required.append(pop_col)
    ensure_columns(df, base_required)

    group_cols = parse_csv_list(args.group_cols)
    inferred_extra_group_cols: List[str] = []
    if not group_cols:
        group_cols, inferred_extra_group_cols = infer_group_cols(
            df,
            sample_col=args.sample_col,
            location_col=args.location_col,
            year_col=args.year_col,
            pop_col=pop_col,
            trait_col=args.trait_col,
        )

    covariate_cols = parse_csv_list(args.covariate_cols)
    if args.write_covariates and not covariate_cols:
        covariate_cols = [c for c in [args.year_col, pop_col] if c and c in df.columns]

    ensure_columns(df, [c for c in group_cols if c])
    if covariate_cols:
        ensure_columns(df, covariate_cols)

    df = df.copy()
    df[args.sample_col] = df[args.sample_col].astype(str).str.strip()
    df[args.location_col] = df[args.location_col].astype(str).str.strip()
    df[args.year_col] = df[args.year_col].astype(str).str.strip()
    df[args.trait_col] = pd.to_numeric(df[args.trait_col], errors="coerce")
    df = df.dropna(subset=[args.trait_col]).reset_index(drop=True)

    agg_fn = "median" if args.agg == "median" else "mean"
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(**{target_col: (args.trait_col, agg_fn)})
        .reset_index(drop=True)
    )

    grouped["SampleID"] = grouped[args.sample_col].astype(str)
    grouped["FID"] = grouped["SampleID"]
    grouped["IID"] = grouped["SampleID"]

    rename_map = {
        args.location_col: "Location",
        args.year_col: "Year",
    }
    if pop_col and pop_col in grouped.columns:
        rename_map[pop_col] = "Pop"
    grouped = grouped.rename(columns=rename_map)

    passthrough_cols = [
        rename_map.get(col, col)
        for col in group_cols
        if col not in {args.sample_col, args.location_col, args.year_col, pop_col}
    ]

    ordered_cols = [
        "SampleID",
        "FID",
        "IID",
        "Location",
        "Year",
    ]
    ordered_cols.extend(passthrough_cols)
    if "Pop" in grouped.columns:
        ordered_cols.append("Pop")
    ordered_cols.append(target_col)
    metadata = grouped[ordered_cols].copy()

    trait_slug = sanitize_name(target_col)
    env_prefix = sanitize_name(target_col.lower())
    metadata_name = args.metadata_name or f"metadata_{env_prefix}.csv"
    metadata_path = outdir / metadata_name
    metadata.to_csv(metadata_path, index=False)

    mean_name = args.mean_name or f"{env_prefix}_mean.txt"
    mean_path = outdir / mean_name
    metadata.to_csv(mean_path, sep="\t", index=False)
    for (loc, year), sub in metadata.groupby(["Location", "Year"], sort=True):
        tag = f"{sanitize_name(loc)}_{sanitize_name(year)}"
        pheno = sub[["FID", "IID", target_col]].rename(columns={target_col: "PHENO"})
        pheno.to_csv(outdir / f"{env_prefix}_{tag}.txt", sep="\t", index=False)
        if args.write_covariates:
            cols = ["FID", "IID"] + [
                "Year" if c == args.year_col else ("Pop" if c == pop_col else c)
                for c in covariate_cols
                if ("Year" if c == args.year_col else ("Pop" if c == pop_col else c)) in metadata.columns
            ]
            cols = list(dict.fromkeys(cols))
            metadata.loc[sub.index, cols].to_csv(outdir / f"covar_{tag}.txt", sep="\t", index=False)

    if args.write_overall:
        overall_group_cols = ["FID", "IID"]
        if "Pop" in metadata.columns:
            overall_group_cols.append("Pop")
        overall = (
            metadata.groupby(overall_group_cols, as_index=False)
            .agg(**{target_col: (target_col, agg_fn)})
            .rename(columns={target_col: "PHENO"})
        )
        overall.to_csv(outdir / f"{env_prefix}_overall.txt", sep="\t", index=False)

    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote mean table: {mean_path}")
    if inferred_extra_group_cols:
        print(
            "Auto-detected extra grouping columns that distinguish rows within "
            f"sample/location/year/pop: {', '.join(inferred_extra_group_cols)}"
        )
    replicate_groups = int((df.groupby(group_cols, dropna=False).size() > 1).sum())
    print(f"Replicate groups averaged: {replicate_groups}")
    print(f"Rows: {metadata.shape[0]} | Unique samples: {metadata['SampleID'].nunique()}")
    print(f"Environments: {metadata[['Location', 'Year']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
