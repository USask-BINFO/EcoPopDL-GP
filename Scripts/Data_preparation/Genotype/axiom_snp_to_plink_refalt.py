#!/usr/bin/env python3


# # Founder-anchored REF (using your 8 lines)
# python3 axiom_to_plink_refalt.py \
#   --in axiom_export.tsv \
#   --sep tab \
#   --out_prefix chickpea_axiom \
#   --ref_mode founders \
#   --founders "CDC Consul" "CDC Leader" "CDC Orion" "CDC Cory" "ILC 533" "ICC 4958" "ILC 3279" "ILC 482"

# # (Alternative) Genome-anchored REF (CDC Frontier FASTA; contig names should match Ca1..Ca8)
# python3 axiom_to_plink_refalt.py \
#   --in axiom_export.tsv \
#   --sep tab \
#   --out_prefix chickpea_axiom \
#   --ref_mode genome \
#   --fasta CDC_Frontier.fa



import argparse, re
import numpy as np, pandas as pd
from pathlib import Path

COMP = {"A":"T","T":"A","C":"G","G":"C"}

# Assumes your table looks like:
# probeset_id | SNP Marker | [SAMPLES ...] | Affy_SNP_ID | Chr_id | Start | ...

# It finds the contiguous block of sample columns between "SNP Marker" and the first known metadata column (e.g., "Affy_SNP_ID" or "Chr_id").

# Returns the list of sample names (e.g., WCC..., CDC..., ILC..., ICC...).

def detect_samples(df):
    if "SNP Marker" in df.columns:
        left = df.columns.get_loc("SNP Marker") + 1
        # find first metadata col after samples
        meta_candidates = [c for c in df.columns[left:]
                           if c.lower() in ("affy_snp_id","affy_snpid","chr_id","chrom",
                                            "start","strand","dbsnp_rs_id","allele_count",
                                            "cr","fld","homfld","hetso","homro","nclus",
                                            "n_aa","n_ab","n_bb","n_nc","conversiontype",
                                            "bestprobeset","bestandrecommended","homhet")]
        if not meta_candidates:
            raise RuntimeError("Could not locate metadata block after samples; check headers.")
        right = df.columns.get_loc(meta_candidates[0])
        return list(df.columns[left:right])

    if "SNP_Array_Name" in df.columns and "N99_Position" in df.columns:
        left = df.columns.get_loc("N99_Position") + 1
        return list(df.columns[left:])

    raise RuntimeError(
        "Unsupported genotype table layout. Expected either Affymetrix-style "
        "headers including 'SNP Marker' or allele-call matrix headers "
        "including 'SNP_Array_Name' and 'N99_Position'."
    )


def normalize_input_columns(df):
    if "probeset_id" in df.columns and "Chr_id" in df.columns and "Start" in df.columns:
        if "Strand" not in df.columns:
            df["Strand"] = "+"
        return df

    if "SNP_Array_Name" in df.columns and "N99_Position" in df.columns:
        out = df.copy()
        pos_parts = out["N99_Position"].fillna("").astype(str).str.extract(r"^(?P<Chr_id>[^_]+)_(?P<Start>\d+)$")
        out["probeset_id"] = out["SNP_Array_Name"].astype(str)
        out["Chr_id"] = pos_parts["Chr_id"]
        out["Start"] = pos_parts["Start"]
        out["Strand"] = "+"
        bad = out["Chr_id"].isna() | out["Start"].isna()
        if bad.any():
            example = out.loc[bad, "N99_Position"].iloc[0]
            raise RuntimeError(
                f"Could not parse chromosome/start from N99_Position value '{example}'. "
                "Expected values like 'N1_9341172'."
            )
        return out

    missing = [c for c in ("probeset_id", "Chr_id", "Start") if c not in df.columns]
    raise RuntimeError(
        "Unsupported genotype table layout. Missing required columns: "
        + ", ".join(missing)
    )

# Normalizes each cell's call to a pair of alleles (A1, A2) for PLINK.

# Handles missing, single-letter, two-letter, and slash/pipe formats.

# Returns '0','0' for missing (PLINK's missing genotype code).

def split_geno(g):
    if pd.isna(g): return ("0","0")
    g = str(g).strip().upper()
    if g in ("", "---", "NC", "NOCALL", "NA", "NULL", "0"): return ("0","0")
    if len(g)==1 and g in "ACGT": return (g,g)
    if len(g)==2 and all(ch in "ACGT" for ch in g): return (g[0], g[1])
    if "/" in g or "|" in g:
        parts = re.split(r"[\/|]", g)
        parts = [p for p in parts if p in ["A","C","G","T"]]
        if len(parts)==2: return (parts[0], parts[1])
    return ("0","0")

# Returns the chromosome label for the .map file.

# With allow_extra=True (default) it keeps labels like Ca1..Ca8 (you'll then use PLINK's --allow-extra-chr if needed).

# If set to numeric, it would keep only trailing digits.

def normalize_chr(s, allow_extra=True):
    if pd.isna(s): return "0"
    s = str(s).strip()
    return s if allow_extra else (re.search(r"(\d+)$", s).group(1) if re.search(r"(\d+)$", s) else "0")

# Ensures alleles are on the forward (+) strand. If an Axiom row has Strand == '-', complement A<->T, C<->G.

def plus_allele(a, strand):
    """Return allele on + strand; complement if Strand == '-'."""
    if a not in "ACGT": return a
    return a if strand == "+" else COMP[a]

# Genome-anchored mode: use the reference FASTA (CDC Frontier) to get the true REF base at (Chr_id, Start).

# Collect observed alleles across your cohort (forced to + strand), pick ALT as the first observed allele != REF (if any).

def call_ref_alt_genome(row, fasta):
    """Use CDC Frontier FASTA to get REF; ALT is the other observed allele (on + strand)."""
    try:
        ref_base = fasta.fetch(str(row["Chr_id"]), int(row["Start"])-1, int(row["Start"])).upper()
    except Exception:
        return (None, None)  # unknown
    # observed alleles (on + strand) from the cohort
    obs = set()
    for a1,a2 in row["_alleles"]:
        if a1 in "ACGT": obs.add(plus_allele(a1, row["Strand"]))
        if a2 in "ACGT": obs.add(plus_allele(a2, row["Strand"]))
    obs.discard("0")
    if not obs: return (ref_base, None)
    alts = [a for a in sorted(obs) if a != ref_base]
    return (ref_base, alts[0] if alts else None)

# Founder-anchored mode:

# If CDC Leader and CDC Consul are both homozygous and agree -> use that allele as REF.

# Otherwise use the modal homozygous allele across the founders you passed in.

# ALT = a different observed allele (if present) on the + strand.

def call_ref_alt_founders(row, founders):
    """Use founders/controls: if Leader & Consul agree and are homozygous, use that; else modal across founders."""
    gt = {}
    for f in founders:
        if f not in row.index: continue
        a1,a2 = split_geno(row[f])
        if a1 in "ACGT" and a2 in "ACGT":
            gt[f] = (a1 if a1==a2 else None)  # None if het
    # priority 1: Leader & Consul agree and homozygous
    leader = gt.get("CDC Leader") or gt.get("CDCLeader")
    consul = gt.get("CDC Consul") or gt.get("CDCConsul")
    if leader and consul and leader == consul:
        ref = leader
    else:
        # modal homozygous allele across founders
        alleles = [v for v in gt.values() if v is not None]
        if alleles:
            ref = pd.Series(alleles).mode().iloc[0]
        else:
            ref = None
    # observed +strand alleles in cohort to define ALT
    obs = set()
    for a1,a2 in row["_alleles"]:
        if a1 in "ACGT": obs.add(plus_allele(a1, row["Strand"]))
        if a2 in "ACGT": obs.add(plus_allele(a2, row["Strand"]))
    obs.discard("0")
    if ref is None:
        return (None, None) if not obs else (sorted(obs)[0], (sorted(obs)[1] if len(obs)>1 else None))
    alts = [a for a in sorted(obs) if a != ref]
    return (ref, alts[0] if alts else None)



# Reads your CSV/TSV as strings (safer for genotype text).

# Finds the sample column names.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--sep", choices=["tab","comma"], default="tab")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ref_mode", choices=["none","genome","founders"], default="none")
    ap.add_argument("--fasta", help="CDC Frontier FASTA (indexed)", default=None)
    ap.add_argument("--founders", nargs="*", default=[])
    args = ap.parse_args()

    sep = "\t" if args.sep=="tab" else ","
    df = pd.read_csv(args.infile, sep=sep, dtype=str)
    df = normalize_input_columns(df)
    sample_cols = detect_samples(df)

    # Build per-SNP allele tuples per sample (before strand correction)
    alleles_by_sample = []
    for s in sample_cols:
        alleles_by_sample.append(df[s].apply(split_geno).values.reshape(-1,1))
    # stack to list-of-tuples per SNP
    # Build per-SNP list of (a1,a2) tuples across all samples
    allele_tuples_per_sample = [df[s].apply(split_geno) for s in sample_cols]
    # zip(*) transposes: now one tuple per SNP, containing (a1,a2) for each sample
    df["_alleles"] = list(zip(*allele_tuples_per_sample))


    # Write MAP
    # Standard PLINK MAP columns (genetic distance unknown -> 0).
    map_df = pd.DataFrame({
        "CHR":  [normalize_chr(c) for c in df["Chr_id"]],
        "SNP":  df["probeset_id"],
        "GD":   ["0"]*len(df),
        "BP":   df["Start"].astype(str)
    })
    map_path = Path(args.out_prefix + ".map")
    map_df.to_csv(map_path, sep="\t", header=False, index=False)

    # # Write PED (FID=IID=sample name; missing sex/phenotype)
    # One PED row per sample: first 6 pedigree/phenotype fields, then two allele fields per SNP (A A, A G, or 0 0 for missing).

    # Order of SNPs in PED matches the MAP (row order).

    fam_cols = ["FID","IID","PID","MID","SEX","PHENO"]
    fam = pd.DataFrame({
        "FID": sample_cols,
        "IID": sample_cols,
        "PID": ["0"]*len(sample_cols),
        "MID": ["0"]*len(sample_cols),
        "SEX": ["0"]*len(sample_cols),
        "PHENO": ["-9"]*len(sample_cols)
    })
    ped_path = Path(args.out_prefix + ".ped")
    with ped_path.open("w") as f:
        for s in sample_cols:
            row = fam.loc[fam["IID"]==s, fam_cols].iloc[0].tolist()
            # flatten this sample's 2-allele calls across SNPs
            al = [split_geno(v) for v in df[s].values]
            flat = [x for t in al for x in t]  # A1,A2 per SNP
            f.write("\t".join(row + flat) + "\n")

    # Build REF/ALT table if requested
    refalt = []
    fasta = None
    if args.ref_mode == "genome":
        try:
            import pysam
            fasta = pysam.FastaFile(args.fasta)
        except Exception as e:
            raise RuntimeError(f"Failed to open FASTA/index: {e}")

    for idx, row in df.iterrows():
        if args.ref_mode == "genome" and fasta is not None:
            ref, alt = call_ref_alt_genome(row, fasta)
        elif args.ref_mode == "founders":
            ref, alt = call_ref_alt_founders(row, args.founders)
        else:
            ref = alt = None
        # fallbacks if needed
        if ref is None:
            # choose modal observed allele as REF
            counts = {}
            for a1,a2 in row["_alleles"]:
                for a in (a1,a2):
                    if a in "ACGT": counts[a]=counts.get(a,0)+1
            if counts:
                ref = max(counts, key=counts.get)
        if alt is None:
            # choose the other allele if present
            obs = set()
            for a1,a2 in row["_alleles"]:
                if a1 in "ACGT": obs.add(a1)
                if a2 in "ACGT": obs.add(a2)
            obs.discard(ref)
            alt = sorted(list(obs))[0] if obs else None
        refalt.append((row["probeset_id"], ref if ref else "0", alt if alt else "0"))

    if refalt:
        ra_path = Path(args.out_prefix + ".alleles_refalt.txt")
        # Decides REF and ALT per SNP using your chosen mode, with sensible fallbacks.

        # This file is meant for PLINK recoding:

        pd.DataFrame(refalt, columns=["SNP","REF","ALT"]).to_csv(ra_path, sep="\t", index=False)
        print(f"Wrote REF/ALT table for PLINK --a2-allele: {ra_path}")

    # Dosage matrix: 0/1/2 copies of ALT (if ALT known); else copies of minor allele
    # Build per-SNP ALT decisions
    alt_dict = {}
    for snp, ref, alt in refalt:
        if alt and alt in "ACGT": alt_dict[snp] = alt
    dosage = pd.DataFrame(index=sample_cols, columns=df["probeset_id"], dtype="float32")
    for i, snp in enumerate(df["probeset_id"]):
        alt = alt_dict.get(snp, None)
        # if ALT not defined, choose cohort-minor allele at this SNP
        if alt is None:
            obs = []
            for s in sample_cols:
                a1,a2 = split_geno(df.at[i, s])
                if a1 in "ACGT": obs.append(a1)
                if a2 in "ACGT": obs.append(a2)
            if obs:
                vals, freqs = np.unique(obs, return_counts=True)
                alt = vals[np.argmin(freqs)]
        # count copies
        for s in sample_cols:
            a1,a2 = split_geno(df.at[i, s])
            if a1 in "ACGT" and a2 in "ACGT":
                dosage.at[s, snp] = (a1==alt) + (a2==alt)
            else:
                dosage.at[s, snp] = np.nan
    dos_path = Path(args.out_prefix + ".ALTdosage.tsv")
    dosage.index.name = "sample"
    # For each SNP, count how many ALT alleles (0/1/2) each sample has.

    # If ALT wasn't decided earlier, use the cohort minor allele at that SNP as a proxy.

    # Note: this is diploid genotype dosage (not intensity-based).

    dosage.to_csv(dos_path, sep="\t")
    print(f"Wrote dosage matrix (ALT copies 0/1/2): {dos_path}")

    print(f"Wrote: {map_path} and {ped_path}")
    print("Next:")
    print(f"  plink --allow-extra-chr --file {args.out_prefix} --make-bed --out {args.out_prefix}_raw")
    print(f"  plink2 --bfile {args.out_prefix}_raw --a2-allele {args.out_prefix}.alleles_refalt.txt 1 2 "
          f"--keep-allele-order --make-bed --out {args.out_prefix}_raw_refA2")

if __name__ == "__main__":
    main()
