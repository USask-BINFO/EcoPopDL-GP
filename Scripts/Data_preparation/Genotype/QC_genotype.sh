
# 1) Binary files already made:
#    chickpea_axiom_refA2.bed/.bim/.fam

# 2) Basic variant/sample QC
# --geno 0.02: drop SNPs with >2% missing calls across samples (low-quality or bad clusters).

# --mind 0.02: drop samples with >2% missing genotypes (poor DNA/sample).

# --maf 0.01: keep SNPs with minor allele frequency >=1% (very rare variants can be unstable in small panels).

./plink2 --bfile chickpea_axiom_raw_refA2 \
  --geno 0.02 --mind 0.02 --maf 0.01 \
  --allow-extra-chr \
  --make-bed --out qc1

# (Optional) Skip or loosen HWE in highly inbred/selfing panels.
# e.g., don't use strict --hwe filters, or apply only within subpopulations.

# 3) (Optional) drop strand-ambiguous palindromic SNPs with high MAF
#    Safer when merging with external data
# Palindromic SNPs: A/T and C/G look the same after strand flips; when MAF is high (~50%), you can't tell if an allele label mismatch is just a strand issue -> risky to merge with other datasets.

# --exclude-if-a1-a2 AT,TA,CG,GC with --maf 0.4: remove only those palindromic SNPs whose MAF >= 0.4 (near-symmetric).
# Low-MAF palindromics are usually fine to keep; high-MAF ones are the troublemakers.

# Output is qc2 (even safer for merges/meta-analyses).

# ./plink --bfile qc1 --exclude-if-a1-a2 AT,TA,CG,GC --maf 0.4 \
#   --make-bed --out qc2

# Make PED/MAP (text)
plink --bfile qc1 \
  --allow-extra-chr \
  --recode \
  --out qc1

# Make a VCF with correct REF/ALT
# PLINK 2.0 (VCF with GT and DS dosage)
./plink2 --bfile qc1 --allow-extra-chr \
  --export vcf bgz id-paste=fid \
  --out qc1
# creates chickpea_axiom_refA2.vcf.gz with FORMAT: GT and DS
./tabix -f -p vcf qc1.vcf.gz
