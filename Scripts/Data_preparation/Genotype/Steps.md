Goal: take an Axiom SNP-array table (rows=SNPs, columns=samples + metadata), turn it into PLINK PED/MAP, and also decide REF/ALT either from a reference genome or from founder/controls. Finally, produce a dosage matrix (0/1/2 copies of ALT per sample).

Outputs it writes:

    PREFIX.map - SNP coordinates (CHR, SNP_ID, 0, BP)

    PREFIX.ped - PLINK PED with two allele fields per SNP for each sample

    PREFIX.alleles_refalt.txt - a 3-column table (SNP REF ALT) you feed to PLINK (--a2-allele) so A2 becomes REF

    PREFIX.ALTdosage.tsv - dosage (0/1/2 ALT copies) per samplexSNP

Run axiom_snp_to_plink_refalt.py
Step 1: Run axiom_snp_to_plink_refalt.py file
    This will convert our axiom formatted snp genotype file to plink .ped and .map
    Have reference and alternate allele from reference genome fasta file

make_plink_bed.sh
Step 2: Run PLINK to generate .bed

Step 3: Run PLINK to generate ref and alternate based .bed files

QC_genotype.sh

Step 4: QC steps
    Basic variant/sample QC
    (Optional) Skip or loosen HWE in highly inbred/selfing panels.
    (Optional) drop strand-ambiguous palindromic SNPs with high MAF


Step 5: Create vcf

impute_missing.sh
Step 6: impute missing SNP
    Create chromosome-wise vcf
    Use beagle to impute
    Merge chromosome-wise vcf to one file

Step 7: PCA / kinship / assoc as needed
    make_pheno_files.py

add_ds_from_gt.py
Step 8: add dosage information to vcf file
    Run this to get dosage plink files: ./plink2 --vcf imp.qc.all.withdc.vcf.gz --allow-extra-chr --make-pgen --out imp.qc.all.withdc

Step 9: Fix VCF header in imp.qc.all.withdc.vcf.gz
    fix_vcf_header.sh

Step 10: Create PLINK bed files and .ped and .map files
    Run impute_missing.sh
    
    # Make a BED/BIM/FAM (hard calls)
    ./plink2 --pfile imp.qc.all.withdc \
    --allow-extra-chr \
    --hard-call-threshold 0.1 \
    --make-bed \
    --out imp.qc.all.withdc


    # Make PED/MAP text files
    ./plink --bfile imp.qc.all.withdc \
    --allow-extra-chr \
    --recode \
    --out imp.qc.all.withdc



