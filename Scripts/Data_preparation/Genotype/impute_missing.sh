set -euo pipefail

# Most imputation tools like per-chromosome VCFs.

for chr in CA1 CA2 CA3 CA4 CA5 CA6 CA7 CA8; do
  ./plink2 --bfile qc1 --allow-extra-chr \
  --chr $chr --export vcf bgz ref-first \
  --out vcf_imputed_after_qc1.$chr
  ./tabix -f -p vcf vcf_imputed_after_qc1.$chr.vcf.gz
done

# For chickpea, if you don't have a large external reference panel, run Beagle on your cohort (LD-based imputation). It fills missing GTs and can output dosage and DR2 (imputation quality).
# Beagle 5.x example; add -Xmx memory and threads to taste
for chr in CA1 CA2 CA3 CA4 CA5 CA6 CA7 CA8; do
  java -Xmx16g -jar ./beagle.06Aug24.a91.jar \
    gt=vcf_imputed_after_qc1.$chr.vcf.gz \
    out=imputed.$chr \
    nthreads=8
done

# Post-imputation QC

# Fix the VCF header with contig lines, then index
# Adds missing ##contig=<ID=...,length=...> lines to the Beagle-imputed VCF header, using your reference index as the source of contig names and lengths.
for chr in CA1 CA2 CA3 CA4 CA5 CA6 CA7 CA8; do
  IN="imputed.$chr.vcf.gz"
  test -s "$IN" || { echo "Missing $IN"; exit 1; }
  bcftools reheader -f CDC_Frontier.fa.fai -o imp.$chr.fix.vcf.gz "$IN" || exit 1
  bcftools index -f imp.$chr.fix.vcf.gz || exit 1
done


# Keep variants with good imputation quality: e.g., DR2 >= 0.8 (or 0.7 if you need more).
# Some records may not carry DR2 in INFO; keep those SNPs and only enforce the threshold when DR2 is present.

# Keeps SNPs only (TYPE="snp"), excluding indels and others.
for chr in CA1 CA2 CA3 CA4 CA5 CA6 CA7 CA8; do
  bcftools view -i 'TYPE="snp" && (INFO/DR2="." || INFO/DR2>=0.8)' imp.$chr.fix.vcf.gz -Oz -o imp.qc.$chr.vcf.gz || exit 1
  bcftools index -f imp.qc.$chr.vcf.gz || exit 1
done

# Merge back to a single VCF (optional)
bcftools concat -Oz -o imp.qc.all.vcf.gz imp.qc.CA{1,2,3,4,5,6,7,8}.vcf.gz
bcftools index -f imp.qc.all.vcf.gz



################### Step 8 an 9 #######################

# # create PLINK format files with dosage
# ./plink2 --vcf imp.qc.all.withdc.clean.vcf.gz --allow-extra-chr --make-pgen --out imp.qc.all.withdc.clean

# # Make a BED/BIM/FAM (hard calls)
# ./plink2 --pfile imp.qc.all.withdc.clean \
#   --allow-extra-chr \
#   --hard-call-threshold 0.1 \
#   --make-bed \
#   --out imp.qc.all.withdc.clean


# # Make PED/MAP text files
# ./plink --bfile imp.qc.all.withdc.clean \
#   --allow-extra-chr \
#   --recode \
#   --out imp.qc.all.withdc.clean


# # # 4) PCA / kinship / assoc as needed
# # # PCA (--pca 10): compute top 10 principal components to capture population structure/stratification.
# # # Use PC1-PCn as covariates in association to control structure.

# # # Association (--linear): run additive linear regression per SNP:
# # # y ~ beta0 + betaSNP * (ALT copies) + betaPCs * PCs + betacov * covariates + epsilon

# # # In your *_refA2 set, A1 = ALT, so betaSNP is per ALT allele.

# # # hide-covar: suppress per-covariate lines in the output (keeps the file tidy).

# # # --pheno pheno.txt: file with sample phenotypes (see mini examples below).

# # # --covar covar.txt: file with covariates (e.g., PCs, batch, location, year).

# # ./plink2 --allow-extra-chr --bfile imp.qc.all.withdc --pca 10 --out pca
# # # Example for LL_2019 yield
# # # Merge on FID+IID (tab-delimited)
# # # awk 'NR==FNR{a[$1"\t"$2]=$0; next} FNR==1{print $0"\tPC1\tPC2\tPC3\tPC4\tPC5"} 
# # #      FNR>1{key=$1"\t"$2; if(key in a){split(a[key],b,"\t"); 
# # #              print $0"\t"b[3]"\t"b[4]"\t"b[5]"\t"b[6]"\t"b[7]} }' \
# # #     pca.eigenvec pheno_out/covar_MJ_2019.txt > pheno_out/covar_MJ_2019_withPCs.txt

# # # for ph in pheno_out/flowering_*_*.txt; do
# # #   base=$(basename "$ph" .txt)           # e.g., flowering_MJ_2019
# # #   loc=${base#flowering_}; loc=${loc%_*} # MJ
# # #   yr=${base##*_}                        # 2019
# # #   ./plink2 --allow-extra-chr --bfile qc1 \
# # #     --pheno "$ph" \
# # #     --covar "pheno_out/covar_${loc}_${yr}_withPCs.txt" \
# # #     --covar-name Year,PC1,PC2,PC3,PC4,PC5 \
# # #     --glm hide-covar omit-ref \
# # #     --out "gwas_${base}"
# # # done

# # # ./plink2 --allow-extra-chr --bfile qc1 --linear hide-covar --pheno pheno_put/flowering.txt --covar covar.txt --out gwas
# # # ./plink2 --bfile qc1 --genome full --out related
