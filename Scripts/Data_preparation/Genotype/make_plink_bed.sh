# ./plink --allow-extra-chr --file chickpea_axiom --make-bed --out chickpea_axiom

# Convert to binary PLINK, enforce A2=REF from the file we write
./plink --allow-extra-chr --file chickpea_axiom --make-bed --out chickpea_axiom_raw
./plink2 --bfile chickpea_axiom_raw --a2-allele chickpea_axiom.alleles_refalt.txt 1 2 \
      --keep-allele-order --make-bed --out chickpea_axiom_raw_refA2
