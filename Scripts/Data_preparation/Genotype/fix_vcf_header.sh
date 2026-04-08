# 1) Extract current sample IDs
bcftools query -l imp.qc.all.withdc.vcf.gz > samples.txt

# 2) Build a cleaned list: if left == right around the last underscore, keep left; else keep original
python3 - << 'PY'
import sys
clean=[]
with open("samples.txt") as f:
    for s in map(str.strip, f):
        # split once on the LAST underscore only
        if "_" in s:
            left, right = s.rsplit("_", 1)
            clean.append(left if left == right else s)
        else:
            clean.append(s)
open("samples_clean.txt","w").write("\n".join(clean)+"\n")
PY

# 3) Reheader with the cleaned names
bcftools reheader -s samples_clean.txt -o imp.qc.all.withdc.clean.vcf.gz imp.qc.all.withdc.vcf.gz
./tabix -f -p vcf imp.qc.all.withdc.clean.vcf.gz