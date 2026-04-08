#!/usr/bin/env python3
# python3 add_ds_from_gt.py qc1.vcf.gz qc1.withDS.vcf.gz

import sys
import pysam

if len(sys.argv) != 3:
    sys.exit("Usage: add_ds_from_gt.py <in.vcf.gz> <out.vcf.gz>")

in_vcf  = sys.argv[1]
out_vcf = sys.argv[2]

# Open input
inp = pysam.VariantFile(in_vcf)  # works even if not indexed; index just removes warning

# *** Add DS to the INPUT header first ***
if "DS" not in inp.header.formats:
    inp.header.formats.add("DS", 1, "Float", "ALT1 dosage derived from GT (0/1/2)")

# Create output using the (now augmented) input header
out = pysam.VariantFile(out_vcf, "wz", header=inp.header)

for rec in inp:
    # DS counts copies of first ALT (allele index 1). Biallelic arrays fit this perfectly.
    for smp in rec.samples:
        gt = rec.samples[smp].get("GT")
        if gt is None or any(a is None for a in gt):
            ds = None
        else:
            ds = float(sum(1 for a in gt if a == 1))
        rec.samples[smp]["DS"] = ds
    out.write(rec)

inp.close()
out.close()

# Index the output
pysam.tabix_index(out_vcf, preset="vcf", force=True)
print(f"Wrote {out_vcf} and {out_vcf}.tbi")
