import os
import re
import glob
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
LOG_DIR = "/birl2/data/brassica/thulani/Research/CMPT898/CMPT-PLSC_819_Project/Paper_revisions/Dataset4_Brassica_NAM"  # <-- your folder with logN.out files
PATTERNS = ["*.out", "*.log", "log*.out", "admixture*.out"]     # file patterns to scan

# Regexes that usually catch ADMIXTURE summaries
# Typical line: "CV error (K=12): 0.69281"
RE_K_CV = re.compile(r"CV\s*error\s*\(K\s*=\s*(\d+)\)\s*:\s*([0-9]*\.?[0-9]+)")
# Fallback: sometimes you only get "CV error: 0.12345" and K appears earlier in the file
RE_CV_ONLY = re.compile(r"CV\s*error\s*:\s*([0-9]*\.?[0-9]+)")
RE_K_ONLY = re.compile(r"\(K\s*=\s*(\d+)\)")

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_log(path):
    """
    Returns (K, CV) if found, else (None, None).
    If multiple matches exist, uses the last one (final CV after training).
    """
    text = read_text(path)

    # Primary: direct "(K=..): CV"
    matches = RE_K_CV.findall(text)
    if matches:
        k, cv = matches[-1]  # last occurrence
        return int(k), float(cv)

    # Fallback: find last CV and last K separately
    cv_matches = RE_CV_ONLY.findall(text)
    k_matches  = RE_K_ONLY.findall(text)
    if cv_matches and k_matches:
        return int(k_matches[-1]), float(cv_matches[-1])

    # Last resort: infer K from filename if it contains something like "...K12..." or "..._12_..."
    # (This is less reliable, so we only use it if we DID find a CV in the text)
    if cv_matches:
        # try to pull an integer close to 'K' in filename
        fname = os.path.basename(path)
        # Try "...K12..." or "...k12..."
        m = re.search(r"[Kk]\s*[_-:]?\s*(\d+)", fname)
        if m:
            return int(m.group(1)), float(cv_matches[-1])
        # Try a plain number in filename (not great, but a final fallback)
        m2 = re.search(r"(\d+)", fname)
        if m2:
            return int(m2.group(1)), float(cv_matches[-1])

    return None, None

# Collect files
files = []
for pat in PATTERNS:
    files.extend(glob.glob(os.path.join(LOG_DIR, pat)))
files = sorted(set(files))

if not files:
    raise FileNotFoundError(f"No log files found in: {LOG_DIR}")

by_k = defaultdict(list)
skipped = []

for fp in files:
    k, cv = parse_log(fp)
    if k is None or cv is None or math.isnan(cv):
        skipped.append(fp)
    else:
        by_k[k].append(cv)

if not by_k:
    raise RuntimeError("No parsable (K, CV) pairs found in logs. "
                       "Check the regex or a sample log format.")

# Aggregate: mean and std per K
ks = sorted(by_k.keys())
means = [float(np.mean(by_k[k])) for k in ks]
stds  = [float(np.std(by_k[k], ddof=1)) if len(by_k[k]) > 1 else 0.0 for k in ks]
counts = [len(by_k[k]) for k in ks]

# Report best K
best_idx = int(np.argmin(means))
best_k, best_cv = ks[best_idx], means[best_idx]

print(f"Parsed {sum(counts)} runs across {len(ks)} K values.")
if skipped:
    print(f"Skipped {len(skipped)} file(s) without parsable K/CV:")
    for s in skipped:
        print("  -", os.path.basename(s))
print("\nPer-K summary:")
for k, m, s, c in zip(ks, means, stds, counts):
    if s > 0:
        print(f"  K={k:>2} | CV={m:.5f} +/- {s:.5f} (n={c})")
    else:
        print(f"  K={k:>2} | CV={m:.5f}          (n={c})")

print(f"\nBest K (min mean CV): K={best_k} with CV={best_cv:.5f}")

# Plot
# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(ks, means, yerr=stds, marker="o", linestyle="-", capsize=3, label="CV error (mean +/- sd)")
plt.xlabel("K (number of ancestral populations)")
plt.ylabel("CV error")
plt.title("ADMIXTURE CV Error vs K (parsed from logs)")
plt.grid(True)
plt.legend()

# Optional: elbow guide bands
plt.axvspan(6, 10, alpha=0.1, label="Broad elbow (~K=6-10)")
plt.axvspan(15, 25, alpha=0.08, label="Fine-scale range (~K=15-25)")
plt.legend()

plt.tight_layout()

# === SAVE FIGURES ===
outdir = "./figures"
os.makedirs(outdir, exist_ok=True)

png_path = os.path.join(outdir, "admixture_cv_error_D4.png")
pdf_path = os.path.join(outdir, "admixture_cv_error_D4.pdf")

plt.savefig(png_path, dpi=300)   # high-res PNG
plt.savefig(pdf_path)            # vector PDF for publication
print(f"Figures saved to:\n  {png_path}\n  {pdf_path}")

plt.show()
