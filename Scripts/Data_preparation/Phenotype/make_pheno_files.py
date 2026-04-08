#!/usr/bin/env python3
"""Backward-compatible wrapper for the generic phenotype file builder."""

from pathlib import Path
import sys


UTILS_DIR = Path(__file__).resolve().parents[2] / "utils"
sys.path.insert(0, str(UTILS_DIR))

from make_pheno_files_cli import main


if __name__ == "__main__":
    main()
