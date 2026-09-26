#!/usr/bin/env python3
"""
scripts/verify_determinism.py — Determinism Verification

Runs inference twice on the same representative sample into two separate directories:
  - run1: output/determinism_run1/
  - run2: output/determinism_run2/

Compares:
  - SHA256 hash of matching_results.tsv and candidate_pairs.tsv
  - Line-by-line exact equality
  - Selected entity IDs and candidate sets
"""

import os
import sys
import hashlib
import time
import shutil
import polars as pl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infer import run_inference


class Args:
    def __init__(self, out_dir):
        self.test_dir = "dataset/test_sample"
        self.model_path = "models/matching_model.pkl"
        self.output_dir = out_dir
        self.chunk_size = 500
        self.max_candidates = 40
        self.max_s1_records = 1000
        self.max_target_records = 25000
        self.threshold = 0.72
        self.no_validate = True


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print(" REQUIREMENT 7: DETERMINISM VERIFICATION")
    print("=" * 80)

    dir1 = os.path.join(PROJECT_ROOT, "output", "determinism_run1")
    dir2 = os.path.join(PROJECT_ROOT, "output", "determinism_run2")

    for d in [dir1, dir2]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print("\n--- Executing Run 1 ---")
    args1 = Args(dir1)
    run_inference(args1)

    print("\n--- Executing Run 2 ---")
    args2 = Args(dir2)
    run_inference(args2)

    # Compare hashes
    m1 = os.path.join(dir1, "matching_results.tsv")
    m2 = os.path.join(dir2, "matching_results.tsv")
    c1 = os.path.join(dir1, "candidate_pairs.tsv")
    c2 = os.path.join(dir2, "candidate_pairs.tsv")

    hash_m1 = sha256_file(m1)
    hash_m2 = sha256_file(m2)
    hash_c1 = sha256_file(c1)
    hash_c2 = sha256_file(c2)

    match_equal = (hash_m1 == hash_m2)
    cand_equal = (hash_c1 == hash_c2)

    # Line by line check
    with open(m1, "r", encoding="utf-8") as f1, open(m2, "r", encoding="utf-8") as f2:
        m1_lines = f1.readlines()
        m2_lines = f2.readlines()

    lines_equal = (m1_lines == m2_lines)

    print("\n" + "=" * 80)
    print(" DETERMINISM COMPARISON RESULTS:")
    print("=" * 80)
    print(f"  matching_results.tsv Run 1 SHA256: {hash_m1}")
    print(f"  matching_results.tsv Run 2 SHA256: {hash_m2}")
    print(f"  Matching Files Identical:          {match_equal} (Line-by-line: {lines_equal})")
    print(f"  candidate_pairs.tsv  Run 1 SHA256: {hash_c1}")
    print(f"  candidate_pairs.tsv  Run 2 SHA256: {hash_c2}")
    print(f"  Candidate Files Identical:         {cand_equal}")
    print(f"  Total Lines Compared:              {len(m1_lines):,} rows")
    print("=" * 80)

    if match_equal and cand_equal:
        print("✓ DETERMINISM VERIFIED: 100% BIT-FOR-BIT IDENTICAL ACROSS REPEATED RUNS.")
    else:
        print("✗ DETERMINISM FAILED: Output differences detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
