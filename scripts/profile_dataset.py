#!/usr/bin/env python3
"""
scripts/profile_dataset.py — Comprehensive Real Dataset Profiler for Entity Resolution

Profiles the ACTUAL challenge datasets using fast, memory-efficient streaming.
Handles 26+ million records safely within 16 GB RAM.

Outputs:
  - reports/dataset_profile.json
  - reports/dataset_profile_summary.txt
"""

import os
import sys
import json
import time
import unicodedata
import collections
from typing import Dict, Any

# Ensure stdout uses UTF-8 encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import polars as pl
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


# ─── Unicode Script Profiling ───────────────────────────────────────────────

INDIC_AND_WORLD_SCRIPTS = [
    ("DEVANAGARI", (0x0900, 0x097F)),
    ("BENGALI", (0x0980, 0x09FF)),
    ("GURMUKHI", (0x0A00, 0x0A7F)),
    ("GUJARATI", (0x0A80, 0x0AFF)),
    ("ORIYA", (0x0B00, 0x0B7F)),
    ("TAMIL", (0x0B80, 0x0BFF)),
    ("TELUGU", (0x0C00, 0x0C7F)),
    ("KANNADA", (0x0C80, 0x0CFF)),
    ("MALAYALAM", (0x0D00, 0x0D7F)),
    ("ARABIC", (0x0600, 0x06FF)),
    ("CYRILLIC", (0x0400, 0x04FF)),
    ("LATIN_EXTENDED", (0x0080, 0x024F)),
]


def profile_scripts_in_sample(df: pl.DataFrame, sample_size: int = 50_000) -> Dict[str, int]:
    """Sample text rows and count detected Unicode scripts."""
    sampled_df = df.sample(min(sample_size, len(df)), seed=42)
    counts = collections.Counter()
    
    for col in ["business_name", "business_address"]:
        if col not in sampled_df.columns:
            continue
        for text in sampled_df[col].drop_nulls():
            for ch in text:
                cp = ord(ch)
                if cp < 128:
                    continue
                matched = False
                for script_name, (lo, hi) in INDIC_AND_WORLD_SCRIPTS:
                    if lo <= cp <= hi:
                        counts[script_name] += 1
                        matched = True
                        break
                if not matched:
                    # Generic unicode category name
                    cat = unicodedata.name(ch, "UNKNOWN").split()[0]
                    counts[cat] += 1
                    
    return dict(counts.most_common(15))


# ─── Source File Profiling ───────────────────────────────────────────────────

def profile_source_file(filepath: str, label: str) -> Dict[str, Any]:
    """Profile a single TSV source file efficiently using Polars."""
    if not os.path.isfile(filepath):
        return {"status": "missing", "filepath": filepath, "label": label}

    t0 = time.time()
    file_size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)
    print(f"  Scanning {label} ({file_size_mb} MB)...")

    # Read TSV with all columns as utf-8 strings
    df = pl.read_csv(
        filepath,
        separator="\t",
        infer_schema_length=0,
        truncate_ragged_lines=True
    )
    total_rows = len(df)
    columns = df.columns

    # Unique & duplicate IDs
    if "entity_id" in columns:
        unique_ids = df["entity_id"].n_unique()
        duplicate_ids = total_rows - unique_ids
    else:
        unique_ids, duplicate_ids = 0, 0

    # Country distribution
    country_dist = {}
    empty_country_count = 0
    if "country" in columns:
        country_counts = df["country"].fill_null("").value_counts()
        for row in country_counts.iter_rows():
            ctry = str(row[0]).strip()
            count = int(row[1])
            if ctry == "":
                empty_country_count += count
            else:
                country_dist[ctry] = count

    # Empty names & addresses
    empty_names = 0
    name_stats = {}
    top_names = []
    if "business_name" in columns:
        name_col = df["business_name"].fill_null("")
        empty_names = int((name_col == "").sum())
        name_lens = name_col.str.len_chars()
        name_stats = {
            "min": int(name_lens.min() or 0),
            "max": int(name_lens.max() or 0),
            "mean": round(float(name_lens.mean() or 0), 2),
            "median": float(name_lens.median() or 0),
            "p90": float(name_lens.quantile(0.90) or 0),
            "p95": float(name_lens.quantile(0.95) or 0),
            "p99": float(name_lens.quantile(0.99) or 0),
        }
        # Top 10 most frequent names
        top_name_df = name_col.filter(name_col != "").value_counts().sort("count", descending=True).head(10)
        top_names = [f"{r[0]} ({r[1]:,})" for r in top_name_df.iter_rows()]

    empty_addrs = 0
    addr_stats = {}
    if "business_address" in columns:
        addr_col = df["business_address"].fill_null("")
        empty_addrs = int((addr_col == "").sum())
        addr_lens = addr_col.str.len_chars()
        addr_stats = {
            "min": int(addr_lens.min() or 0),
            "max": int(addr_lens.max() or 0),
            "mean": round(float(addr_lens.mean() or 0), 2),
            "median": float(addr_lens.median() or 0),
            "p90": float(addr_lens.quantile(0.90) or 0),
            "p95": float(addr_lens.quantile(0.95) or 0),
            "p99": float(addr_lens.quantile(0.99) or 0),
        }

    # Detect scripts in sample
    script_distribution = profile_scripts_in_sample(df, sample_size=50_000)

    elapsed = round(time.time() - t0, 2)
    print(f"    Completed in {elapsed}s: {total_rows:,} rows, {unique_ids:,} unique IDs.")

    return {
        "status": "loaded",
        "label": label,
        "filepath": filepath,
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "columns": columns,
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "country_distribution": country_dist,
        "empty_names": empty_names,
        "empty_addresses": empty_addrs,
        "empty_countries": empty_country_count,
        "empty_name_pct": round(100.0 * empty_names / max(total_rows, 1), 3),
        "empty_addr_pct": round(100.0 * empty_addrs / max(total_rows, 1), 3),
        "empty_country_pct": round(100.0 * empty_country_count / max(total_rows, 1), 3),
        "name_length_stats": name_stats,
        "addr_length_stats": addr_stats,
        "top_frequent_names": top_names,
        "unicode_script_distribution": script_distribution,
        "scan_time_sec": elapsed
    }


# ─── Ground Truth Profiling ──────────────────────────────────────────────────

def profile_ground_truth(filepath: str) -> Dict[str, Any]:
    """Profile train_ground_truth.tsv in detail."""
    if not os.path.isfile(filepath):
        return {"status": "missing", "filepath": filepath}

    t0 = time.time()
    file_size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)
    print(f"  Scanning Ground Truth ({file_size_mb} MB)...")

    df = pl.read_csv(
        filepath,
        separator="\t",
        infer_schema_length=0,
        truncate_ragged_lines=True
    )
    total_s1 = len(df)
    unique_s1 = df["source1_entity_id"].n_unique()

    # Parse matched entity IDs
    matches_col = df["matched_entity_ids"].fill_null("")
    
    # Identify singletons
    singletons_mask = (matches_col == "")
    singleton_count = int(singletons_mask.sum())
    non_singleton_count = total_s1 - singleton_count

    # Split matches into counts
    # For rows with matches, count comma-separated elements
    non_empty = matches_col.filter(~singletons_mask)
    match_lengths = non_empty.str.split(",").list.len()

    # Overall match cardinality stats (including 0 for singletons)
    all_cardinalities = pl.concat([
        pl.Series("card", [0] * singleton_count, dtype=pl.UInt32),
        match_lengths.cast(pl.UInt32)
    ])

    card_stats = {
        "min": int(all_cardinalities.min() or 0),
        "max": int(all_cardinalities.max() or 0),
        "mean": round(float(all_cardinalities.mean() or 0), 3),
        "median": float(all_cardinalities.median() or 0),
        "p90": float(all_cardinalities.quantile(0.90) or 0),
        "p95": float(all_cardinalities.quantile(0.95) or 0),
        "p99": float(all_cardinalities.quantile(0.99) or 0),
    }

    # Frequency histogram of match counts
    card_hist_df = all_cardinalities.value_counts().sort("card")
    card_hist = {int(r[0]): int(r[1]) for r in card_hist_df.head(20).iter_rows()}

    # Source origin breakdown (S2 vs S3)
    # Sample 100k non-empty rows for fast exact counting
    s2_count, s3_count = 0, 0
    sample_matches = non_empty.sample(min(100_000, len(non_empty)), seed=42)
    for raw in sample_matches:
        for mid in raw.split(","):
            mid = mid.strip()
            if mid.startswith("S2-"):
                s2_count += 1
            elif mid.startswith("S3-"):
                s3_count += 1

    total_true_matches = int(match_lengths.sum() or 0)
    elapsed = round(time.time() - t0, 2)
    print(f"    Completed in {elapsed}s: {total_s1:,} S1, {total_true_matches:,} true matches.")

    return {
        "status": "loaded",
        "filepath": filepath,
        "file_size_mb": file_size_mb,
        "total_s1_records": total_s1,
        "unique_s1_ids": unique_s1,
        "total_true_matches": total_true_matches,
        "singleton_count": singleton_count,
        "singleton_pct": round(100.0 * singleton_count / max(total_s1, 1), 3),
        "non_singleton_count": non_singleton_count,
        "match_count_stats": card_stats,
        "match_cardinality_histogram": card_hist,
        "source2_share_pct": round(100.0 * s2_count / max(s2_count + s3_count, 1), 1),
        "source3_share_pct": round(100.0 * s3_count / max(s2_count + s3_count, 1), 1),
        "scan_time_sec": elapsed
    }


# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print(" Amazon ML Challenge — Real Data Profiling (Phase 2)")
    print("=" * 75)
    print(f" Data Directory: {os.path.abspath(data_root)}")
    print(f" Reports Output: {os.path.abspath(reports_dir)}")
    overall_t0 = time.time()

    profile = {"train": {}, "test": {}, "ground_truth": {}}

    # 1. Profile Training Sources
    print("\n━━━ [1/3] Profiling Training Sources ━━━")
    train_dir = os.path.join(data_root, "train")
    for s_idx in [1, 2, 3]:
        fname = f"train_source{s_idx}.tsv"
        fpath = os.path.join(train_dir, fname)
        profile["train"][f"source{s_idx}"] = profile_source_file(fpath, f"Train Source {s_idx}")

    # 2. Profile Ground Truth
    print("\n━━━ [2/3] Profiling Training Ground Truth ━━━")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    profile["ground_truth"] = profile_ground_truth(gt_path)

    # 3. Profile Test Sources
    print("\n━━━ [3/3] Profiling Test Sources ━━━")
    test_dir = os.path.join(data_root, "test")
    for s_idx in [1, 2, 3]:
        fname = f"test_source{s_idx}.tsv"
        fpath = os.path.join(test_dir, fname)
        profile["test"][f"source{s_idx}"] = profile_source_file(fpath, f"Test Source {s_idx}")

    total_time = round(time.time() - overall_t0, 2)
    profile["total_execution_time_sec"] = total_time

    # ── Write Machine-Readable JSON ──
    json_path = os.path.join(reports_dir, "dataset_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved machine-readable report: {json_path}")

    # ── Write Human-Readable Summary ──
    summary_path = os.path.join(reports_dir, "dataset_profile_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("       BUSINESS ENTITY RESOLUTION — REAL DATASET PROFILE REPORT\n")
        f.write("=" * 78 + "\n\n")

        for split in ["train", "test"]:
            f.write(f"━" * 50 + f"\n  {split.upper()} DATASETS\n" + f"━" * 50 + "\n")
            for s_key in ["source1", "source2", "source3"]:
                info = profile[split].get(s_key, {})
                if info.get("status") != "loaded":
                    f.write(f"  {s_key}: NOT LOADED\n")
                    continue
                f.write(f"\n  [{info['label'].upper()}] ({info['filepath']})\n")
                f.write(f"    - File Size:       {info['file_size_mb']:>10.1f} MB\n")
                f.write(f"    - Total Records:   {info['total_rows']:>10,}\n")
                f.write(f"    - Unique IDs:      {info['unique_ids']:>10,} (Duplicates: {info['duplicate_ids']})\n")
                f.write(f"    - Missing Names:   {info['empty_names']:>10,} ({info['empty_name_pct']:.2f}%)\n")
                f.write(f"    - Missing Addrs:   {info['empty_addresses']:>10,} ({info['empty_addr_pct']:.2f}%)\n")
                f.write(f"    - Missing Ctry:    {info['empty_countries']:>10,} ({info['empty_country_pct']:.2f}%)\n")
                f.write(f"    - Countries:       {info['country_distribution']}\n")
                f.write(f"    - Name Lengths:    mean={info['name_length_stats'].get('mean')} chars, "
                        f"median={info['name_length_stats'].get('median')}, "
                        f"p95={info['name_length_stats'].get('p95')}, "
                        f"max={info['name_length_stats'].get('max')}\n")
                f.write(f"    - Addr Lengths:    mean={info['addr_length_stats'].get('mean')} chars, "
                        f"median={info['addr_length_stats'].get('median')}, "
                        f"p95={info['addr_length_stats'].get('p95')}, "
                        f"max={info['addr_length_stats'].get('max')}\n")
                f.write(f"    - Top Names:       {', '.join(info['top_frequent_names'][:5])}\n")
                f.write(f"    - Unicode Scripts: {info['unicode_script_distribution']}\n")
            f.write("\n")

        gt = profile.get("ground_truth", {})
        if gt.get("status") == "loaded":
            f.write(f"━" * 50 + "\n  GROUND TRUTH ASSOCIATIONS\n" + f"━" * 50 + "\n")
            f.write(f"    - Total S1 Entities:     {gt['total_s1_records']:>10,}\n")
            f.write(f"    - Total True Matches:    {gt['total_true_matches']:>10,}\n")
            f.write(f"    - Singletons (0 match):  {gt['singleton_count']:>10,} ({gt['singleton_pct']:.2f}%)\n")
            f.write(f"    - Non-Singletons:        {gt['non_singleton_count']:>10,}\n")
            f.write(f"    - Origin Share:          Source 2: {gt['source2_share_pct']}%, Source 3: {gt['source3_share_pct']}%\n")
            f.write(f"    - Matches / Entity:      mean={gt['match_count_stats'].get('mean')}, "
                    f"median={gt['match_count_stats'].get('median')}, "
                    f"p90={gt['match_count_stats'].get('p90')}, "
                    f"p95={gt['match_count_stats'].get('p95')}, "
                    f"max={gt['match_count_stats'].get('max')}\n")
            f.write(f"    - Multiplicity Hist:     {gt['match_cardinality_histogram']}\n\n")

        f.write(f"  Total Profiling Runtime: {total_time}s\n")
        f.write("=" * 78 + "\n")

    print(f"✓ Saved human-readable summary: {summary_path}")
    print(f"\nCompleted in {total_time}s across all datasets!")


if __name__ == "__main__":
    main()
