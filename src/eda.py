#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/eda.py
#
# Description:
#   Data Audit & Exploratory Data Analysis (EDA) Module.
#   Implements Stage 1 of the recommended architecture:
#     - Inspects dataset columns, shapes, null/missing counts, and duplicates.
#     - Explicitly reads TSV with sep='\t' to prevent single-column corruption.
#     - Analyzes country distributions across train (US, India) and test (France, etc.)
#       to verify Open-Set compliance.
#     - Analyzes match multiplicity:
#         * Singletons (Source 1 entities with 0 matches)
#         * 1-to-1 matches
#         * 1-to-Many matches
#         * Split between Source 2 and Source 3 matches
#     - Outputs an actionable audit report.
# ==============================================================================

# Standard library imports
import os
import collections
from typing import Dict, List, Set, Any, Optional

# Third-party data manipulation library
import pandas as pd


def audit_single_source_file(filepath: str, source_label: str) -> Dict[str, Any]:
    """
    Audits a single source TSV file (*_source1.tsv, *_source2.tsv, or *_source3.tsv).
    
    Args:
        filepath (str): Path to the TSV file.
        source_label (str): Descriptive label (e.g., 'Train Source 1').
        
    Returns:
        Dict[str, Any]: Audit statistics including row count, null counts, countries, etc.
    """
    if not os.path.isfile(filepath):
        return {"status": "missing", "filepath": filepath}

    # CRITICAL: Always use sep='\t' to prevent silent corruption where commas split addresses
    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

    total_rows = len(df)
    unique_ids = df["entity_id"].nunique() if "entity_id" in df.columns else 0
    duplicate_ids = total_rows - unique_ids

    # Country distribution
    country_counts = df["country"].value_counts().to_dict() if "country" in df.columns else {}

    # Missing/empty value counts
    empty_names = (df["business_name"].str.strip() == "").sum() if "business_name" in df.columns else 0
    empty_addresses = (df["business_address"].str.strip() == "").sum() if "business_address" in df.columns else 0
    empty_countries = (df["country"].str.strip() == "").sum() if "country" in df.columns else 0

    return {
        "status": "loaded",
        "filepath": filepath,
        "source_label": source_label,
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "country_distribution": country_counts,
        "empty_names": int(empty_names),
        "empty_addresses": int(empty_addresses),
        "empty_countries": int(empty_countries),
        "columns": list(df.columns)
    }


def audit_ground_truth_file(filepath: str) -> Dict[str, Any]:
    """
    Audits the training ground truth file (train_ground_truth.tsv).
    
    Args:
        filepath (str): Path to train_ground_truth.tsv.
        
    Returns:
        Dict[str, Any]: Statistics on match multiplicity, singletons, and source origins.
    """
    if not os.path.isfile(filepath):
        return {"status": "missing", "filepath": filepath}

    # Read tab-separated ground truth
    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

    total_s1 = len(df)
    singleton_count = 0
    multi_match_count = 0
    single_match_count = 0
    total_matches = 0
    match_distribution = collections.defaultdict(int)

    s2_match_count = 0
    s3_match_count = 0

    for _, row in df.iterrows():
        raw_matches = str(row.get("matched_entity_ids", "")).strip()
        if not raw_matches:
            singleton_count += 1
            match_distribution[0] += 1
            continue

        match_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
        num_matches = len(match_ids)
        total_matches += num_matches
        match_distribution[num_matches] += 1

        if num_matches == 1:
            single_match_count += 1
        elif num_matches > 1:
            multi_match_count += 1

        for mid in match_ids:
            if mid.startswith("S2-"):
                s2_match_count += 1
            elif mid.startswith("S3-"):
                s3_match_count += 1

    return {
        "status": "loaded",
        "total_s1_records": total_s1,
        "total_true_matches": total_matches,
        "singleton_count": singleton_count,
        "singleton_percentage": (singleton_count / max(total_s1, 1)) * 100.0,
        "single_match_count": single_match_count,
        "multi_match_count": multi_match_count,
        "match_multiplicity_histogram": dict(sorted(match_distribution.items())),
        "matches_from_source2": s2_match_count,
        "matches_from_source3": s3_match_count
    }


def run_full_data_audit(data_root: str = "dataset") -> Dict[str, Any]:
    """
    Runs complete audit over both training and test datasets.
    
    Args:
        data_root (str): Root directory containing 'train/' and 'test/' folders.
        
    Returns:
        Dict[str, Any]: Aggregated audit report.
    """
    print("=" * 75)
    print(" Amazon ML Challenge - Dataset Audit & EDA Report")
    print("=" * 75)

    report = {"train": {}, "test": {}, "ground_truth": {}}

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    # 1. Audit Training Sources
    print("\n--- [1] Training Source Datasets ---")
    for s_idx in [1, 2, 3]:
        fname = f"train_source{s_idx}.tsv"
        fpath = os.path.join(train_dir, fname)
        info = audit_single_source_file(fpath, f"Train Source {s_idx}")
        report["train"][f"source{s_idx}"] = info
        if info.get("status") == "loaded":
            print(f" Source {s_idx} ({fname}):")
            print(f"   * Total Records     : {info['total_rows']:,}")
            print(f"   * Unique IDs        : {info['unique_ids']:,} (Duplicates: {info['duplicate_ids']})")
            print(f"   * Countries Observed: {info['country_distribution']}")
            print(f"   * Empty Name/Addr   : {info['empty_names']} names, {info['empty_addresses']} addrs")
        else:
            print(f" Source {s_idx} ({fname}): NOT FOUND at {fpath}")

    # 2. Audit Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print("\n--- [2] Ground Truth Match Analysis ---")
    gt_info = audit_ground_truth_file(gt_path)
    report["ground_truth"] = gt_info
    if gt_info.get("status") == "loaded":
        print(f" Ground Truth ({gt_path}):")
        print(f"   * Total Source 1 Entities : {gt_info['total_s1_records']:,}")
        print(f"   * Total True Matches      : {gt_info['total_true_matches']:,}")
        print(f"   * Singletons (0 matches)  : {gt_info['singleton_count']:,} ({gt_info['singleton_percentage']:.2f}%)")
        print(f"   * 1-to-1 Matches          : {gt_info['single_match_count']:,}")
        print(f"   * 1-to-Many Matches       : {gt_info['multi_match_count']:,}")
        print(f"   * Matches from Source 2   : {gt_info['matches_from_source2']:,}")
        print(f"   * Matches from Source 3   : {gt_info['matches_from_source3']:,}")
        print(f"   * Multiplicity Histogram  : {gt_info['match_multiplicity_histogram']}")
    else:
        print(f" Ground Truth file NOT FOUND at {gt_path}")

    # 3. Audit Test Sources (if present)
    print("\n--- [3] Test Source Datasets ---")
    for s_idx in [1, 2, 3]:
        fname = f"test_source{s_idx}.tsv"
        fpath = os.path.join(test_dir, fname)
        info = audit_single_source_file(fpath, f"Test Source {s_idx}")
        report["test"][f"source{s_idx}"] = info
        if info.get("status") == "loaded":
            print(f" Test Source {s_idx} ({fname}):")
            print(f"   * Total Records     : {info['total_rows']:,}")
            print(f"   * Countries Observed: {info['country_distribution']}")
        else:
            print(f" Test Source {s_idx} ({fname}): NOT FOUND at {fpath}")

    print("\n" + "=" * 75)
    return report


if __name__ == "__main__":
    # Execute full audit on default dataset directory
    run_full_data_audit("dataset")
