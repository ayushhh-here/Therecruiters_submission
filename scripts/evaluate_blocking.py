#!/usr/bin/env python3
"""
scripts/evaluate_blocking.py — Measured Blocking Evaluation Script

Measures candidate generation recall and distribution on real challenge data:
  - Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100, Recall@200
  - Candidate count percentiles (p50, p90, p95, p99, max)
  - Zero-candidate query percentage
  - Mean rank of true matches
  - Adaptive vs Fixed candidate budget comparison

Outputs:
  - reports/blocking_evaluation.json
"""

import os
import sys
import json
import time
import polars as pl
import numpy as np

# Ensure UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import NormalizedRecord
from src.blocking import CompactTargetPool, MultiStageBlocker, evaluate_blocking_recall_curve


def main():
    print("=" * 75)
    print(" Amazon ML Challenge — Measured Blocking System Evaluation (Phase 6)")
    print("=" * 75)
    t0 = time.time()

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Ground Truth for Evaluation Entities
    print("\n[Step 1/4] Loading Validation Split & Ground Truth...")
    val_split_path = os.path.join(PROJECT_ROOT, "splits", "val_entity_ids_sample10k.tsv")
    if not os.path.isfile(val_split_path):
        val_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=2000, infer_schema_length=0)
        eval_s1_ids = set(val_df["entity_id"].to_list())
    else:
        val_df = pl.read_csv(val_split_path, separator="\t", n_rows=2000, infer_schema_length=0)
        eval_s1_ids = set(val_df["entity_id"].to_list())

    # Load Ground Truth
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
    ground_truth = {}
    needed_target_ids = set()

    for r in gt_df.iter_rows():
        s1_id = r[0]
        if s1_id in eval_s1_ids:
            raw_matches = str(r[1] or "").strip()
            if raw_matches:
                m_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
                ground_truth[s1_id] = m_list
                needed_target_ids.update(m_list)
            else:
                ground_truth[s1_id] = []

    print(f"  Loaded ground truth for {len(ground_truth):,} validation entities.")
    print(f"  Identified {len(needed_target_ids):,} true target match IDs.")

    # 2. Build Target Pool (All True Match Targets + 200,000 Distractor Pool)
    print("\n[Step 2/4] Building Compact Target Pool & Inverted Indexes...")
    pool = CompactTargetPool()
    t_index_start = time.time()

    # Load targets that match needed_target_ids, plus random distractors
    for s_idx in [2, 3]:
        tsv_path = f"dataset/train/train_source{s_idx}.tsv"
        print(f"  Scanning {tsv_path} for true targets & distractors...")
        df_scan = pl.read_csv(tsv_path, separator="\t", infer_schema_length=0)
        
        # True targets
        matched_targets = df_scan.filter(pl.col("entity_id").is_in(needed_target_ids))
        for r in matched_targets.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
            
        # Distractors (first 100,000 records)
        distractors = df_scan.head(100_000)
        for r in distractors.iter_rows():
            if r[0] not in needed_target_ids:
                pool.add(r[0], r[1], r[2], r[3])

    print(f"  Target pool populated with {len(pool):,} targets ({len(needed_target_ids):,} true matches + distractors) in {round(time.time() - t_index_start, 2)}s.")

    # Initialize Blocker
    blocker = MultiStageBlocker(pool, max_candidates=200, enable_adaptive_budget=False)
    print("  Multi-stage inverted indexes constructed.")

    # 3. Load Normalized S1 Validation Queries
    print("\n[Step 3/4] Preparing S1 Validation Query Records...")
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    val_records = []
    for r in s1_all.iter_rows():
        if r[0] in eval_s1_ids:
            val_records.append(NormalizedRecord(r[0], r[1], r[2], r[3]))

    print(f"  Prepared {len(val_records):,} query records for recall measurement.")

    # 4. Measure Recall@K and Percentiles
    print("\n[Step 4/4] Computing Measured Recall Curve across K ∈ [1, 5, 10, 20, 50, 100, 200]...")
    metrics = evaluate_blocking_recall_curve(
        blocker=blocker,
        val_records=val_records,
        ground_truth=ground_truth,
        recall_k_thresholds=[1, 5, 10, 20, 50, 100, 200]
    )

    elapsed = round(time.time() - t0, 2)
    metrics["evaluation_time_sec"] = elapsed

    # Save to JSON report
    report_path = os.path.join(reports_dir, "blocking_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved blocking evaluation report to: {report_path}")
    print("\n" + "=" * 75)
    print(" MEASURED BLOCKING RECALL RESULTS:")
    print("=" * 75)
    for k_name, rec in metrics["recall_at_k"].items():
        print(f"  * {k_name:12s}: {rec * 100:>6.2f}%")
    print("-" * 75)
    print(" CANDIDATE DISTRIBUTION:")
    pcts = metrics["candidate_count_percentiles"]
    print(f"  * Median Candidates/Query : {pcts['p50_median']:.1f}")
    print(f"  * Mean Candidates/Query   : {pcts['mean']:.1f}")
    print(f"  * p90 Candidates/Query    : {pcts['p90']:.1f}")
    print(f"  * p95 Candidates/Query    : {pcts['p95']:.1f}")
    print(f"  * p99 Candidates/Query    : {pcts['p99']:.1f}")
    print(f"  * Max Candidates/Query    : {pcts['max']}")
    print(f"  * Zero-Candidate Queries  : {metrics['zero_candidate_query_pct']:.2f}%")
    print(f"  * Mean True Match Rank    : {metrics['mean_true_match_rank']:.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
