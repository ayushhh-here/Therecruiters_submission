#!/usr/bin/env python3
"""
scripts/infer.py — Memory-Safe Streaming Inference Pipeline

Implements Phase 5, 10, 13, 15:
  - Memory-safe chunked batching (Source 1 processed in configurable chunks)
  - Prebuilt compact target index over Source 2 & 3
  - Batch feature extraction and calibrated model scoring
  - Entity-level decision logic (margin & singleton thresholds)
  - Incremental streaming output writing to output/submission/
  - Real-time logging: chunk number, rows processed, candidates/sec, RAM usage, ETA
  - Automatic submission validation via utils/validate_submission.py
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional

# Ensure UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import polars as pl
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import NormalizedRecord
from src.blocking import CompactTargetPool, MultiStageBlocker
from src.features import FeatureExtractor
from src.matching import EntityMatchingModel


def get_process_ram_mb() -> float:
    """Returns current process RAM usage in MB using psutil or Windows ctypes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        try:
            import ctypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', ctypes.c_ulong),
                    ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                ctypes.sizeof(counters)
            )
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
        except Exception:
            return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Data Entity Resolution Inference Pipeline")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Directory containing test_source{1,2,3}.tsv")
    parser.add_argument("--model-path", type=str, default="models/matching_model.pkl", help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/submission", help="Directory for output TSVs")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of S1 records per chunk")
    parser.add_argument("--max-candidates", type=int, default=40, help="Maximum candidates per query entity")
    parser.add_argument("--max-s1-records", type=int, default=None, help="Optional limit on S1 records for benchmarking")
    parser.add_argument("--max-target-records", type=int, default=None, help="Optional limit on target records for benchmarking")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override of decision threshold")
    parser.add_argument("--no-validate", action="store_true", help="Skip submission validation check")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress in output TSVs")
    return parser.parse_args()


def run_inference(args):
    print("=" * 78)
    print("  AMAZON ML CHALLENGE — STREAMING INFERENCE PIPELINE (Phase 13)")
    print("=" * 78)
    overall_t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    matching_tsv_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    # 1. Load Model Checkpoint
    print(f"\n[1/5] Loading Model Checkpoint from: {args.model_path}")
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model file not found at: {args.model_path}")
        sys.exit(1)
    model = EntityMatchingModel.load(args.model_path)
    threshold = args.threshold if args.threshold is not None else model.optimal_threshold
    print(f"  ✓ Loaded {model.model_type.upper()} model.")
    print(f"  ✓ Decision Threshold: {threshold:.2f} | Extended Features: {model.use_extended_features}")

    # 2. Build Compact Target Pool
    print("\n[2/5] Building Compact Target Pool from Source 2 & Source 3...")
    t_pool_start = time.time()
    pool = CompactTargetPool()

    s2_file = os.path.join(args.test_dir, "test_source2.tsv")
    s3_file = os.path.join(args.test_dir, "test_source3.tsv")

    for fpath, label in [(s2_file, "Source 2"), (s3_file, "Source 3")]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] Target file missing: {fpath}")
            sys.exit(1)
        print(f"  Scanning {label} ({fpath})...")
        df_target = pl.read_csv(
            fpath,
            separator="\t",
            infer_schema_length=0,
            truncate_ragged_lines=True,
            n_rows=args.max_target_records
        )
        for r in df_target.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        print(f"    Loaded {len(df_target):,} records from {label}.")

    pool_elapsed = round(time.time() - t_pool_start, 2)
    print(f"  ✓ Total Target Entities Indexed: {len(pool):,} in {pool_elapsed}s.")
    print(f"  ✓ RAM after indexing: {get_process_ram_mb()} MB.")

    # Build multi-stage inverted indexes
    t_idx_start = time.time()
    blocker = MultiStageBlocker(pool, max_candidates=args.max_candidates, enable_adaptive_budget=True)
    idx_elapsed = round(time.time() - t_idx_start, 2)
    print(f"  ✓ Inverted indexes constructed in {idx_elapsed}s.")

    # 3. Stream Source 1 in Chunks
    print(f"\n[3/5] Processing Source 1 in Chunks (chunk_size={args.chunk_size:,})...")
    s1_file = os.path.join(args.test_dir, "test_source1.tsv")
    if not os.path.isfile(s1_file):
        print(f"[ERROR] Test Source 1 file missing: {s1_file}")
        sys.exit(1)

    # Read Source 1 in batches using Polars
    total_s1_processed = 0
    total_candidates_generated = 0
    total_matches_predicted = 0
    chunk_index = 0

    # Scan total S1 count
    df_s1_full = pl.read_csv(
        s1_file,
        separator="\t",
        infer_schema_length=0,
        truncate_ragged_lines=True,
        n_rows=args.max_s1_records
    )
    total_s1_records = len(df_s1_full)
    print(f"  Total Source 1 Records to Process: {total_s1_records:,}")

    start_offset = 0
    if args.resume and os.path.isfile(matching_tsv_path) and os.path.isfile(candidate_tsv_path):
        with open(matching_tsv_path, "r", encoding="utf-8") as f_chk:
            processed_lines = sum(1 for _ in f_chk) - 1  # exclude header
        if processed_lines > 0:
            start_offset = min(processed_lines, total_s1_records)
            total_s1_processed = start_offset
            print(f"  [RESUME] Detected existing output with {start_offset:,} rows processed.")
            print(f"  [RESUME] Resuming from row index {start_offset:,}...")
    
    if start_offset == 0:
        # Initialize output TSVs with exact required headers
        with open(matching_tsv_path, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
        with open(candidate_tsv_path, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    t_inference_start = time.time()

    for start_idx in range(start_offset, total_s1_records, args.chunk_size):
        chunk_index += 1
        t_chunk_start = time.time()
        chunk_df = df_s1_full.slice(start_idx, args.chunk_size)
        n_chunk = len(chunk_df)

        # Normalize S1 records
        s1_records = [
            NormalizedRecord(r[0], r[1], r[2], r[3])
            for r in chunk_df.iter_rows()
        ]

        # Candidate Retrieval & Feature Extraction for Chunk
        feature_rows = []
        pair_index = []
        chunk_cands_map = {}

        for s1_rec in s1_records:
            cands, meta = blocker.generate_candidates_for_record(s1_rec)
            chunk_cands_map[s1_rec.entity_id] = cands

            for rank_idx, tid in enumerate(cands):
                c_meta = meta.get(tid, {})
                feat = FeatureExtractor.extract_pairwise_features(
                    s1=s1_rec,
                    target_raw_name=c_meta.get("target_raw_name", ""),
                    target_raw_address=c_meta.get("target_raw_address", ""),
                    target_country=c_meta.get("target_country", ""),
                    candidate_meta=c_meta,
                    candidate_rank=rank_idx,
                    use_extended_features=model.use_extended_features
                )
                feature_rows.append(feat)
                pair_index.append((s1_rec.entity_id, tid))

        chunk_cands_count = len(feature_rows)
        total_candidates_generated += chunk_cands_count

        # Model Scoring
        if feature_rows:
            X_chunk = np.array(feature_rows, dtype=np.float32)
            probs = model.predict_proba(X_chunk)
        else:
            probs = np.array([])

        # Group Scores per S1 Entity
        candidate_scores_per_s1 = {rec.entity_id: [] for rec in s1_records}
        for (s1_id, tid), prob in zip(pair_index, probs):
            candidate_scores_per_s1[s1_id].append((tid, float(prob)))

        # Entity-Level Decision Logic (Phase 10)
        decisions = model.make_entity_decisions(
            candidate_scores_per_s1=candidate_scores_per_s1,
            decision_threshold=threshold,
            margin_threshold=model.margin_threshold
        )

        # Streaming Output Append
        with open(matching_tsv_path, "a", encoding="utf-8") as f_match, \
             open(candidate_tsv_path, "a", encoding="utf-8") as f_cand:
            for s1_rec in s1_records:
                eid = s1_rec.entity_id
                # Matches
                accepted = decisions[eid]["accepted_matches"]
                f_match.write(f"{eid}\t{','.join(accepted)}\n")
                total_matches_predicted += len(accepted)
                # Candidates
                cands_list = chunk_cands_map.get(eid, [])
                f_cand.write(f"{eid}\t{','.join(cands_list)}\n")

        total_s1_processed += n_chunk
        chunk_elapsed = time.time() - t_chunk_start
        throughput_rows_sec = round(n_chunk / max(chunk_elapsed, 0.001), 1)
        total_elapsed = time.time() - t_inference_start
        pct_done = (total_s1_processed / total_s1_records) * 100.0
        est_total = (total_elapsed / max(total_s1_processed, 1)) * total_s1_records
        eta_sec = max(0, int(est_total - total_elapsed))

        print(f"  Chunk {chunk_index:3d} | Rows: {total_s1_processed:7,}/{total_s1_records:,} ({pct_done:5.1f}%) | "
              f"Cands: {chunk_cands_count:5,} | {throughput_rows_sec:6.1f} rows/s | "
              f"RAM: {get_process_ram_mb():5.1f} MB | ETA: {eta_sec:4d}s")

    overall_time = round(time.time() - overall_t0, 2)
    print("\n" + "=" * 78)
    print("  INFERENCE COMPLETE:")
    print(f"  * Total S1 Records Processed   : {total_s1_processed:,}")
    print(f"  * Total Candidates Generated   : {total_candidates_generated:,}")
    print(f"  * Total Matches Predicted       : {total_matches_predicted:,}")
    print(f"  * Overall Throughput            : {round(total_s1_processed / max(overall_time, 1), 1)} rows/sec")
    print(f"  * Overall Runtime               : {overall_time}s")
    print(f"  * Candidate TSV Exported        : {candidate_tsv_path}")
    print(f"  * Matching TSV Exported         : {matching_tsv_path}")
    print("=" * 78)

    # 4. Submission Validation (Phase 13)
    if not args.no_validate:
        print("\n[4/5] Executing Official Submission Validator...")
        validator_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
        cmd = [
            sys.executable,
            validator_script,
            "--matching", matching_tsv_path,
            "--candidate", candidate_tsv_path,
            "--test-dir", args.test_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(result.stdout)
        if result.returncode != 0:
            print("[ERROR] Submission validation failed! Error output:")
            print(result.stderr)
            sys.exit(1)
        else:
            print("  ✓ Submission validator returned PASS (Exit Code 0)!")

    print("\n[5/5] Pipeline successfully finished and verified!")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
