#!/usr/bin/env python3
"""
scripts/benchmark_large_sample.py — Large-Scale Performance & Profiling Test

Implements Requirement 8:
  - Runs 10,000 Source 1 records in 10 chunks of 1,000 records
  - Against 200,000 target entities from real test set
  - Measures:
      * retrieval time
      * feature extraction time
      * model inference time
      * output writing time
      * process RAM per chunk
      * candidates/query mean, p95, max
      * rows/sec and candidates/sec throughput
  - Verifies RAM stability across chunks (no memory leakage)
"""

import os
import sys
import time
import json
import collections
import polars as pl
import numpy as np
from typing import Dict, List, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import NormalizedRecord
from src.blocking import CompactTargetPool, MultiStageBlocker
from src.features import FeatureExtractor
from src.matching import EntityMatchingModel


def get_memory_mb() -> float:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return round(p.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        # Windows ctypes fallback
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
            return round(counters.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            return 0.0


def main():
    print("=" * 80)
    print(" REQUIREMENT 8: FULL-SCALE PERFORMANCE TEST (10,000 S1 QUERIES)")
    print("=" * 80)
    t0 = time.time()

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_dir = os.path.join(PROJECT_ROOT, "output", "perf_test")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Model Checkpoint
    model_path = os.path.join(PROJECT_ROOT, "models", "matching_model.pkl")
    print(f"\n[1/4] Loading Model from: {model_path}")
    model = EntityMatchingModel.load(model_path)
    print(f"  Model loaded. Decision Threshold: {model.optimal_threshold:.2f}")

    # 2. Build 200,000 Target Pool from Test Files
    print("\n[2/4] Building 200,000 Target Pool from Test Set...")
    t_pool_start = time.time()
    pool = CompactTargetPool()
    for s_idx in [2, 3]:
        tsv = f"dataset/test/test_source{s_idx}.tsv"
        df = pl.read_csv(tsv, separator="\t", infer_schema_length=0, n_rows=100_000)
        for r in df.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])

    pool_build_time = round(time.time() - t_pool_start, 2)
    ram_after_pool = get_memory_mb()
    print(f"  Indexed {len(pool):,} target records in {pool_build_time}s | RAM: {ram_after_pool} MB")

    t_idx_start = time.time()
    blocker = MultiStageBlocker(pool, max_candidates=40, enable_adaptive_budget=True)
    index_build_time = round(time.time() - t_idx_start, 2)
    ram_after_idx = get_memory_mb()
    print(f"  Constructed 9 inverted indexes in {index_build_time}s | RAM: {ram_after_idx} MB")

    # 3. Stream 10,000 S1 queries in 10 chunks of 1,000
    print("\n[3/4] Streaming 10,000 Source 1 Queries in 10 Chunks of 1,000...")
    s1_tsv = "dataset/test/test_source1.tsv"
    s1_df = pl.read_csv(s1_tsv, separator="\t", infer_schema_length=0, n_rows=10_000)

    chunk_size = 1000
    total_records = len(s1_df)

    chunk_metrics = []
    all_candidate_counts = []
    tot_candidates = 0
    tot_matches = 0

    t_retrieval_total = 0.0
    t_features_total = 0.0
    t_inference_total = 0.0
    t_writing_total = 0.0

    match_file = os.path.join(out_dir, "matching_results.tsv")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")

    with open(match_file, "w", encoding="utf-8") as fm, open(cand_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    chunk_idx = 0
    for start_idx in range(0, total_records, chunk_size):
        chunk_idx += 1
        t_chunk_start = time.time()
        c_df = s1_df.slice(start_idx, chunk_size)
        recs = [NormalizedRecord(r[0], r[1], r[2], r[3]) for r in c_df.iter_rows()]

        # Step A: Retrieval
        t_r0 = time.time()
        cands_map = {}
        cands_meta = {}
        for r in recs:
            cands, meta = blocker.generate_candidates_for_record(r)
            cands_map[r.entity_id] = cands
            cands_meta[r.entity_id] = meta
            all_candidate_counts.append(len(cands))
        t_retrieval = time.time() - t_r0
        t_retrieval_total += t_retrieval

        # Step B: Feature Extraction
        t_f0 = time.time()
        feat_rows = []
        pair_idx = []
        for r in recs:
            cands = cands_map[r.entity_id]
            meta = cands_meta[r.entity_id]
            for rk, tid in enumerate(cands):
                cm = meta.get(tid, {})
                feat = FeatureExtractor.extract_pairwise_features(
                    s1=r,
                    target_raw_name=cm.get("target_raw_name", ""),
                    target_raw_address=cm.get("target_raw_address", ""),
                    target_country=cm.get("target_country", ""),
                    candidate_meta=cm,
                    candidate_rank=rk,
                    use_extended_features=True
                )
                feat_rows.append(feat)
                pair_idx.append((r.entity_id, tid))
        t_features = time.time() - t_f0
        t_features_total += t_features

        num_cands = len(feat_rows)
        tot_candidates += num_cands

        # Step C: Model Inference
        t_m0 = time.time()
        if feat_rows:
            X = np.array(feat_rows, dtype=np.float32)
            probs = model.predict_proba(X)
        else:
            probs = np.array([])

        scores_by_s1 = collections.defaultdict(list)
        for (sid, tid), p in zip(pair_idx, probs):
            scores_by_s1[sid].append((tid, float(p)))
        for r in recs:
            if r.entity_id not in scores_by_s1:
                scores_by_s1[r.entity_id] = []

        decisions = model.make_entity_decisions(
            scores_by_s1,
            decision_threshold=model.optimal_threshold,
            margin_threshold=0.20
        )
        t_inference = time.time() - t_m0
        t_inference_total += t_inference

        # Step D: Output Writing
        t_w0 = time.time()
        chunk_matches = 0
        with open(match_file, "a", encoding="utf-8") as fm, open(cand_file, "a", encoding="utf-8") as fc:
            for r in recs:
                eid = r.entity_id
                acc = decisions[eid]["accepted_matches"]
                fm.write(f"{eid}\t{','.join(acc)}\n")
                chunk_matches += len(acc)
                c_list = cands_map.get(eid, [])
                fc.write(f"{eid}\t{','.join(c_list)}\n")
        t_writing = time.time() - t_w0
        t_writing_total += t_writing
        tot_matches += chunk_matches

        chunk_time = time.time() - t_chunk_start
        ram_now = get_memory_mb()

        c_info = {
            "chunk": chunk_idx,
            "queries": len(recs),
            "candidates": num_cands,
            "matches": chunk_matches,
            "chunk_sec": round(chunk_time, 2),
            "retrieval_sec": round(t_retrieval, 2),
            "feature_sec": round(t_features, 2),
            "inference_sec": round(t_inference, 2),
            "writing_sec": round(t_writing, 2),
            "throughput_rows_sec": round(len(recs) / chunk_time, 1),
            "candidates_sec": round(num_cands / max(t_features + t_inference, 0.001), 1),
            "ram_mb": ram_now
        }
        chunk_metrics.append(c_info)

        print(f"  Chunk {chunk_idx:2d}/10 | Rows: {chunk_idx*chunk_size:6,}/10,000 | "
              f"Cands: {num_cands:5,} | {c_info['throughput_rows_sec']:5.1f} rows/s | "
              f"Feat: {t_features:4.2f}s | Infer: {t_inference:4.2f}s | "
              f"RAM: {ram_now:6.1f} MB")

    total_infer_time = round(time.time() - t_pool_start, 2)
    c_arr = np.array(all_candidate_counts) if all_candidate_counts else np.array([0])

    # Summary
    summary = {
        "total_queries_processed": total_records,
        "total_targets_indexed": len(pool),
        "total_candidates_evaluated": tot_candidates,
        "total_matches_predicted": tot_matches,
        "overall_time_sec": total_infer_time,
        "overall_rows_per_sec": round(total_records / total_infer_time, 1),
        "mean_candidates_per_query": round(float(np.mean(c_arr)), 2),
        "median_candidates_per_query": float(np.median(c_arr)),
        "p90_candidates_per_query": float(np.percentile(c_arr, 90)),
        "p95_candidates_per_query": float(np.percentile(c_arr, 95)),
        "p99_candidates_per_query": float(np.percentile(c_arr, 99)),
        "max_candidates_per_query": int(np.max(c_arr)),
        "time_breakdown_sec": {
            "target_pool_build": pool_build_time,
            "inverted_index_build": index_build_time,
            "candidate_retrieval_total": round(t_retrieval_total, 2),
            "feature_extraction_total": round(t_features_total, 2),
            "model_inference_total": round(t_inference_total, 2),
            "output_writing_total": round(t_writing_total, 2)
        },
        "time_breakdown_pct": {
            "retrieval": round(100.0 * t_retrieval_total / total_infer_time, 1),
            "feature_extraction": round(100.0 * t_features_total / total_infer_time, 1),
            "model_inference": round(100.0 * t_inference_total / total_infer_time, 1),
            "output_writing": round(100.0 * t_writing_total / total_infer_time, 1)
        },
        "ram_tracking_mb": [c["ram_mb"] for c in chunk_metrics],
        "ram_start_mb": chunk_metrics[0]["ram_mb"],
        "ram_end_mb": chunk_metrics[-1]["ram_mb"],
        "ram_growth_mb": round(chunk_metrics[-1]["ram_mb"] - chunk_metrics[0]["ram_mb"], 2),
        "chunks": chunk_metrics
    }

    print("\n" + "=" * 80)
    print(" PERFORMANCE TEST SUMMARY:")
    print("=" * 80)
    print(f"  * Total Queries Processed       : {total_records:,}")
    print(f"  * Total Candidates Evaluated     : {tot_candidates:,}")
    print(f"  * Mean Candidates / Query        : {summary['mean_candidates_per_query']}")
    print(f"  * P95 Candidates / Query         : {summary['p95_candidates_per_query']}")
    print(f"  * Overall Throughput             : {summary['overall_rows_per_sec']} rows/sec")
    print(f"  * Retrieval Time Total           : {summary['time_breakdown_sec']['candidate_retrieval_total']}s ({summary['time_breakdown_pct']['retrieval']}%)")
    print(f"  * Feature Extraction Time Total  : {summary['time_breakdown_sec']['feature_extraction_total']}s ({summary['time_breakdown_pct']['feature_extraction']}%)")
    print(f"  * Model Inference Time Total     : {summary['time_breakdown_sec']['model_inference_total']}s ({summary['time_breakdown_pct']['model_inference']}%)")
    print(f"  * Output Writing Time Total      : {summary['time_breakdown_sec']['output_writing_total']}s ({summary['time_breakdown_pct']['output_writing']}%)")
    print(f"  * Initial Chunk RAM              : {summary['ram_start_mb']} MB")
    print(f"  * Final Chunk RAM                : {summary['ram_end_mb']} MB (Growth: {summary['ram_growth_mb']} MB)")
    print("=" * 80)

    perf_report_path = os.path.join(reports_dir, "large_sample_performance_test.json")
    with open(perf_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved performance test report to: {perf_report_path}")


if __name__ == "__main__":
    main()
