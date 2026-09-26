#!/usr/bin/env python3
"""
scripts/pre_submission_audit.py — Final Pre-Submission Audit

Implements Requirements 2, 3, 4 of the final audit:
  - Req 2: Full blocking statistics (Recall@K + candidate count percentiles)
  - Req 3: Street component index audit (key distribution, collisions, frequency)
  - Req 4: Categorized failure analysis for the 8.5% retrieval gap (with NEW blocker)
"""

import os
import sys
import json
import time
import csv
import collections
import re
import polars as pl
import numpy as np
from typing import Dict, List, Set, Any

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import (
    NormalizedRecord,
    normalize_unicode_and_case,
    remove_punctuation_unicode,
    detect_primary_script,
    generate_char_ngrams
)
from src.blocking import (
    CompactTargetPool, MultiStageBlocker, STOP_TOKENS,
    extract_street_components
)
from src.features import fast_levenshtein_similarity, jaccard_similarity


def main():
    print("=" * 80)
    print(" FINAL PRE-SUBMISSION AUDIT (Requirements 2, 3, 4)")
    print("=" * 80)
    t0 = time.time()

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Validation Split & Ground Truth
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/6] Loading Validation Split & Ground Truth...")
    val_split_path = os.path.join(PROJECT_ROOT, "splits", "val_entity_ids_sample10k.tsv")
    val_df = pl.read_csv(val_split_path, separator="\t", n_rows=2000, infer_schema_length=0)
    eval_s1_ids = set(val_df["entity_id"].to_list())

    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
    ground_truth = {}
    needed_target_ids = set()

    for r in gt_df.iter_rows():
        s1_id = r[0]
        if s1_id in eval_s1_ids:
            raw = str(r[1] or "").strip()
            if raw:
                matches = [m.strip() for m in raw.split(",") if m.strip()]
                ground_truth[s1_id] = matches
                needed_target_ids.update(matches)
            else:
                ground_truth[s1_id] = []

    total_pairs = sum(len(m) for m in ground_truth.values())
    print(f"  Queries: {len(eval_s1_ids):,}, True positive pairs: {total_pairs:,}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Build Target Pool
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/6] Building Target Pool (True Targets + 200K Distractors)...")
    pool = CompactTargetPool()
    target_raw_data = {}

    for s_idx in [2, 3]:
        tsv = f"dataset/train/train_source{s_idx}.tsv"
        df = pl.read_csv(tsv, separator="\t", infer_schema_length=0)
        matched = df.filter(pl.col("entity_id").is_in(needed_target_ids))
        for r in matched.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
            target_raw_data[r[0]] = (r[1], r[2], r[3])
        for r in df.head(100_000).iter_rows():
            if r[0] not in needed_target_ids:
                pool.add(r[0], r[1], r[2], r[3])
                target_raw_data[r[0]] = (r[1], r[2], r[3])

    print(f"  Indexed {len(pool):,} targets in {round(time.time() - t0, 2)}s.")
    blocker = MultiStageBlocker(pool, max_candidates=200, enable_adaptive_budget=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: REQ 3 — Street Component Index Audit
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/6] REQ 3: Auditing Street Component Index...")
    street_idx = blocker.street_component_index.index
    total_keys = len(street_idx)
    posting_lengths = [len(v) for v in street_idx.values()]
    pl_arr = np.array(posting_lengths) if posting_lengths else np.array([0])

    singleton_keys = sum(1 for l in posting_lengths if l == 1)
    heavy_keys = sum(1 for l in posting_lengths if l > 50)
    very_heavy_keys = sum(1 for l in posting_lengths if l > 100)

    # Top-20 most populated keys
    top_keys = sorted(street_idx.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    street_audit = {
        "total_unique_street_keys": total_keys,
        "total_postings": int(np.sum(pl_arr)),
        "avg_entities_per_key": round(float(np.mean(pl_arr)), 2),
        "median_entities_per_key": float(np.median(pl_arr)),
        "p90_entities_per_key": float(np.percentile(pl_arr, 90)),
        "p95_entities_per_key": float(np.percentile(pl_arr, 95)),
        "p99_entities_per_key": float(np.percentile(pl_arr, 99)),
        "max_entities_per_key": int(np.max(pl_arr)),
        "singleton_keys_count": singleton_keys,
        "singleton_keys_pct": round(100.0 * singleton_keys / max(total_keys, 1), 2),
        "keys_with_50_plus_entities": heavy_keys,
        "keys_with_100_plus_entities": very_heavy_keys,
        "top_20_collision_heavy_keys": [
            {"key": k, "entity_count": len(v)} for k, v in top_keys
        ]
    }

    print(f"  Total unique street keys        : {total_keys:,}")
    print(f"  Avg entities per key             : {street_audit['avg_entities_per_key']}")
    print(f"  Median entities per key          : {street_audit['median_entities_per_key']}")
    print(f"  P95 entities per key             : {street_audit['p95_entities_per_key']}")
    print(f"  P99 entities per key             : {street_audit['p99_entities_per_key']}")
    print(f"  Max entities per key             : {street_audit['max_entities_per_key']}")
    print(f"  Singleton keys (occur once)      : {singleton_keys:,} ({street_audit['singleton_keys_pct']}%)")
    print(f"  Keys with 50+ entities           : {heavy_keys}")
    print(f"  Keys with 100+ entities          : {very_heavy_keys}")
    print(f"  Top-5 collision-heavy keys:")
    for item in street_audit["top_20_collision_heavy_keys"][:5]:
        print(f"    '{item['key']}' → {item['entity_count']} entities")

    street_audit_path = os.path.join(reports_dir, "street_component_index_audit.json")
    with open(street_audit_path, "w", encoding="utf-8") as f:
        json.dump(street_audit, f, indent=2)
    print(f"  ✓ Saved to: {street_audit_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: REQ 2 — Full Blocking Recall & Candidate Statistics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/6] REQ 2: Computing Full Blocking Recall & Candidate Statistics...")
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    val_records = {}
    for r in s1_all.iter_rows():
        if r[0] in eval_s1_ids:
            val_records[r[0]] = NormalizedRecord(r[0], r[1], r[2], r[3])

    ks = [1, 5, 10, 20, 50, 100, 200]
    captured_at_k = {k: 0 for k in ks}
    candidate_counts = []
    true_match_ranks = []
    zero_cand = 0

    for s1_id, true_targets in ground_truth.items():
        if not true_targets:
            continue
        rec = val_records[s1_id]
        cands, _ = blocker.generate_candidates_for_record(rec, custom_max_candidates=200)
        candidate_counts.append(len(cands))
        if len(cands) == 0:
            zero_cand += 1
            continue
        cand_sets = {k: set(cands[:k]) for k in ks}
        for tid in true_targets:
            for k in ks:
                if tid in cand_sets[k]:
                    captured_at_k[k] += 1
            if tid in cands:
                true_match_ranks.append(cands.index(tid) + 1)

    c_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    blocking_stats = {
        "total_queries": len(candidate_counts),
        "total_true_matches": total_pairs,
        "recall_at_k": {f"Recall@{k}": round(captured_at_k[k] / total_pairs, 4) for k in ks},
        "zero_candidate_pct": round(100.0 * zero_cand / max(len(candidate_counts), 1), 2),
        "mean_candidates_per_query": round(float(np.mean(c_arr)), 2),
        "median_candidates_per_query": float(np.median(c_arr)),
        "p90_candidates_per_query": float(np.percentile(c_arr, 90)),
        "p95_candidates_per_query": float(np.percentile(c_arr, 95)),
        "p99_candidates_per_query": float(np.percentile(c_arr, 99)),
        "max_candidates_per_query": int(np.max(c_arr)),
    }

    print(f"\n  BLOCKING RECALL (New Blocker, K=200):")
    for kn, rv in blocking_stats["recall_at_k"].items():
        print(f"    {kn:12s}: {rv * 100:>6.2f}%")
    print(f"  Zero-candidate queries : {blocking_stats['zero_candidate_pct']}%")
    print(f"  Mean candidates/query  : {blocking_stats['mean_candidates_per_query']}")
    print(f"  Median candidates/query: {blocking_stats['median_candidates_per_query']}")
    print(f"  P90 candidates/query   : {blocking_stats['p90_candidates_per_query']}")
    print(f"  P95 candidates/query   : {blocking_stats['p95_candidates_per_query']}")
    print(f"  P99 candidates/query   : {blocking_stats['p99_candidates_per_query']}")
    print(f"  Max candidates/query   : {blocking_stats['max_candidates_per_query']}")

    blocking_stats_path = os.path.join(reports_dir, "blocking_stats_final.json")
    with open(blocking_stats_path, "w", encoding="utf-8") as f:
        json.dump(blocking_stats, f, indent=2)
    print(f"  ✓ Saved to: {blocking_stats_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: REQ 4 — Failure Analysis for the ~8.5% Retrieval Gap (NEW Blocker)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/6] REQ 4: Analyzing Retrieval Failures with NEW Blocker (missed at R@200)...")
    target_id_to_idx = {pool.entity_ids[i]: i for i in range(len(pool))}
    failure_records = []
    failure_counts = collections.Counter()

    for s1_id, true_targets in ground_truth.items():
        if not true_targets:
            continue
        rec = val_records[s1_id]
        cands, _ = blocker.generate_candidates_for_record(rec, custom_max_candidates=200)
        cand_set = set(cands)

        for tid in true_targets:
            if tid in cand_set:
                continue  # retrieved — skip

            raw_tuple = target_raw_data.get(tid, ("", "", ""))
            t_name = str(raw_tuple[0] or "")
            t_addr = str(raw_tuple[1] or "")
            t_ctry = str(raw_tuple[2] or "")
            s_addr_str = str(rec.raw_address or "")

            t_norm_name = remove_punctuation_unicode(normalize_unicode_and_case(t_name))
            t_tokens = set(t_norm_name.split())
            t_script = detect_primary_script(t_name)
            t_street_keys = extract_street_components(t_addr)
            s_street_keys = extract_street_components(s_addr_str)

            # Categorize failure
            tidx = target_id_to_idx.get(tid, -1)
            country_compat = True
            if tidx >= 0:
                country_compat = blocker.is_country_compatible(rec.norm_country, tidx)

            if not country_compat:
                cat = "country_mismatch"
            elif not t_addr.strip() or not s_addr_str.strip():
                cat = "missing_address"
            elif s1_rec_script := (rec.primary_script if hasattr(rec, 'primary_script') else "UNKNOWN"):
                if s1_rec_script != t_script and s1_rec_script != "UNKNOWN" and t_script != "UNKNOWN":
                    # Check if address could bridge the gap
                    shared_street = set(s_street_keys) & set(t_street_keys)
                    if shared_street:
                        cat = "script_mismatch_but_shared_address"
                    else:
                        cat = "script_mismatch_no_shared_address"
                elif len(rec.name_tokens & t_tokens) == 0:
                    lev = fast_levenshtein_similarity(rec.norm_name, t_norm_name)
                    if lev >= 0.65:
                        cat = "fuzzy_spelling_variation"
                    elif len(rec.norm_name) <= 4 or len(t_norm_name) <= 4:
                        cat = "acronym_or_abbreviation"
                    else:
                        # Check if addresses match but names are completely different
                        s_addr_norm = remove_punctuation_unicode(normalize_unicode_and_case(rec.raw_address))
                        t_addr_norm = remove_punctuation_unicode(normalize_unicode_and_case(t_addr))
                        addr_lev = fast_levenshtein_similarity(s_addr_norm, t_addr_norm)
                        if addr_lev >= 0.60:
                            cat = "different_business_name_same_address"
                        else:
                            cat = "severe_name_and_address_divergence"
                else:
                    # Has shared tokens but still missed
                    shared = rec.name_tokens & t_tokens
                    all_stop = all(tok in STOP_TOKENS for tok in shared)
                    if all_stop:
                        cat = "shared_tokens_are_stop_words"
                    else:
                        # Check if all shared tokens were frequency-capped
                        capped = [tok for tok in shared if tok not in STOP_TOKENS and len(blocker.token_index.index.get(tok, [])) > blocker.token_index.max_token_frequency]
                        informative = [tok for tok in shared if tok not in STOP_TOKENS and len(tok) >= 3]
                        if len(capped) == len(informative) and informative:
                            cat = "all_informative_tokens_frequency_capped"
                        else:
                            cat = "low_retrieval_score_outranked"
            else:
                cat = "unknown"

            failure_counts[cat] += 1
            failure_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "category": cat,
                "s1_name": rec.raw_name,
                "t_name": t_name,
                "s1_addr": rec.raw_address,
                "t_addr": t_addr,
                "s1_script": getattr(rec, 'primary_script', 'UNKNOWN'),
                "t_script": t_script,
            })

    total_failures = sum(failure_counts.values())
    print(f"\n  Total unretrieved true matches (at R@200): {total_failures:,}")
    print(f"  Unretrieved as % of all positive pairs: {100.0 * total_failures / total_pairs:.2f}%")
    print(f"\n  FAILURE CATEGORIZATION:")
    for cat, count in failure_counts.most_common():
        pct_miss = 100.0 * count / max(total_failures, 1)
        pct_all = 100.0 * count / total_pairs
        print(f"    {cat:50s}: {count:>4,} ({pct_miss:>5.1f}% of misses | {pct_all:>5.2f}% of all)")

    # Save CSV
    fail_csv_path = os.path.join(reports_dir, "blocking_failure_analysis_new_blocker.csv")
    with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source1_id", "target_id", "category", "s1_name", "t_name",
            "s1_addr", "t_addr", "s1_script", "t_script"
        ])
        writer.writeheader()
        writer.writerows(failure_records)

    # Save summary JSON
    fail_json_path = os.path.join(reports_dir, "blocking_failure_summary_new_blocker.json")
    with open(fail_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_positive_pairs": total_pairs,
            "total_unretrieved_at_200": total_failures,
            "unretrieved_pct": round(100.0 * total_failures / total_pairs, 2),
            "failure_counts": dict(failure_counts.most_common()),
        }, f, indent=2)

    print(f"  ✓ Saved failure CSV to: {fail_csv_path}")
    print(f"  ✓ Saved failure summary to: {fail_json_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Summary
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = round(time.time() - t0, 2)
    print(f"\n{'=' * 80}")
    print(f" PRE-SUBMISSION AUDIT COMPLETE ({elapsed}s)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
