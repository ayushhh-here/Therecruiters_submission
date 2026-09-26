#!/usr/bin/env python3
"""
scripts/diagnose_blocking.py — True-Match Rank Distribution & Failure Analysis

Implements Phase 19 Requirements 1, 2, and 3:
  - Exact rank computation for every true positive pair (S1, T_true)
  - Detailed distribution: rank <= 1, 5, 10, 20, 50, 100, 200, >200, not retrieved
  - Percentiles: p50, p75, p90, p95, p99
  - Exhaustive categorization of why true matches are missed:
      * country mismatch
      * missing address
      * script mismatch (e.g. Latin vs Devanagari transliteration gap)
      * zero token overlap (severe lexical divergence / synonym)
      * token capped (all shared tokens exceeded max_token_frequency)
      * abbreviation / acronym mismatch
      * candidate cap (retrieved by rules but rank > 200)
      * address mismatch with low name similarity
  - Outputs:
      * reports/true_match_rank_distribution.csv
      * reports/blocking_failure_analysis.json
"""

import os
import sys
import json
import time
import csv
import collections
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

from src.normalization import (
    NormalizedRecord,
    normalize_unicode_and_case,
    remove_punctuation_unicode,
    detect_primary_script,
    generate_char_ngrams
)
from src.blocking import CompactTargetPool, MultiStageBlocker, STOP_TOKENS
from src.features import fast_levenshtein_similarity, jaccard_similarity


def main():
    print("=" * 78)
    print(" PHASE 19: TRUE-MATCH RANK DISTRIBUTION & FAILURE ANALYSIS")
    print("=" * 78)
    t0 = time.time()

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Validation Split & Ground Truth
    print("\n[1/5] Loading Validation Split & Ground Truth...")
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
    print(f"  Validation queries: {len(eval_s1_ids):,} ({len(ground_truth):,} evaluated)")
    print(f"  Total true target matches: {total_pairs:,} across {len(needed_target_ids):,} unique target IDs")

    # 2. Build Target Pool (True Targets + 200,000 Distractors)
    print("\n[2/5] Building Target Pool (True Targets + 200,000 Distractors)...")
    pool = CompactTargetPool()
    target_raw_data = {}  # tid -> (name, addr, ctry)

    for s_idx in [2, 3]:
        tsv_path = f"dataset/train/train_source{s_idx}.tsv"
        df_scan = pl.read_csv(tsv_path, separator="\t", infer_schema_length=0)
        
        # True targets
        matched_targets = df_scan.filter(pl.col("entity_id").is_in(needed_target_ids))
        for r in matched_targets.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
            target_raw_data[r[0]] = (r[1], r[2], r[3])
            
        # Distractors (first 100k)
        distractors = df_scan.head(100_000)
        for r in distractors.iter_rows():
            if r[0] not in needed_target_ids:
                pool.add(r[0], r[1], r[2], r[3])
                target_raw_data[r[0]] = (r[1], r[2], r[3])

    print(f"  Indexed {len(pool):,} targets into pool in {round(time.time() - t0, 2)}s.")
    blocker = MultiStageBlocker(pool, max_candidates=200, enable_adaptive_budget=False)

    # Map target ID to pool index for fast lookup
    target_id_to_idx = {pool.entity_ids[i]: i for i in range(len(pool))}

    # 3. Load Normalized S1 Records
    print("\n[3/5] Normalizing Query S1 Records...")
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    val_records = {}
    for r in s1_all.iter_rows():
        if r[0] in eval_s1_ids:
            val_records[r[0]] = NormalizedRecord(r[0], r[1], r[2], r[3])

    # 4. Rank Calculation & In-Depth Diagnostic for Every Positive Pair
    print("\n[4/5] Computing Rank Distribution & Diagnosing Missed Matches...")
    pair_rank_records = []
    ranks = []
    
    # Failure categorization counters
    failure_counts = collections.Counter()
    failure_samples = collections.defaultdict(list)

    # Retrieval stats
    unretrieved_count = 0
    past_200_count = 0

    for s1_id, true_targets in ground_truth.items():
        if not true_targets:
            continue
        s1_rec = val_records[s1_id]

        # Get full candidate ranking without capping (query internal candidate scores)
        candidate_scores: Dict[int, float] = collections.defaultdict(float)
        candidate_rules: Dict[int, set] = collections.defaultdict(set)

        # Rule 1: Exact Name
        if s1_rec.norm_name:
            for tidx in blocker.exact_name_index.query(s1_rec.norm_name):
                if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                    candidate_scores[tidx] += 12.0
                    candidate_rules[tidx].add("exact_name")

        # Rule 2: Sorted Name
        if s1_rec.sorted_name_key:
            for tidx in blocker.sorted_name_index.query(s1_rec.sorted_name_key):
                if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                    candidate_scores[tidx] += 9.0
                    candidate_rules[tidx].add("sorted_name")

        # Rule 3: Translit Name
        if s1_rec.translit_name and s1_rec.translit_name != s1_rec.norm_name:
            for tidx in blocker.translit_name_index.query(s1_rec.translit_name):
                if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                    candidate_scores[tidx] += 8.0
                    candidate_rules[tidx].add("translit_name")

        # Rule 4: Core Name
        if s1_rec.core_name and s1_rec.core_name != s1_rec.norm_name:
            for tidx in blocker.core_name_index.query(s1_rec.core_name):
                if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                    candidate_scores[tidx] += 7.0
                    candidate_rules[tidx].add("core_name")

        # Rule 5: Postal Code
        if s1_rec.postal_code:
            for tidx in blocker.postal_code_index.query(s1_rec.postal_code):
                if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                    shared = s1_rec.name_tokens & pool.name_tokens_list[tidx]
                    if shared:
                        candidate_scores[tidx] += 6.0 + len(shared)
                        candidate_rules[tidx].add("postal_code")

        # Rule 6: Informative Tokens
        tok_hits = collections.defaultdict(int)
        for tok in s1_rec.name_tokens:
            if len(tok) >= 3 and tok not in STOP_TOKENS:
                for tidx in blocker.token_index.query(tok):
                    if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                        tok_hits[tidx] += 1
        for tidx, count in tok_hits.items():
            candidate_scores[tidx] += count * 2.5
            candidate_rules[tidx].add("token_overlap")

        # Rule 7: Trigram Fallback (when candidates < 100)
        if len(candidate_scores) < 100 and s1_rec.name_trigrams:
            tri_hits = collections.defaultdict(int)
            for tri in s1_rec.name_trigrams:
                for tidx in blocker.trigram_index.query(tri):
                    if blocker.is_country_compatible(s1_rec.norm_country, tidx):
                        tri_hits[tidx] += 1
            for tidx, count in tri_hits.items():
                if count >= 3:
                    candidate_scores[tidx] += count * 0.25
                    candidate_rules[tidx].add("trigram_overlap")

        # Sort all scored candidates
        sorted_candidates = sorted(
            candidate_scores.keys(),
            key=lambda idx: (len(candidate_rules[idx]), candidate_scores[idx]),
            reverse=True
        )

        ranked_pool_ids = [pool.entity_ids[tidx] for tidx in sorted_candidates]
        pool_id_to_rank = {pid: r + 1 for r, pid in enumerate(ranked_pool_ids)}

        # Evaluate each true target
        for true_id in true_targets:
            rank = pool_id_to_rank.get(true_id, -1)
            t_raw_name, t_raw_addr, t_raw_ctry = target_raw_data.get(true_id, ("", "", ""))
            
            pair_rank_records.append({
                "source1_id": s1_id,
                "target_id": true_id,
                "rank": rank if rank > 0 else "NOT_RETRIEVED",
                "is_in_top_20": 1 if (0 < rank <= 20) else 0,
                "is_in_top_50": 1 if (0 < rank <= 50) else 0,
                "is_in_top_100": 1 if (0 < rank <= 100) else 0,
                "is_in_top_200": 1 if (0 < rank <= 200) else 0,
            })

            if rank > 0:
                ranks.append(rank)
            else:
                unretrieved_count += 1

            # ─── DIAGNOSTIC FAILURE CATEGORIZATION (for rank > 50 or rank == -1) ───
            if rank > 50 or rank == -1:
                t_norm_name = remove_punctuation_unicode(normalize_unicode_and_case(t_raw_name))
                t_tokens = set(t_norm_name.split())
                t_script = detect_primary_script(t_raw_name)
                
                # Check country compatibility
                if not blocker.is_country_compatible(s1_rec.norm_country, target_id_to_idx.get(true_id, 0)):
                    cat = "country_mismatch"
                # Check if true target was scored by rules but pushed past candidate budget (rank > 200)
                elif rank > 200:
                    cat = "candidate_cap_exceeded (rank > 200)"
                    past_200_count += 1
                elif 50 < rank <= 200:
                    cat = "candidate_cap_exceeded (50 < rank <= 200)"
                # Check script divergence
                elif s1_rec.primary_script != t_script and s1_rec.primary_script != "UNKNOWN" and t_script != "UNKNOWN":
                    cat = "script_mismatch"
                # Check missing address
                elif not t_raw_addr or not s1_rec.raw_address:
                    cat = "missing_address"
                # Check token overlap
                elif len(s1_rec.name_tokens & t_tokens) == 0:
                    # No shared word tokens
                    lev = fast_levenshtein_similarity(s1_rec.norm_name, t_norm_name)
                    tri_jacc = jaccard_similarity(s1_rec.name_trigrams, generate_char_ngrams(t_norm_name, 3))
                    if lev >= 0.70 or tri_jacc >= 0.40:
                        cat = "fuzzy_spelling_variation (no exact token)"
                    elif len(s1_rec.norm_name) <= 4 or len(t_norm_name) <= 4:
                        cat = "acronym_or_short_code"
                    else:
                        cat = "severe_name_divergence / synonym"
                # Shared token was stop token or capped frequency
                elif all(tok in STOP_TOKENS for tok in (s1_rec.name_tokens & t_tokens)):
                    cat = "shared_tokens_are_generic_stop_tokens"
                else:
                    # Token overlap exists, but why wasn't it retrieved?
                    # Check token frequency in blocker index
                    shared_toks = s1_rec.name_tokens & t_tokens
                    capped_toks = [tok for tok in shared_toks if len(blocker.token_index.index.get(tok, [])) > blocker.max_token_freq]
                    if len(capped_toks) == len(shared_toks):
                        cat = "token_frequency_capped (> 400 records)"
                    else:
                        cat = "low_retrieval_priority_score"

                failure_counts[cat] += 1
                if len(failure_samples[cat]) < 3:
                    failure_samples[cat].append({
                        "s1_id": s1_id,
                        "s1_name": s1_rec.raw_name,
                        "s1_addr": s1_rec.raw_address,
                        "target_id": true_id,
                        "t_name": t_raw_name,
                        "t_addr": t_raw_addr,
                        "rank": rank
                    })

    # 5. Output Reports
    print("\n[5/5] Saving Rank Distribution & Failure Analysis Reports...")
    rank_csv_path = os.path.join(reports_dir, "true_match_rank_distribution.csv")
    with open(rank_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source1_id", "target_id", "rank", "is_in_top_20", "is_in_top_50", "is_in_top_100", "is_in_top_200"])
        writer.writeheader()
        writer.writerows(pair_rank_records)

    # Compute percentiles on retrieved ranks
    r_arr = np.array(ranks) if ranks else np.array([0])
    p50 = float(np.percentile(r_arr, 50))
    p75 = float(np.percentile(r_arr, 75))
    p90 = float(np.percentile(r_arr, 90))
    p95 = float(np.percentile(r_arr, 95))
    p99 = float(np.percentile(r_arr, 99))

    # Bracket counts
    b_1 = sum(1 for r in ranks if r <= 1)
    b_5 = sum(1 for r in ranks if r <= 5)
    b_10 = sum(1 for r in ranks if r <= 10)
    b_20 = sum(1 for r in ranks if r <= 20)
    b_50 = sum(1 for r in ranks if r <= 50)
    b_100 = sum(1 for r in ranks if r <= 100)
    b_200 = sum(1 for r in ranks if r <= 200)
    b_gt_200 = sum(1 for r in ranks if r > 200)

    print("\n" + "=" * 78)
    print(" TRUE-MATCH RANK DISTRIBUTION RESULTS (Total Positive Pairs: {:,}):".format(total_pairs))
    print("=" * 78)
    print(f"  * Rank <= 1   : {b_1:>5,} ({100.0 * b_1 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 5   : {b_5:>5,} ({100.0 * b_5 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 10  : {b_10:>5,} ({100.0 * b_10 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 20  : {b_20:>5,} ({100.0 * b_20 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 50  : {b_50:>5,} ({100.0 * b_50 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 100 : {b_100:>5,} ({100.0 * b_100 / total_pairs:>6.2f}%)")
    print(f"  * Rank <= 200 : {b_200:>5,} ({100.0 * b_200 / total_pairs:>6.2f}%)")
    print(f"  * Rank > 200  : {b_gt_200:>5,} ({100.0 * b_gt_200 / total_pairs:>6.2f}%)  [Scored by rules but past cap]")
    print(f"  * Not Scored  : {unretrieved_count:>5,} ({100.0 * unretrieved_count / total_pairs:>6.2f}%)  [Zero rule hits]")
    print("-" * 78)
    print(" PERCENTILES (among retrieved matches):")
    print(f"  * Median (p50): Rank {p50:.1f}")
    print(f"  * p75         : Rank {p75:.1f}")
    print(f"  * p90         : Rank {p90:.1f}")
    print(f"  * p95         : Rank {p95:.1f}")
    print(f"  * p99         : Rank {p99:.1f}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print(" FAILURE ANALYSIS: WHY TRUE MATCHES ARE MISSED (rank > 50 or unretrieved):")
    print("=" * 78)
    total_failures = sum(failure_counts.values())
    for cat, count in failure_counts.most_common():
        pct = (count / total_failures) * 100.0
        pct_of_all = (count / total_pairs) * 100.0
        print(f"  * {cat:48s}: {count:>4,} ({pct:>5.1f}% of misses | {pct_of_all:>5.2f}% of all pairs)")

    # Save Failure Analysis JSON
    failure_report_path = os.path.join(reports_dir, "blocking_failure_analysis.json")
    with open(failure_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_positive_pairs": total_pairs,
            "rank_brackets": {
                "rank_lte_1": b_1,
                "rank_lte_5": b_5,
                "rank_lte_10": b_10,
                "rank_lte_20": b_20,
                "rank_lte_50": b_50,
                "rank_lte_100": b_100,
                "rank_lte_200": b_200,
                "rank_gt_200": b_gt_200,
                "not_retrieved": unretrieved_count
            },
            "percentiles": {"p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99},
            "failure_counts": dict(failure_counts),
            "failure_samples": dict(failure_samples)
        }, f, indent=2)

    print(f"\n✓ Saved rank distribution to: {rank_csv_path}")
    print(f"✓ Saved failure analysis to: {failure_report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
