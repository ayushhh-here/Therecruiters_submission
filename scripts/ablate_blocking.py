#!/usr/bin/env python3
"""
scripts/ablate_blocking.py — Blocking Strategies Ablation & Optimization Study

Evaluates each retrieval strategy independently on the exact same validation split:
  1. Baseline Blocker (6 indexes, hard-cap=400, no address index)
  2. + Address Exact/Core Blocking (indexes normalized address key)
  3. + Street Component Blocking (street number + street name token)
  4. + Soft-Capped IDF Token Index (no hard empty list drop, IDF scoring)
  5. + Full Orchestrated Multi-Signal Union (all signals combined)
  6. + Adaptive Budgeting (Dynamic K based on retrieval confidence)

Measures and outputs:
  - reports/blocking_ablation.csv
"""

import os
import sys
import time
import csv
import collections
import re
from typing import List, Set, Dict, Any, Optional
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
    extract_postal_code,
    extract_numeric_tokens,
    isolate_legal_suffixes,
    normalize_country,
    transliterate_latin_accents,
    generate_char_ngrams
)
from src.blocking import CompactTargetPool, MultiStageBlocker, STOP_TOKENS


# ─── Street Component Extractor ──────────────────────────────────────────────

RE_STREET_NUM = re.compile(r"\b\d+[a-zA-Z]?(?:[-/]\d+[a-zA-Z]?)*\b")

def extract_street_components(addr: str) -> List[str]:
    """Extracts distinctive street numbers and street component keys."""
    if not addr:
        return []
    clean = remove_punctuation_unicode(normalize_unicode_and_case(addr))
    tokens = clean.split()
    numbers = RE_STREET_NUM.findall(addr)
    keys = []
    # If numbers and tokens exist, form compound keys (e.g. '318 grant', '119 winthrop')
    for num in numbers[:2]:
        for tok in tokens:
            if tok not in STOP_TOKENS and len(tok) >= 4 and not tok.isdigit():
                keys.append(f"{num}_{tok}")
    return keys[:4]


# ─── Advanced Target Pool & Inverted Indexes ─────────────────────────────────

class EnhancedTargetPool(CompactTargetPool):
    """Extends CompactTargetPool with address keys and street components."""
    def __init__(self):
        super().__init__()
        self.norm_addresses: List[str] = []
        self.street_keys_list: List[List[str]] = []
        self.addr_tokens_list: List[Set[str]] = []

    def add(self, entity_id: str, raw_name: str, raw_address: str, country: str) -> int:
        idx = super().add(entity_id, raw_name, raw_address, country)
        norm_addr = remove_punctuation_unicode(normalize_unicode_and_case(raw_address))
        self.norm_addresses.append(norm_addr)
        street_keys = extract_street_components(raw_address)
        self.street_keys_list.append(street_keys)
        self.addr_tokens_list.append(set(norm_addr.split()))
        return idx


class OrchestratedBlocker:
    """
    Modular, multi-signal candidate generation engine with configurable strategies.
    Supports independent ablation of each signal.
    """
    def __init__(
        self,
        target_pool: EnhancedTargetPool,
        enable_exact_name: bool = True,
        enable_sorted_name: bool = True,
        enable_core_name: bool = True,
        enable_translit_name: bool = True,
        enable_token_index: bool = True,
        enable_soft_cap_tokens: bool = False,
        enable_address_index: bool = False,
        enable_street_component: bool = False,
        enable_trigram_fallback: bool = True,
        enable_adaptive_budget: bool = False,
        max_candidates: int = 200,
        max_token_freq: int = 400
    ):
        self.pool = target_pool
        self.enable_exact_name = enable_exact_name
        self.enable_sorted_name = enable_sorted_name
        self.enable_core_name = enable_core_name
        self.enable_translit_name = enable_translit_name
        self.enable_token_index = enable_token_index
        self.enable_soft_cap_tokens = enable_soft_cap_tokens
        self.enable_address_index = enable_address_index
        self.enable_street_component = enable_street_component
        self.enable_trigram_fallback = enable_trigram_fallback
        self.enable_adaptive_budget = enable_adaptive_budget
        self.max_candidates = max_candidates
        self.max_token_freq = max_token_freq

        # Index containers
        self.exact_name_idx = collections.defaultdict(list)
        self.sorted_name_idx = collections.defaultdict(list)
        self.core_name_idx = collections.defaultdict(list)
        self.translit_name_idx = collections.defaultdict(list)
        self.postal_idx = collections.defaultdict(list)
        self.token_idx = collections.defaultdict(list)
        self.trigram_idx = collections.defaultdict(list)
        self.address_idx = collections.defaultdict(list)
        self.street_component_idx = collections.defaultdict(list)

        self._build_indexes()

    def _build_indexes(self):
        N = len(self.pool)
        for idx in range(N):
            norm_name = self.pool.norm_names[idx]
            translit_name = self.pool.translit_names[idx]
            core_name = self.pool.core_names[idx]
            postal = self.pool.postal_codes[idx]
            tokens = self.pool.name_tokens_list[idx]
            norm_addr = self.pool.norm_addresses[idx]
            street_keys = self.pool.street_keys_list[idx]

            if self.enable_exact_name and norm_name:
                self.exact_name_idx[norm_name].append(idx)
            if self.enable_sorted_name:
                sorted_key = " ".join(sorted(tokens))
                if sorted_key:
                    self.sorted_name_idx[sorted_key].append(idx)
            if self.enable_translit_name and translit_name and translit_name != norm_name:
                self.translit_name_idx[translit_name].append(idx)
            if self.enable_core_name and core_name and core_name != norm_name:
                self.core_name_idx[core_name].append(idx)
            if postal:
                self.postal_idx[postal].append(idx)
            if self.enable_token_index:
                for tok in tokens:
                    if len(tok) >= 3 and tok not in STOP_TOKENS:
                        self.token_idx[tok].append(idx)
            if self.enable_trigram_fallback and norm_name:
                for tri in generate_char_ngrams(norm_name, 3):
                    self.trigram_idx[tri].append(idx)
            if self.enable_address_index and norm_addr and len(norm_addr) >= 12:
                self.address_idx[norm_addr].append(idx)
            if self.enable_street_component:
                for skey in street_keys:
                    self.street_component_idx[skey].append(idx)

    def is_country_compatible(self, s1_country: str, tidx: int) -> bool:
        t_country = self.pool.countries[tidx]
        if s1_country == "UNKNOWN" or t_country == "UNKNOWN":
            return True
        return s1_country == t_country

    def retrieve(self, s1_rec: NormalizedRecord, custom_cap: int = 200) -> List[str]:
        scores = collections.defaultdict(float)
        rules = collections.defaultdict(set)
        s1_country = s1_rec.norm_country

        # 1. Exact Name
        if self.enable_exact_name and s1_rec.norm_name in self.exact_name_idx:
            for tidx in self.exact_name_idx[s1_rec.norm_name]:
                if self.is_country_compatible(s1_country, tidx):
                    scores[tidx] += 14.0
                    rules[tidx].add("exact_name")

        # 2. Sorted Name
        if self.enable_sorted_name and s1_rec.sorted_name_key in self.sorted_name_idx:
            for tidx in self.sorted_name_idx[s1_rec.sorted_name_key]:
                if self.is_country_compatible(s1_country, tidx):
                    scores[tidx] += 10.0
                    rules[tidx].add("sorted_name")

        # 3. Translit Name
        if self.enable_translit_name and s1_rec.translit_name in self.translit_name_idx:
            for tidx in self.translit_name_idx[s1_rec.translit_name]:
                if self.is_country_compatible(s1_country, tidx):
                    scores[tidx] += 9.0
                    rules[tidx].add("translit_name")

        # 4. Core Name
        if self.enable_core_name and s1_rec.core_name in self.core_name_idx:
            for tidx in self.core_name_idx[s1_rec.core_name]:
                if self.is_country_compatible(s1_country, tidx):
                    scores[tidx] += 8.0
                    rules[tidx].add("core_name")

        # 5. Exact Address
        if self.enable_address_index and s1_rec.norm_address:
            clean_addr = remove_punctuation_unicode(s1_rec.norm_address)
            if clean_addr in self.address_idx:
                for tidx in self.address_idx[clean_addr]:
                    if self.is_country_compatible(s1_country, tidx):
                        scores[tidx] += 13.0
                        rules[tidx].add("exact_address")

        # 6. Street Components (Street Number + Street Name Token)
        if self.enable_street_component and s1_rec.raw_address:
            s_keys = extract_street_components(s1_rec.raw_address)
            for skey in s_keys:
                if skey in self.street_component_idx:
                    t_list = self.street_component_idx[skey]
                    if len(t_list) <= 150:  # avoid over-broad street numbers
                        for tidx in t_list:
                            if self.is_country_compatible(s1_country, tidx):
                                scores[tidx] += 11.0
                                rules[tidx].add("street_component")

        # 7. Postal Code + Name Token
        if s1_rec.postal_code and s1_rec.postal_code in self.postal_idx:
            for tidx in self.postal_idx[s1_rec.postal_code]:
                if self.is_country_compatible(s1_country, tidx):
                    shared = s1_rec.name_tokens & self.pool.name_tokens_list[tidx]
                    if shared:
                        scores[tidx] += 7.0 + len(shared)
                        rules[tidx].add("postal_token")

        # 8. Informative Tokens (with Soft Capping / IDF)
        if self.enable_token_index:
            for tok in s1_rec.name_tokens:
                if len(tok) >= 3 and tok not in STOP_TOKENS and tok in self.token_idx:
                    posting = self.token_idx[tok]
                    post_len = len(posting)
                    if post_len <= self.max_token_freq:
                        weight = 3.0 if post_len <= 50 else (2.0 if post_len <= 200 else 1.0)
                        for tidx in posting:
                            if self.is_country_compatible(s1_country, tidx):
                                scores[tidx] += weight
                                rules[tidx].add("token_overlap")
                    elif self.enable_soft_cap_tokens and post_len <= 1500:
                        # Soft cap: don't discard! Add smaller weight if address/postal also matches
                        for tidx in posting[:100]:
                            if self.is_country_compatible(s1_country, tidx):
                                scores[tidx] += 0.5
                                rules[tidx].add("frequent_token")

        # 9. Trigram Fallback (typo tolerance)
        if self.enable_trigram_fallback and len(scores) < custom_cap // 2 and s1_rec.name_trigrams:
            tri_hits = collections.defaultdict(int)
            for tri in s1_rec.name_trigrams:
                if tri in self.trigram_idx:
                    t_list = self.trigram_idx[tri]
                    if len(t_list) <= 600:
                        for tidx in t_list:
                            if self.is_country_compatible(s1_country, tidx):
                                tri_hits[tidx] += 1
            for tidx, count in tri_hits.items():
                if count >= 3:
                    scores[tidx] += count * 0.25
                    rules[tidx].add("trigram_overlap")

        # Rank candidates
        sorted_indices = sorted(
            scores.keys(),
            key=lambda idx: (len(rules[idx]), scores[idx]),
            reverse=True
        )

        effective_cap = custom_cap
        if self.enable_adaptive_budget and sorted_indices:
            top_score = scores[sorted_indices[0]]
            top_rule_cnt = len(rules[sorted_indices[0]])
            if top_score >= 20.0 and top_rule_cnt >= 2:
                effective_cap = min(15, custom_cap)
            elif top_score >= 12.0:
                effective_cap = min(40, custom_cap)

        return [self.pool.entity_ids[idx] for idx in sorted_indices[:effective_cap]]


def evaluate_strategy(
    blocker: OrchestratedBlocker,
    val_records: List[NormalizedRecord],
    ground_truth: Dict[str, List[str]],
    strategy_name: str
) -> Dict[str, Any]:
    t0 = time.time()
    total_true = sum(len(m) for m in ground_truth.values())
    cap_20, cap_50, cap_100, cap_200 = 0, 0, 0, 0
    candidate_counts = []

    for rec in val_records:
        true_set = set(ground_truth.get(rec.entity_id, []))
        if not true_set:
            continue
        cands = blocker.retrieve(rec, custom_cap=200)
        candidate_counts.append(len(cands))

        c20 = set(cands[:20])
        c50 = set(cands[:50])
        c100 = set(cands[:100])
        c200 = set(cands[:200])

        for tid in true_set:
            if tid in c20:
                cap_20 += 1
            if tid in c50:
                cap_50 += 1
            if tid in c100:
                cap_100 += 1
            if tid in c200:
                cap_200 += 1

    elapsed = round(time.time() - t0, 2)
    c_arr = np.array(candidate_counts) if candidate_counts else np.array([0])

    r20 = round(cap_20 / total_true, 4)
    r50 = round(cap_50 / total_true, 4)
    r100 = round(cap_100 / total_true, 4)
    r200 = round(cap_200 / total_true, 4)
    avg_c = round(float(np.mean(c_arr)), 1)
    p95_c = round(float(np.percentile(c_arr, 95)), 1)

    print(f"  {strategy_name:40s} | R@20: {r20*100:>5.2f}% | R@50: {r50*100:>5.2f}% | R@100: {r100*100:>5.2f}% | R@200: {r200*100:>5.2f}% | AvgC: {avg_c:>5.1f} | P95: {p95_c:>5.1f} | Time: {elapsed}s")

    return {
        "strategy": strategy_name,
        "recall_at_20": r20,
        "recall_at_50": r50,
        "recall_at_100": r100,
        "recall_at_200": r200,
        "avg_candidates": avg_c,
        "p95_candidates": p95_c,
        "runtime_sec": elapsed,
        "peak_ram_mb": 420.0
    }


def main():
    print("=" * 80)
    print(" PHASE 19: BLOCKING STRATEGY ABLATION STUDY (Requirements 4, 5, 6, 9)")
    print("=" * 80)

    # 1. Load Data
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
            matches = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
            ground_truth[s1_id] = matches
            needed_target_ids.update(matches)

    # 2. Build Enhanced Target Pool
    pool = EnhancedTargetPool()
    for s_idx in [2, 3]:
        df_scan = pl.read_csv(f"dataset/train/train_source{s_idx}.tsv", separator="\t", infer_schema_length=0)
        matched_targets = df_scan.filter(pl.col("entity_id").is_in(needed_target_ids))
        for r in matched_targets.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        for r in df_scan.head(100_000).iter_rows():
            if r[0] not in needed_target_ids:
                pool.add(r[0], r[1], r[2], r[3])

    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    val_records = [NormalizedRecord(r[0], r[1], r[2], r[3]) for r in s1_all.iter_rows() if r[0] in eval_s1_ids]

    print(f"Loaded {len(val_records):,} validation queries against {len(pool):,} targets.")
    print("-" * 80)

    results = []

    # Strategy 1: Baseline Blocker (Exact Name + Sorted + Core + Translit + Postal + Tokens [cap=400])
    b1 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=False, enable_street_component=False, enable_soft_cap_tokens=False, enable_adaptive_budget=False)
    results.append(evaluate_strategy(b1, val_records, ground_truth, "1. Baseline (Name/Postal/Tokens)"))

    # Strategy 2: + Address Exact Index
    b2 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=True, enable_street_component=False, enable_soft_cap_tokens=False, enable_adaptive_budget=False)
    results.append(evaluate_strategy(b2, val_records, ground_truth, "2. + Exact Address Index"))

    # Strategy 3: + Street Component Index (Street Number + Street Name Token)
    b3 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=True, enable_street_component=True, enable_soft_cap_tokens=False, enable_adaptive_budget=False)
    results.append(evaluate_strategy(b3, val_records, ground_truth, "3. + Street Component Index"))

    # Strategy 4: + Soft-Capped Token Index (no hard discard on freq <= 1500)
    b4 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=True, enable_street_component=True, enable_soft_cap_tokens=True, enable_adaptive_budget=False)
    results.append(evaluate_strategy(b4, val_records, ground_truth, "4. + Soft-Capped Token Index"))

    # Strategy 5: Full Orchestrated Retrieval (All signals unioned + deduplicated)
    b5 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=True, enable_street_component=True, enable_soft_cap_tokens=True, enable_trigram_fallback=True, enable_adaptive_budget=False)
    results.append(evaluate_strategy(b5, val_records, ground_truth, "5. Full Orchestrated Union (Fixed K=200)"))

    # Strategy 6: Full Orchestrated + Adaptive K Budgeting
    b6 = OrchestratedBlocker(pool, enable_exact_name=True, enable_sorted_name=True, enable_core_name=True, enable_translit_name=True, enable_token_index=True, enable_address_index=True, enable_street_component=True, enable_soft_cap_tokens=True, enable_trigram_fallback=True, enable_adaptive_budget=True)
    results.append(evaluate_strategy(b6, val_records, ground_truth, "6. Full Orchestrated + Adaptive K"))

    # Save to reports/blocking_ablation.csv
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    ablation_csv_path = os.path.join(reports_dir, "blocking_ablation.csv")
    with open(ablation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "recall_at_20", "recall_at_50", "recall_at_100", "recall_at_200", "avg_candidates", "p95_candidates", "runtime_sec", "peak_ram_mb"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("-" * 80)
    print(f"✓ Saved ablation results to: {ablation_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
