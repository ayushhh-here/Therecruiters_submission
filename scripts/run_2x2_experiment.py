#!/usr/bin/env python3
"""
scripts/run_2x2_experiment.py — Factorial 2x2 Evaluation & Leak-Free Audit

Implements Phase 19 Requirements 7 & 8:
  - Leak-free 4-way data partition:
      1. TRAIN (60%)       : Used ONLY to train classifier & mine hard negatives
      2. CALIBRATION (15%) : Used ONLY to fit Isotonic probability calibrator
      3. DEV / THRESH (15%): Used ONLY to grid-search decision threshold & margin
      4. FINAL HOLDOUT (10%): Untouched test split for final unbiased evaluation
  - 2x2 Factorial Comparison isolating Retrieval vs Classification gains:
      [1] Old Blocker + Baseline Matcher
      [2] Old Blocker + Improved Matcher
      [3] New Blocker + Baseline Matcher
      [4] New Blocker + Improved Matcher
  - Saves:
      * reports/matrix_2x2_evaluation.csv
      * updates reports/experiments.csv
"""

import os
import sys
import time
import csv
import collections
import polars as pl
import numpy as np
from typing import Dict, List, Set, Tuple, Any

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
from src.blocking import CompactTargetPool, MultiStageBlocker
from src.features import FeatureExtractor, BASELINE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES
from src.matching import EntityMatchingModel
from src.evaluate import evaluate_macro_f05, find_optimal_decision_threshold


# ─── Legacy Old Blocker (Phase 6 implementation without Street/Address index) ───

class OldBlocker:
    """Legacy blocker matching Phase 6 baseline (no street component, no address index)."""
    def __init__(self, target_pool: CompactTargetPool, max_candidates: int = 40):
        self.pool = target_pool
        self.max_candidates = max_candidates
        self.exact_name_idx = collections.defaultdict(list)
        self.sorted_name_idx = collections.defaultdict(list)
        self.postal_idx = collections.defaultdict(list)
        self.token_idx = collections.defaultdict(list)
        self.trigram_idx = collections.defaultdict(list)

        from src.blocking import STOP_TOKENS
        self.stop_tokens = STOP_TOKENS

        for idx in range(len(self.pool)):
            norm_name = self.pool.norm_names[idx]
            postal = self.pool.postal_codes[idx]
            tokens = self.pool.name_tokens_list[idx]
            if norm_name:
                self.exact_name_idx[norm_name].append(idx)
                sorted_key = " ".join(sorted(tokens))
                if sorted_key:
                    self.sorted_name_idx[sorted_key].append(idx)
            if postal:
                self.postal_idx[postal].append(idx)
            for tok in tokens:
                if len(tok) >= 3 and tok not in self.stop_tokens:
                    self.token_idx[tok].append(idx)

    def generate_candidates_for_record(self, s1_rec: NormalizedRecord, custom_max_candidates: int = 40):
        scores = collections.defaultdict(float)
        rules = collections.defaultdict(set)
        s1_country = s1_rec.norm_country

        if s1_rec.norm_name in self.exact_name_idx:
            for tidx in self.exact_name_idx[s1_rec.norm_name]:
                if s1_country == "UNKNOWN" or self.pool.countries[tidx] == "UNKNOWN" or s1_country == self.pool.countries[tidx]:
                    scores[tidx] += 12.0
                    rules[tidx].add("exact_name")

        if s1_rec.sorted_name_key in self.sorted_name_idx:
            for tidx in self.sorted_name_idx[s1_rec.sorted_name_key]:
                if s1_country == "UNKNOWN" or self.pool.countries[tidx] == "UNKNOWN" or s1_country == self.pool.countries[tidx]:
                    scores[tidx] += 9.0
                    rules[tidx].add("sorted_name")

        if s1_rec.postal_code and s1_rec.postal_code in self.postal_idx:
            for tidx in self.postal_idx[s1_rec.postal_code]:
                if s1_country == "UNKNOWN" or self.pool.countries[tidx] == "UNKNOWN" or s1_country == self.pool.countries[tidx]:
                    shared = s1_rec.name_tokens & self.pool.name_tokens_list[tidx]
                    if shared:
                        scores[tidx] += 6.0 + len(shared)
                        rules[tidx].add("postal_code")

        for tok in s1_rec.name_tokens:
            if len(tok) >= 3 and tok not in self.stop_tokens and tok in self.token_idx:
                posting = self.token_idx[tok]
                if len(posting) <= 400:  # hard cap!
                    for tidx in posting:
                        if s1_country == "UNKNOWN" or self.pool.countries[tidx] == "UNKNOWN" or s1_country == self.pool.countries[tidx]:
                            scores[tidx] += 2.5
                            rules[tidx].add("token_overlap")

        sorted_indices = sorted(scores.keys(), key=lambda idx: (len(rules[idx]), scores[idx]), reverse=True)
        top_indices = sorted_indices[:custom_max_candidates]

        ranked_ids = []
        meta_dict = {}
        for tidx in top_indices:
            eid = self.pool.entity_ids[tidx]
            ranked_ids.append(eid)
            meta_dict[eid] = {
                "score": scores[tidx],
                "rule_count": len(rules[tidx]),
                "rules": rules[tidx],
                "target_raw_name": self.pool.raw_names[tidx],
                "target_raw_address": self.pool.raw_addresses[tidx],
                "target_country": self.pool.countries[tidx],
                "target_idx": tidx
            }
        return ranked_ids, meta_dict


# ─── Dataset Feature Builder Helper ──────────────────────────────────────────

def build_features(records, blocker, ground_truth, use_extended=False, mine_hard=False):
    X_rows = []
    y_list = []
    pairs = []
    scores_dict = collections.defaultdict(list)

    for rec in records:
        true_set = set(ground_truth.get(rec.entity_id, []))
        cands, meta = blocker.generate_candidates_for_record(rec)

        for rank_idx, tid in enumerate(cands):
            c_meta = meta.get(tid, {})
            feat = FeatureExtractor.extract_pairwise_features(
                s1=rec,
                target_raw_name=c_meta.get("target_raw_name", ""),
                target_raw_address=c_meta.get("target_raw_address", ""),
                target_country=c_meta.get("target_country", ""),
                candidate_meta=c_meta,
                candidate_rank=rank_idx,
                use_extended_features=use_extended
            )
            is_match = 1 if tid in true_set else 0

            if mine_hard and is_match == 0:
                if c_meta.get("score", 0.0) >= 8.0 or c_meta.get("rule_count", 0) >= 2:
                    X_rows.append(feat)
                    y_list.append(0)
                    pairs.append((rec.entity_id, tid))

            X_rows.append(feat)
            y_list.append(is_match)
            pairs.append((rec.entity_id, tid))

    X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, len(EXTENDED_FEATURE_NAMES if use_extended else BASELINE_FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty((0,), dtype=np.int32)
    return X, y, pairs


def main():
    print("=" * 80)
    print(" PHASE 19: FACTORIAL 2x2 EVALUATION & LEAK-FREE AUDIT (Req 7, 8, 9)")
    print("=" * 80)

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Leak-Free 4-Way Splits
    print("\n[Step 1/5] Loading Leak-Free 4-Way Partitions...")
    train_ids = set(pl.read_csv("splits/sample_train_ids.tsv", separator="\t")["entity_id"].to_list())
    calib_ids = set(pl.read_csv("splits/sample_calib_ids.tsv", separator="\t")["entity_id"].to_list())
    dev_ids = set(pl.read_csv("splits/sample_dev_ids.tsv", separator="\t")["entity_id"].to_list())
    holdout_ids = set(pl.read_csv("splits/sample_holdout_ids.tsv", separator="\t")["entity_id"].to_list())

    print(f"  1. TRAIN Set       : {len(train_ids):,} entities (model fitting & hard negatives)")
    print(f"  2. CALIBRATION Set : {len(calib_ids):,} entities (isotonic calibrator fitting)")
    print(f"  3. DEV/THRESH Set  : {len(dev_ids):,} entities (threshold grid-search)")
    print(f"  4. FINAL HOLDOUT   : {len(holdout_ids):,} entities (single final evaluation)")

    # Load Ground Truth
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
    train_gt, calib_gt, dev_gt, holdout_gt = {}, {}, {}, {}
    needed_targets = set()

    for r in gt_df.iter_rows():
        s1_id = r[0]
        raw = str(r[1] or "").strip()
        matches = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
        if s1_id in train_ids:
            train_gt[s1_id] = matches
            needed_targets.update(matches)
        elif s1_id in calib_ids:
            calib_gt[s1_id] = matches
            needed_targets.update(matches)
        elif s1_id in dev_ids:
            dev_gt[s1_id] = matches
            needed_targets.update(matches)
        elif s1_id in holdout_ids:
            holdout_gt[s1_id] = matches
            needed_targets.update(matches)

    # 2. Build Target Pool (True Targets + 150k Distractors)
    print("\n[Step 2/5] Building Compact Target Pool...")
    pool = CompactTargetPool()
    for s_idx in [2, 3]:
        df_scan = pl.read_csv(f"dataset/train/train_source{s_idx}.tsv", separator="\t", infer_schema_length=0)
        matched = df_scan.filter(pl.col("entity_id").is_in(needed_targets))
        for r in matched.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        for r in df_scan.head(75_000).iter_rows():
            if r[0] not in needed_targets:
                pool.add(r[0], r[1], r[2], r[3])

    print(f"  Target pool populated with {len(pool):,} entities.")

    # 3. Load Normalized S1 Records for all 4 partitions
    print("\n[Step 3/5] Normalizing Query S1 Records...")
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    train_recs, calib_recs, dev_recs, holdout_recs = [], [], [], []

    for r in s1_all.iter_rows():
        eid = r[0]
        if eid in train_ids:
            train_recs.append(NormalizedRecord(eid, r[1], r[2], r[3]))
        elif eid in calib_ids:
            calib_recs.append(NormalizedRecord(eid, r[1], r[2], r[3]))
        elif eid in dev_ids:
            dev_recs.append(NormalizedRecord(eid, r[1], r[2], r[3]))
        elif eid in holdout_ids:
            holdout_recs.append(NormalizedRecord(eid, r[1], r[2], r[3]))

    # Instantiate Old Blocker and New Blocker
    old_blocker = OldBlocker(pool, max_candidates=40)
    new_blocker = MultiStageBlocker(pool, max_candidates=40, enable_adaptive_budget=True)

    # 4. Execute Factorial 2x2 Configurations
    print("\n[Step 4/5] Executing 2x2 Factorial Matrix on Untouched Holdout Split...")
    matrix_results = []

    configs = [
        ("Old Blocking + Baseline Matcher", old_blocker, False, False, False, "fixed"),
        ("Old Blocking + Improved Matcher", old_blocker, True, True, True, "margin"),
        ("New Blocking + Baseline Matcher", new_blocker, False, False, False, "fixed"),
        ("New Blocking + Improved Matcher", new_blocker, True, True, True, "margin"),
    ]

    for cfg_name, blocker, use_ext, use_hard, use_calib, decision_mode in configs:
        print(f"\n  Running Configuration: {cfg_name}...")
        t_start = time.time()

        # Step A: Train Matcher on TRAIN set
        X_tr, y_tr, _ = build_features(train_recs, blocker, train_gt, use_extended=use_ext, mine_hard=use_hard)
        model = EntityMatchingModel(model_type="lightgbm", use_extended_features=use_ext, random_state=42)
        model.fit(X_tr, y_tr)

        # Step B: Calibrate on CALIBRATION set (if enabled)
        if use_calib:
            X_cal, y_cal, _ = build_features(calib_recs, blocker, calib_gt, use_extended=use_ext, mine_hard=False)
            model.calibrate(X_cal, y_cal, method="isotonic")

        # Step C: Threshold Tuning on DEV set
        X_dev, y_dev, dev_pairs = build_features(dev_recs, blocker, dev_gt, use_extended=use_ext, mine_hard=False)
        dev_probs = model.predict_proba(X_dev)
        dev_scores = collections.defaultdict(list)
        for (s1_id, tid), p in zip(dev_pairs, dev_probs):
            dev_scores[s1_id].append((tid, float(p)))
        for rec in dev_recs:
            if rec.entity_id not in dev_scores:
                dev_scores[rec.entity_id] = []

        best_thresh, _, _ = find_optimal_decision_threshold(dev_scores, dev_gt)

        # Step D: Final Unbiased Evaluation on FINAL HOLDOUT set
        X_test, y_test, test_pairs = build_features(holdout_recs, blocker, holdout_gt, use_extended=use_ext, mine_hard=False)
        test_probs = model.predict_proba(X_test)
        test_scores = collections.defaultdict(list)
        for (s1_id, tid), p in zip(test_pairs, test_probs):
            test_scores[s1_id].append((tid, float(p)))
        for rec in holdout_recs:
            if rec.entity_id not in test_scores:
                test_scores[rec.entity_id] = []

        # Predict matches using decision logic
        if decision_mode == "margin":
            decisions = model.make_entity_decisions(test_scores, decision_threshold=best_thresh, margin_threshold=0.20)
            pred_matches = {s: set(d["accepted_matches"]) for s, d in decisions.items()}
        else:
            pred_matches = {
                s: {tid for tid, p in pairs if p >= best_thresh}
                for s, pairs in test_scores.items()
            }

        eval_res = evaluate_macro_f05(pred_matches, holdout_gt)
        runtime = round(time.time() - t_start, 2)

        tot_preds = sum(len(m) for m in pred_matches.values())
        res_row = {
            "configuration": cfg_name,
            "optimal_threshold": round(best_thresh, 2),
            "macro_f05": round(eval_res["macro_f05"], 4),
            "precision": round(eval_res["macro_precision"], 4),
            "recall": round(eval_res["macro_recall"], 4),
            "singleton_f05": round(eval_res["singleton_f05"], 4),
            "non_singleton_f05": round(eval_res["non_singleton_f05"], 4),
            "total_candidates": len(X_test),
            "total_predictions": tot_preds,
            "runtime_sec": runtime
        }
        matrix_results.append(res_row)

        print(f"    Holdout Macro F0.5 : {res_row['macro_f05']:.4f}")
        print(f"    Holdout Precision  : {res_row['precision']:.4f}")
        print(f"    Holdout Recall     : {res_row['recall']:.4f}")
        print(f"    Singleton F0.5     : {res_row['singleton_f05']:.4f}")
        print(f"    Non-Singleton F0.5 : {res_row['non_singleton_f05']:.4f}")
        print(f"    Runtime            : {runtime}s")

    # 5. Output 2x2 Matrix Report
    print("\n[Step 5/5] Saving 2x2 Evaluation Matrix...")
    csv_matrix_path = os.path.join(reports_dir, "matrix_2x2_evaluation.csv")
    with open(csv_matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "configuration", "optimal_threshold", "macro_f05", "precision",
            "recall", "singleton_f05", "non_singleton_f05",
            "total_candidates", "total_predictions", "runtime_sec"
        ])
        writer.writeheader()
        writer.writerows(matrix_results)

    print("\n" + "=" * 80)
    print(" FACTORIAL 2x2 EVALUATION RESULTS (ON UNTOUCHED FINAL HOLDOUT SPLIT):")
    print("=" * 80)
    print(f"  {'Configuration':36s} | {'Macro F0.5':10s} | {'Precision':10s} | {'Recall':8s} | {'Non-Single':10s} | {'Thresh':6s}")
    print("-" * 80)
    for r in matrix_results:
        print(f"  {r['configuration']:36s} | {r['macro_f05']:>10.4f} | {r['precision']:>10.4f} | {r['recall']:>8.4f} | {r['non_singleton_f05']:>10.4f} | {r['optimal_threshold']:>6.2f}")
    print("=" * 80)
    print(f"✓ Saved 2x2 matrix to: {csv_matrix_path}")


if __name__ == "__main__":
    main()
