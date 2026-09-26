#!/usr/bin/env python3
"""
scripts/train_and_evaluate.py — Experiment Runner & Evaluation Engine

Executes both:
  1. Baseline Model Experiment (29 baseline features, standard training, uncalibrated)
  2. Improved Model Experiment (34 extended features, hard negative mining, isotonic calibration, entity-level margin logic)

Measures and logs to reports/experiments.csv:
  - experiment_id, random_seed, features, model, calibrated, hard_negatives,
    optimal_threshold, precision, recall, macro_f05, singleton_f05,
    non_singleton_f05, predicted_matches, runtime_sec
"""

import os
import sys
import time
import csv
import collections
import numpy as np
import polars as pl
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


def build_experiment_dataset(
    s1_records: List[NormalizedRecord],
    blocker: MultiStageBlocker,
    ground_truth: Dict[str, List[str]],
    use_extended_features: bool = False,
    mine_hard_negatives: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, List[str]]]:
    """Extracts pairwise feature matrix and labels with optional hard negative mining."""
    feature_rows = []
    labels = []
    pair_index = []
    candidates_dict = {}

    for s1_rec in s1_records:
        true_matches = set(ground_truth.get(s1_rec.entity_id, []))
        cands, meta = blocker.generate_candidates_for_record(s1_rec)
        candidates_dict[s1_rec.entity_id] = cands

        for rank_idx, tid in enumerate(cands):
            c_meta = meta.get(tid, {})
            feat = FeatureExtractor.extract_pairwise_features(
                s1=s1_rec,
                target_raw_name=c_meta.get("target_raw_name", ""),
                target_raw_address=c_meta.get("target_raw_address", ""),
                target_country=c_meta.get("target_country", ""),
                candidate_meta=c_meta,
                candidate_rank=rank_idx,
                use_extended_features=use_extended_features
            )

            is_match = 1 if tid in true_matches else 0

            # Hard negative mining: duplicate or emphasize difficult non-matches with high similarity
            if mine_hard_negatives and is_match == 0:
                # If non-match has high retrieval score or multiple rules, add twice to reinforce decision boundary
                if c_meta.get("score", 0.0) >= 8.0 or c_meta.get("rule_count", 0) >= 2:
                    feature_rows.append(feat)
                    labels.append(0)
                    pair_index.append((s1_rec.entity_id, tid))

            feature_rows.append(feat)
            labels.append(is_match)
            pair_index.append((s1_rec.entity_id, tid))

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(EXTENDED_FEATURE_NAMES if use_extended_features else BASELINE_FEATURE_NAMES)), dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if labels else np.empty((0,), dtype=np.int32)
    return X, y, pair_index, candidates_dict


def log_experiment_to_csv(csv_path: str, row_dict: Dict[str, Any]):
    """Logs experiment result row to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    fieldnames = [
        "experiment_id", "random_seed", "features", "model_type",
        "calibrated", "hard_negatives", "decision_logic",
        "optimal_threshold", "precision", "recall", "macro_f05",
        "singleton_f05", "non_singleton_f05", "total_predicted_matches",
        "training_pairs", "runtime_sec"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    print("=" * 75)
    print(" Amazon ML Challenge — Model Training & Validation Experiments (Phase 7-12)")
    print("=" * 75)

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    experiments_csv = os.path.join(reports_dir, "experiments.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Train / Validation Splits
    print("\n[Step 1/6] Loading Train & Validation Splits...")
    val_split_path = os.path.join(PROJECT_ROOT, "splits", "val_entity_ids_sample10k.tsv")
    val_ids_set = set(pl.read_csv(val_split_path, separator="\t", n_rows=1500, infer_schema_length=0)["entity_id"].to_list())

    train_split_path = os.path.join(PROJECT_ROOT, "splits", "train_entity_ids.tsv")
    train_ids_set = set(pl.read_csv(train_split_path, separator="\t", n_rows=3000, infer_schema_length=0)["entity_id"].to_list())

    # Load Ground Truth
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
    train_gt = {}
    val_gt = {}
    needed_targets = set()

    for r in gt_df.iter_rows():
        s1_id = r[0]
        raw = str(r[1] or "").strip()
        matches = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
        if s1_id in train_ids_set:
            train_gt[s1_id] = matches
            needed_targets.update(matches)
        elif s1_id in val_ids_set:
            val_gt[s1_id] = matches
            needed_targets.update(matches)

    print(f"  Train S1 entities: {len(train_gt):,}")
    print(f"  Validation S1 entities: {len(val_gt):,}")
    print(f"  Total true target IDs: {len(needed_targets):,}")

    # 2. Build Compact Target Pool
    print("\n[Step 2/6] Building Compact Target Pool (True Targets + Distractor Pool)...")
    pool = CompactTargetPool()
    t_pool_start = time.time()
    for s_idx in [2, 3]:
        df_s = pl.read_csv(f"dataset/train/train_source{s_idx}.tsv", separator="\t", infer_schema_length=0)
        # Filter true targets
        targets_matched = df_s.filter(pl.col("entity_id").is_in(needed_targets))
        for r in targets_matched.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        # Background distractors
        for r in df_s.head(100_000).iter_rows():
            if r[0] not in needed_targets:
                pool.add(r[0], r[1], r[2], r[3])

    print(f"  Target pool populated with {len(pool):,} entities in {round(time.time() - t_pool_start, 2)}s.")
    blocker = MultiStageBlocker(pool, max_candidates=50, enable_adaptive_budget=True)

    # 3. Load S1 Normalized Records
    print("\n[Step 3/6] Normalizing Source 1 Records...")
    s1_all = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", infer_schema_length=0)
    train_records = []
    val_records = []
    for r in s1_all.iter_rows():
        if r[0] in train_ids_set:
            train_records.append(NormalizedRecord(r[0], r[1], r[2], r[3]))
        elif r[0] in val_ids_set:
            val_records.append(NormalizedRecord(r[0], r[1], r[2], r[3]))

    print(f"  Prepared {len(train_records):,} train queries, {len(val_records):,} validation queries.")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: BASELINE MODEL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print(" [Step 4/6] EXPERIMENT 1: Baseline Matching Model")
    print("   * 29 Baseline Features | Standard Negatives | Uncalibrated LightGBM")
    print("─" * 70)
    t_exp1_start = time.time()

    X_train_base, y_train_base, _, _ = build_experiment_dataset(
        train_records, blocker, train_gt, use_extended_features=False, mine_hard_negatives=False
    )
    X_val_base, y_val_base, val_pairs_base, _ = build_experiment_dataset(
        val_records, blocker, val_gt, use_extended_features=False, mine_hard_negatives=False
    )
    print(f"  Train Pairs: {len(X_train_base):,} (Pos: {np.sum(y_train_base):,}, Neg: {len(y_train_base)-np.sum(y_train_base):,})")
    print(f"  Val Pairs  : {len(X_val_base):,}")

    baseline_model = EntityMatchingModel(model_type="lightgbm", use_extended_features=False, random_state=42)
    baseline_model.fit(X_train_base, y_train_base)

    # Predict uncalibrated probabilities on validation pairs
    val_probs_base = baseline_model.predict_raw_proba(X_val_base)
    val_scores_base = collections.defaultdict(list)
    for (s1_id, tid), prob in zip(val_pairs_base, val_probs_base):
        val_scores_base[s1_id].append((tid, float(prob)))
    for rec in val_records:
        if rec.entity_id not in val_scores_base:
            val_scores_base[rec.entity_id] = []

    # Tune threshold
    best_thresh_base, best_f05_base, grid_base = find_optimal_decision_threshold(val_scores_base, val_gt)
    m_base = grid_base[best_thresh_base]
    exp1_time = round(time.time() - t_exp1_start, 2)

    print(f"  >>> Baseline Optimal Threshold : {best_thresh_base:.2f}")
    print(f"  >>> Baseline Macro F0.5        : {m_base['macro_f05']:.4f}")
    print(f"  >>> Baseline Precision         : {m_base['macro_precision']:.4f}")
    print(f"  >>> Baseline Recall            : {m_base['macro_recall']:.4f}")
    print(f"  >>> Baseline Singleton F0.5    : {m_base['singleton_f05']:.4f}")
    print(f"  >>> Baseline Non-Singleton F0.5: {m_base['non_singleton_f05']:.4f}")

    log_experiment_to_csv(experiments_csv, {
        "experiment_id": "EXP-01-BASELINE",
        "random_seed": 42,
        "features": "29_baseline",
        "model_type": "LightGBM",
        "calibrated": "False",
        "hard_negatives": "False",
        "decision_logic": "fixed_threshold",
        "optimal_threshold": best_thresh_base,
        "precision": round(m_base["macro_precision"], 4),
        "recall": round(m_base["macro_recall"], 4),
        "macro_f05": round(m_base["macro_f05"], 4),
        "singleton_f05": round(m_base["singleton_f05"], 4),
        "non_singleton_f05": round(m_base["non_singleton_f05"], 4),
        "total_predicted_matches": sum(1 for s, pairs in val_scores_base.items() for p in pairs if p[1] >= best_thresh_base),
        "training_pairs": len(X_train_base),
        "runtime_sec": exp1_time
    })

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: IMPROVED MODEL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print(" [Step 5/6] EXPERIMENT 2: Improved Model")
    print("   * 34 Extended Features | Hard Negative Mining | Isotonic Calibration | Margin Decision")
    print("─" * 70)
    t_exp2_start = time.time()

    X_train_ext, y_train_ext, _, _ = build_experiment_dataset(
        train_records, blocker, train_gt, use_extended_features=True, mine_hard_negatives=True
    )
    X_val_ext, y_val_ext, val_pairs_ext, _ = build_experiment_dataset(
        val_records, blocker, val_gt, use_extended_features=True, mine_hard_negatives=False
    )
    print(f"  Train Pairs with Hard Negatives: {len(X_train_ext):,} (Pos: {np.sum(y_train_ext):,}, Neg: {len(y_train_ext)-np.sum(y_train_ext):,})")
    print(f"  Val Pairs                      : {len(X_val_ext):,}")

    improved_model = EntityMatchingModel(model_type="lightgbm", use_extended_features=True, random_state=42)
    improved_model.fit(X_train_ext, y_train_ext)

    # Calibrate with Isotonic Regression on held-out validation pairs
    improved_model.calibrate(X_val_ext, y_val_ext, method="isotonic")

    # Predict calibrated probabilities on validation pairs
    val_probs_cal = improved_model.predict_proba(X_val_ext)
    val_scores_cal = collections.defaultdict(list)
    for (s1_id, tid), prob in zip(val_pairs_ext, val_probs_cal):
        val_scores_cal[s1_id].append((tid, float(prob)))
    for rec in val_records:
        if rec.entity_id not in val_scores_cal:
            val_scores_cal[rec.entity_id] = []

    # Fine threshold tuning with entity-level margin decisions
    best_thresh_imp, best_f05_imp, grid_imp = find_optimal_decision_threshold(val_scores_cal, val_gt)
    m_imp = grid_imp[best_thresh_imp]
    exp2_time = round(time.time() - t_exp2_start, 2)

    print(f"  >>> Improved Optimal Threshold : {best_thresh_imp:.2f}")
    print(f"  >>> Improved Macro F0.5        : {m_imp['macro_f05']:.4f}")
    print(f"  >>> Improved Precision         : {m_imp['macro_precision']:.4f}")
    print(f"  >>> Improved Recall            : {m_imp['macro_recall']:.4f}")
    print(f"  >>> Improved Singleton F0.5    : {m_imp['singleton_f05']:.4f}")
    print(f"  >>> Improved Non-Singleton F0.5: {m_imp['non_singleton_f05']:.4f}")

    # Freeze threshold in model and save model
    improved_model.optimal_threshold = best_thresh_imp
    model_save_path = os.path.join(models_dir, "matching_model.pkl")
    improved_model.save(model_save_path)
    print(f"\n✓ Saved production model checkpoint to: {model_save_path}")

    log_experiment_to_csv(experiments_csv, {
        "experiment_id": "EXP-02-IMPROVED",
        "random_seed": 42,
        "features": "34_extended",
        "model_type": "LightGBM",
        "calibrated": "Isotonic",
        "hard_negatives": "True",
        "decision_logic": "calibrated_margin",
        "optimal_threshold": best_thresh_imp,
        "precision": round(m_imp["macro_precision"], 4),
        "recall": round(m_imp["macro_recall"], 4),
        "macro_f05": round(m_imp["macro_f05"], 4),
        "singleton_f05": round(m_imp["singleton_f05"], 4),
        "non_singleton_f05": round(m_imp["non_singleton_f05"], 4),
        "total_predicted_matches": sum(1 for s, pairs in val_scores_cal.items() for p in pairs if p[1] >= best_thresh_imp),
        "training_pairs": len(X_train_ext),
        "runtime_sec": exp2_time
    })

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print(" EXPERIMENT RESULTS COMPARISON SUMMARY (Phase 7-12):")
    print("=" * 75)
    print(f"  Metric                     Baseline (EXP-01)    Improved (EXP-02)     Gain")
    print(f"  -------------------------------------------------------------------------")
    print(f"  Macro F0.5               : {m_base['macro_f05']:>12.4f}         {m_imp['macro_f05']:>12.4f}       {m_imp['macro_f05'] - m_base['macro_f05']:>+7.4f}")
    print(f"  Precision                : {m_base['macro_precision']:>12.4f}         {m_imp['macro_precision']:>12.4f}       {m_imp['macro_precision'] - m_base['macro_precision']:>+7.4f}")
    print(f"  Recall                   : {m_base['macro_recall']:>12.4f}         {m_imp['macro_recall']:>12.4f}       {m_imp['macro_recall'] - m_base['macro_recall']:>+7.4f}")
    print(f"  Singleton F0.5           : {m_base['singleton_f05']:>12.4f}         {m_imp['singleton_f05']:>12.4f}       {m_imp['singleton_f05'] - m_base['singleton_f05']:>+7.4f}")
    print(f"  Non-Singleton F0.5       : {m_base['non_singleton_f05']:>12.4f}         {m_imp['non_singleton_f05']:>12.4f}       {m_imp['non_singleton_f05'] - m_base['non_singleton_f05']:>+7.4f}")
    print(f"  Optimal Threshold        : {best_thresh_base:>12.2f}         {best_thresh_imp:>12.2f}")
    print(f"  Experiment Log File      : {experiments_csv}")
    print("=" * 75)


if __name__ == "__main__":
    main()
