#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/pipeline.py
#
# Description:
#   End-to-End Orchestrator Pipeline:
#     1. Data Loading:
#        - Reads TSV files with explicit sep='\t' (avoids single-column comma corruption).
#        - Normalizes all business records into NormalizedRecord objects.
#     2. Leakage-Safe Train / Validation Split:
#        - Holds out 20% of Source 1 entities for validation.
#        - Ensures zero information leakage across splits.
#     3. Candidate Generation (Blocking):
#        - Multi-stage blocking with open-set country awareness.
#        - Evaluates candidate recall and reduction ratio.
#     4. Pairwise Feature Engineering:
#        - Computes name, address, country, cross-field, and candidate metadata features.
#     5. Model Training & Optimization:
#        - Trains Gradient-Boosted Decision Trees (LightGBM / HistGradientBoosting).
#        - Evaluates probability calibration and feature importances.
#     6. Precision-Heavy Decision Layer:
#        - Fine grid search over decision thresholds against macro F0.5.
#        - Freezes optimal threshold before test inference.
#     7. Test Set Prediction & Output Generation:
#        - Generates output/candidate_pairs.tsv.
#        - Generates output/matching_results.tsv.
#        - Executes submission validator to verify 100% compliance.
# ==============================================================================

# Standard library imports
import os
import sys
import random
import subprocess
import collections
from typing import Dict, List, Set, Tuple, Any, Optional

# Third-party data manipulation library
import pandas as pd
import numpy as np

# Internal module imports
from src.normalization import NormalizedRecord
from src.blocking import MultiStageBlocker, evaluate_blocking_quality, export_candidate_pairs_tsv
from src.features import FeatureExtractor, FEATURE_NAMES
from src.matching import EntityMatchingModel
from src.evaluate import evaluate_macro_f05, find_optimal_decision_threshold


# ==============================================================================
# 1. DATA LOADING UTILITIES
# ==============================================================================

def load_source_records(tsv_path: str) -> Dict[str, NormalizedRecord]:
    """
    Loads a source TSV file and converts each row into a NormalizedRecord.
    
    Args:
        tsv_path (str): Filepath to the source TSV (*_source1.tsv, *_source2.tsv, or *_source3.tsv).
        
    Returns:
        Dict[str, NormalizedRecord]: Dictionary mapping entity_id to NormalizedRecord.
    """
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"Source file not found at: {tsv_path}")

    # Explicit tab separator is essential: address text contains commas!
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

    records: Dict[str, NormalizedRecord] = {}
    for _, row in df.iterrows():
        eid = str(row.get("entity_id", "")).strip()
        b_name = str(row.get("business_name", ""))
        b_addr = str(row.get("business_address", ""))
        country = str(row.get("country", ""))

        if eid:
            records[eid] = NormalizedRecord(
                entity_id=eid,
                business_name=b_name,
                business_address=b_addr,
                country=country
            )
            
    return records


def load_ground_truth(ground_truth_path: str) -> Dict[str, List[str]]:
    """
    Loads ground truth associations from train_ground_truth.tsv.
    
    Args:
        ground_truth_path (str): Path to train_ground_truth.tsv.
        
    Returns:
        Dict[str, List[str]]: Mapping from source1_entity_id to list of matching target IDs.
    """
    if not os.path.isfile(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_path}")

    df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)

    ground_truth: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row.get("source1_entity_id", "")).strip()
        raw_matches = str(row.get("matched_entity_ids", "")).strip()
        
        if not s1_id:
            continue
            
        if raw_matches:
            matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
            ground_truth[s1_id] = matches
        else:
            # Singleton entity (zero true matches)
            ground_truth[s1_id] = []
            
    return ground_truth


# ==============================================================================
# 2. DATASET SPLITTING & FEATURE MATRIX GENERATION
# ==============================================================================

def split_s1_entities(
    all_s1_ids: List[str],
    val_fraction: float = 0.20,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits Source 1 entity IDs into disjoint train and validation sets.
    Ensures zero leakage: validation entities are never seen during training.
    
    Args:
        all_s1_ids (List[str]): List of all available Source 1 entity IDs.
        val_fraction (float): Fraction of entities held out for validation.
        random_seed (int): Seed for deterministic splitting.
        
    Returns:
        Tuple[List[str], List[str]]: (train_s1_ids, val_s1_ids)
    """
    # Deterministic shuffle
    rng = random.Random(random_seed)
    shuffled_ids = list(all_s1_ids)
    rng.shuffle(shuffled_ids)

    split_idx = int(len(shuffled_ids) * (1.0 - val_fraction))
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]
    
    return train_ids, val_ids


def build_pairwise_dataset(
    s1_ids: List[str],
    s1_records: Dict[str, NormalizedRecord],
    blocker: MultiStageBlocker,
    target_records: Dict[str, NormalizedRecord],
    ground_truth: Optional[Dict[str, List[str]]] = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, List[str]]]:
    """
    Generates candidates via blocking, extracts pairwise feature vectors, and
    assigns binary labels (if ground truth is supplied).
    
    Args:
        s1_ids (List[str]): List of Source 1 IDs to process.
        s1_records (Dict[str, NormalizedRecord]): Normalized Source 1 records.
        blocker (MultiStageBlocker): Initialized candidate generator.
        target_records (Dict[str, NormalizedRecord]): Normalized target (S2/S3) records.
        ground_truth (Dict[str, List[str]], optional): Ground truth mapping if training.
        
    Returns:
        Tuple:
            - X: np.ndarray feature matrix (num_pairs, num_features)
            - y: np.ndarray binary labels (num_pairs,) or empty array
            - pair_index: List of (s1_id, target_id) corresponding to rows in X
            - candidates_dict: Mapping of s1_id -> candidate target IDs list
    """
    feature_rows: List[List[float]] = []
    labels: List[int] = []
    pair_index: List[Tuple[str, str]] = []
    candidates_dict: Dict[str, List[str]] = {}

    for s1_id in s1_ids:
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            candidates_dict[s1_id] = []
            continue

        # Generate candidates using multi-stage blocker
        candidates, cands_meta = blocker.generate_candidates_for_record(s1_rec)
        candidates_dict[s1_id] = candidates

        # True matches for this entity (if ground truth provided)
        true_set = set(ground_truth.get(s1_id, [])) if ground_truth is not None else set()

        # Extract features for each candidate pair
        for rank_idx, target_id in enumerate(candidates):
            target_rec = target_records.get(target_id)
            if not target_rec:
                continue

            meta = cands_meta.get(target_id)
            feat_vec = FeatureExtractor.extract_pairwise_features(
                s1=s1_rec,
                target=target_rec,
                candidate_meta=meta,
                candidate_rank=rank_idx
            )

            feature_rows.append(feat_vec)
            pair_index.append((s1_id, target_id))

            if ground_truth is not None:
                # Binary label: 1 if this target is in true matches, else 0
                label = 1 if target_id in true_set else 0
                labels.append(label)

    # Convert to NumPy arrays
    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if labels else np.empty((0,), dtype=np.int32)

    return X, y, pair_index, candidates_dict


# ==============================================================================
# 3. TRAINING & VALIDATION PIPELINE
# ==============================================================================

def train_and_validate(
    data_dir: str = "dataset/train",
    model_save_path: str = "models/matching_model.pkl",
    model_type: str = "lightgbm",
    val_fraction: float = 0.20
) -> Tuple[EntityMatchingModel, float, Dict[str, Any]]:
    """
    Executes complete training, candidate generation audit, feature extraction,
    model fitting, and threshold tuning on held-out validation data.
    
    Args:
        data_dir (str): Directory containing train_source{1,2,3}.tsv and train_ground_truth.tsv.
        model_save_path (str): Destination path for saving trained model checkpoint.
        model_type (str): Type of model ('lightgbm', 'hist_gbdt', 'logistic').
        val_fraction (float): Held-out fraction of Source 1 entities for validation.
        
    Returns:
        Tuple:
            - fitted_model: Trained EntityMatchingModel instance.
            - optimal_threshold: Tuned probability threshold maximizing macro F0.5.
            - validation_report: Dictionary of validation metrics and ablation stats.
    """
    print("\n" + "=" * 75)
    print(" Amazon ML Challenge - Training & Validation Pipeline")
    print("=" * 75)

    # Step 1: Load all training source datasets
    print("\n[Step 1/6] Loading and Normalizing Training Sources...")
    s1_records = load_source_records(os.path.join(data_dir, "train_source1.tsv"))
    s2_records = load_source_records(os.path.join(data_dir, "train_source2.tsv"))
    s3_records = load_source_records(os.path.join(data_dir, "train_source3.tsv"))
    ground_truth = load_ground_truth(os.path.join(data_dir, "train_ground_truth.tsv"))

    # Combine S2 and S3 into target entity pool
    target_records = {**s2_records, **s3_records}
    print(f" Loaded {len(s1_records):,} Source 1 records.")
    print(f" Loaded {len(s2_records):,} Source 2 and {len(s3_records):,} Source 3 records "
          f"(Total Target Pool: {len(target_records):,}).")
    print(f" Loaded {len(ground_truth):,} Ground Truth associations.")

    # Step 2: Leakage-safe Train/Val Split
    print("\n[Step 2/6] Performing Leakage-Safe Entity-Level Validation Split...")
    all_s1_ids = list(s1_records.keys())
    train_s1_ids, val_s1_ids = split_s1_entities(all_s1_ids, val_fraction=val_fraction, random_seed=42)
    print(f" Train Split: {len(train_s1_ids):,} entities ({(1-val_fraction)*100:.0f}%)")
    print(f" Val Split  : {len(val_s1_ids):,} entities ({val_fraction*100:.0f}%)")

    # Step 3: Build Inverted Indexes & Multi-Stage Blocker
    print("\n[Step 3/6] Initializing Candidate Blocker & Inverted Indexes...")
    blocker = MultiStageBlocker(target_records=target_records, max_candidates_per_source=60)

    # Step 4: Generate Training & Validation Features
    print("\n[Step 4/6] Generating Candidates and Extracting Pairwise Features...")
    print("   * Building training feature matrix...")
    X_train, y_train, _, train_candidates = build_pairwise_dataset(
        s1_ids=train_s1_ids,
        s1_records=s1_records,
        blocker=blocker,
        target_records=target_records,
        ground_truth=ground_truth
    )
    print(f"   * Training Pairs: {len(X_train):,} (Matches: {np.sum(y_train):,}, Non-Matches: {len(y_train)-np.sum(y_train):,})")

    print("   * Building validation feature matrix...")
    X_val, y_val, val_pair_index, val_candidates = build_pairwise_dataset(
        s1_ids=val_s1_ids,
        s1_records=s1_records,
        blocker=blocker,
        target_records=target_records,
        ground_truth=ground_truth
    )
    print(f"   * Validation Pairs: {len(X_val):,}")

    # Evaluate blocking quality on validation set
    val_gt = {eid: ground_truth.get(eid, []) for eid in val_s1_ids}
    blocking_metrics = evaluate_blocking_quality(val_candidates, val_gt, len(target_records))
    print("\n   --- Validation Candidate Blocking Metrics ---")
    print(f"   * Candidate Recall   : {blocking_metrics['candidate_recall']*100:.2f}% "
          f"({blocking_metrics['captured_matches']}/{blocking_metrics['total_true_matches']} matches)")
    print(f"   * Avg Candidates / S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")
    print(f"   * Reduction Ratio    : {blocking_metrics['reduction_ratio']*100:.4f}%")

    # Step 5: Fit Machine Learning Model
    print(f"\n[Step 5/6] Training Pairwise Matching Model ({model_type.upper()})...")
    model = EntityMatchingModel(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    # Log feature importances
    importances = model.get_feature_importances()
    if importances:
        print("   --- Top 10 Most Predictive Features ---")
        top_feats = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for idx, (f_name, imp) in enumerate(top_feats, start=1):
            print(f"   {idx:2d}. {f_name:35s}: {imp:5.2f}%")

    # Step 6: Tune Threshold for Precision-Heavy F0.5
    print("\n[Step 6/6] Tuning Decision Threshold for Macro F0.5 on Validation Set...")
    # Predict probabilities on validation candidate pairs
    val_probs = model.predict_proba(X_val)

    # Reconstruct validation scores per S1 entity
    val_scores_per_s1: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (s1_id, target_id), prob in zip(val_pair_index, val_probs):
        val_scores_per_s1[s1_id].append((target_id, float(prob)))

    # Ensure all validation S1 IDs have an entry (even if blocker returned 0 candidates)
    for s1_id in val_s1_ids:
        if s1_id not in val_scores_per_s1:
            val_scores_per_s1[s1_id] = []

    best_threshold, best_f05, grid_results = find_optimal_decision_threshold(
        val_candidate_scores=val_scores_per_s1,
        val_ground_truth=val_gt
    )

    best_metrics = grid_results[best_threshold]
    print(f"\n   >>> OPTIMAL VALIDATION RESULTS <<<")
    print(f"   * Best Probability Threshold : {best_threshold:.2f}")
    print(f"   * Validation Macro F0.5      : {best_metrics['macro_f05']:.4f}")
    print(f"   * Validation Macro Precision : {best_metrics['macro_precision']:.4f}")
    print(f"   * Validation Macro Recall    : {best_metrics['macro_recall']:.4f}")
    print(f"   * Singleton F0.5 Score       : {best_metrics['singleton_f05']:.4f} "
          f"({best_metrics['singleton_count']} singletons)")
    print(f"   * Non-Singleton F0.5 Score   : {best_metrics['non_singleton_f05']:.4f} "
          f"({best_metrics['non_singleton_count']} non-singletons)")

    # Freeze threshold in model
    model.optimal_threshold = best_threshold

    # Save model checkpoint
    model.save(model_save_path)
    print(f"\n Saved trained model checkpoint to: {model_save_path}")
    print("=" * 75)

    report = {
        "model_type": model.model_type,
        "optimal_threshold": best_threshold,
        "best_metrics": best_metrics,
        "blocking_metrics": blocking_metrics,
        "grid_results": grid_results
    }
    return model, best_threshold, report


# ==============================================================================
# 4. TEST INFERENCE & OFFICIAL SUBMISSION GENERATION
# ==============================================================================

def predict_test_dataset(
    test_dir: str = "dataset/test",
    model_path: str = "models/matching_model.pkl",
    output_dir: str = "output",
    decision_threshold: Optional[float] = None
) -> Tuple[str, str]:
    """
    Executes inference over the test dataset, generating:
      1. output/candidate_pairs.tsv (blocking candidates fed to model)
      2. output/matching_results.tsv (final entity match predictions)
    And automatically executes utils/validate_submission.py to ensure 100% PASS.
    
    Args:
        test_dir (str): Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv.
        model_path (str): Path to saved trained model checkpoint.
        output_dir (str): Output folder for submission files.
        decision_threshold (float, optional): Custom threshold; defaults to model's optimal_threshold.
        
    Returns:
        Tuple[str, str]: Paths to (matching_results.tsv, candidate_pairs.tsv).
    """
    print("\n" + "=" * 75)
    print(" Amazon ML Challenge - Test Inference & Submission Generation")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    matching_output_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_output_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Step 1: Load trained model checkpoint
    print(f"\n[1/5] Loading Model from: {model_path}")
    model = EntityMatchingModel.load(model_path)
    threshold = decision_threshold if decision_threshold is not None else model.optimal_threshold
    print(f" Loaded {model.model_type.upper()} model. Applied Decision Threshold: {threshold:.2f}")

    # Step 2: Load and normalize test sources
    print("\n[2/5] Loading Test Datasets...")
    test_s1 = load_source_records(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = load_source_records(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = load_source_records(os.path.join(test_dir, "test_source3.tsv"))
    test_targets = {**test_s2, **test_s3}
    print(f" Loaded {len(test_s1):,} Source 1 test entities.")
    print(f" Loaded {len(test_targets):,} Target test entities ({len(test_s2)} S2 + {len(test_s3)} S3).")

    # Step 3: Candidate Generation (Blocking)
    print("\n[3/5] Running Multi-Stage Blocking on Test Set...")
    blocker = MultiStageBlocker(target_records=test_targets, max_candidates_per_source=60)
    all_test_s1_ids = list(test_s1.keys())

    X_test, _, test_pair_index, test_candidates = build_pairwise_dataset(
        s1_ids=all_test_s1_ids,
        s1_records=test_s1,
        blocker=blocker,
        target_records=test_targets,
        ground_truth=None
    )
    print(f" Generated {len(X_test):,} candidate pairs across {len(all_test_s1_ids):,} entities.")

    # Export candidate_pairs.tsv immediately
    export_candidate_pairs_tsv(test_candidates, candidate_output_path, all_test_s1_ids)
    print(f" Successfully exported: {candidate_output_path}")

    # Step 4: Model Inference & Precision Thresholding
    print("\n[4/5] Scoring Candidate Pairs with ML Classifier...")
    if len(X_test) > 0:
        test_probs = model.predict_proba(X_test)
    else:
        test_probs = np.array([])

    # Group predictions by Source 1 entity ID
    matches_dict: Dict[str, List[str]] = {eid: [] for eid in all_test_s1_ids}
    for (s1_id, target_id), prob in zip(test_pair_index, test_probs):
        if prob >= threshold:
            matches_dict[s1_id].append(target_id)

    # Export matching_results.tsv
    with open(matching_output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            matches = matches_dict.get(s1_id, [])
            matches_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{matches_str}\n")

    print(f" Successfully exported: {matching_output_path}")

    # Statistics on test predictions
    predicted_singletons = sum(1 for m in matches_dict.values() if len(m) == 0)
    predicted_multi = sum(1 for m in matches_dict.values() if len(m) > 1)
    total_predicted_links = sum(len(m) for m in matches_dict.values())
    print("\n   --- Test Predictions Overview ---")
    print(f"   * Total Source 1 Entities : {len(all_test_s1_ids):,}")
    print(f"   * Total Matches Predicted : {total_predicted_links:,}")
    print(f"   * Predicted Singletons    : {predicted_singletons:,} ({(predicted_singletons/len(all_test_s1_ids))*100:.1f}%)")
    print(f"   * Predicted Multi-Matches : {predicted_multi:,}")

    # Step 5: Validate Submission Files
    print("\n[5/5] Running Submission Validator...")
    validator_cmd = [
        sys.executable,
        os.path.join("utils", "validate_submission.py"),
        "--matching", matching_output_path,
        "--candidate", candidate_output_path,
        "--test-dir", test_dir
    ]
    result = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR] Submission validation failed! Output:")
        print(result.stderr)
        sys.exit(1)
    else:
        print("[SUCCESS] Submission Validator returned PASS (Exit Code 0)!")

    print("=" * 75)
    return matching_output_path, candidate_output_path
