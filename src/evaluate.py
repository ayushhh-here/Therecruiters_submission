#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/evaluate.py
#
# Description:
#   Official Challenge Metric Implementation & Validation Engine.
#   Implements the exact macro-averaged entity-level F0.5 metric specified in the
#   challenge rules:
#     - Formula:
#         F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
#     - Precision-heavy weighting: False merges (false positives) penalize the
#       score twice as heavily as missed matches (false negatives).
#     - Singletons (entities with 0 true matches):
#         * Predict empty list [] -> score = 1.0
#         * Predict any match [S2-xxx] -> score = 0.0 (heavily penalized!)
#     - Multi-match entities (entities with >= 1 true match):
#         * TP = |Predicted ∩ GroundTruth|
#         * Precision = TP / |Predicted| (if |Predicted| > 0 else 0.0)
#         * Recall = TP / |GroundTruth|
#         * If Precision + Recall == 0: score = 0.0
#         * Else: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
#     - Macro-average: Sum of entity F0.5 scores divided by total Source 1 entities.
# ==============================================================================

# Standard library imports
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np


def compute_entity_f05(
    predicted_ids: Set[str],
    ground_truth_ids: Set[str]
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F0.5 score for a single Source 1 entity.
    
    Args:
        predicted_ids (Set[str]): Set of predicted target entity IDs (S2/S3).
        ground_truth_ids (Set[str]): Set of true matching target entity IDs.
        
    Returns:
        Tuple[float, float, float]: (precision, recall, f05_score)
    """
    is_singleton = len(ground_truth_ids) == 0

    # Case 1: Singleton entity (has no true matches in Source 2 or 3)
    if is_singleton:
        if len(predicted_ids) == 0:
            # Correctly predicted no match -> earns full 1.0 credit
            return 1.0, 1.0, 1.0
        else:
            # False merge on a singleton -> 0.0 credit
            return 0.0, 0.0, 0.0

    # Case 2: Entity with true matches, but model predicted nothing
    if len(predicted_ids) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Entity with true matches and model predicted >= 1 match
    true_positives = len(predicted_ids & ground_truth_ids)
    
    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(ground_truth_ids)

    # If denominator would be zero, return 0.0
    denominator = (0.25 * precision) + recall
    if denominator <= 1e-9:
        return precision, recall, 0.0

    f05 = (1.25 * precision * recall) / denominator
    return precision, recall, f05


def evaluate_macro_f05(
    predictions_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Computes macro-averaged F0.5, Precision, and Recall across all Source 1 entities.
    Also tracks singleton accuracy and multi-match performance.
    
    Args:
        predictions_dict (Dict[str, List[str]]):
            Predicted matches {s1_id: [s2_or_s3_ids]}.
        ground_truth_dict (Dict[str, List[str]]):
            Ground truth matches {s1_id: [s2_or_s3_ids]}.
            
    Returns:
        Dict[str, float]: Comprehensive evaluation dictionary containing:
            - 'macro_f05': Overall macro F0.5 score
            - 'macro_precision': Macro Precision
            - 'macro_recall': Macro Recall
            - 'singleton_f05': Average F0.5 on singletons
            - 'non_singleton_f05': Average F0.5 on non-singletons
            - 'singleton_count': Total number of singletons
            - 'non_singleton_count': Total number of non-singletons
    """
    total_entities = len(ground_truth_dict)
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_f05": 0.0,
            "non_singleton_f05": 0.0
        }

    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0

    singleton_scores = []
    non_singleton_scores = []

    for s1_id, gt_list in ground_truth_dict.items():
        gt_set = set(gt_list)
        pred_list = predictions_dict.get(s1_id, [])
        pred_set = set(pred_list)

        p, r, f = compute_entity_f05(pred_set, gt_set)

        sum_f05 += f
        sum_prec += p
        sum_rec += r

        if len(gt_set) == 0:
            singleton_scores.append(f)
        else:
            non_singleton_scores.append(f)

    return {
        "macro_f05": sum_f05 / total_entities,
        "macro_precision": sum_prec / total_entities,
        "macro_recall": sum_rec / total_entities,
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_scores)
    }


def find_optimal_decision_threshold(
    val_candidate_scores: Dict[str, List[Tuple[str, float]]],
    val_ground_truth: Dict[str, List[str]],
    threshold_grid: Optional[List[float]] = None
) -> Tuple[float, float, Dict[float, Dict[str, float]]]:
    """
    Searches a grid of probability thresholds on validation data to maximize macro F0.5.
    
    Args:
        val_candidate_scores (Dict[str, List[Tuple[str, float]]]):
            Mapping of s1_id -> list of (target_id, predicted_match_probability).
        val_ground_truth (Dict[str, List[str]]):
            Ground truth matches {s1_id: [true_targets]}.
        threshold_grid (List[float], optional):
            Grid of threshold values to test (defaults to 0.40 through 0.95 in 0.02 steps).
            
    Returns:
        Tuple[float, float, Dict[float, Dict[str, float]]]:
            - best_threshold: Threshold achieving highest macro F0.5.
            - best_macro_f05: The peak validation macro F0.5 score.
            - detailed_grid_results: Mapping of threshold -> metrics dictionary.
    """
    if threshold_grid is None:
        # Fine grid from 0.10 to 0.96 with 0.02 increments
        threshold_grid = [round(t, 2) for t in np.arange(0.10, 0.96, 0.02)]

    best_threshold = 0.50
    best_f05 = -1.0
    grid_results = {}

    for thresh in threshold_grid:
        # Construct thresholded predictions for this candidate pool
        preds_at_thresh = {}
        for s1_id, candidate_pairs in val_candidate_scores.items():
            accepted_targets = []
            for target_id, prob in candidate_pairs:
                if prob >= thresh:
                    accepted_targets.append(target_id)
            preds_at_thresh[s1_id] = accepted_targets

        metrics = evaluate_macro_f05(preds_at_thresh, val_ground_truth)
        grid_results[thresh] = metrics

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = thresh

    return best_threshold, best_f05, grid_results
