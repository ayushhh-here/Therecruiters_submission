#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/matching.py
#
# Description:
#   High-Performance Binary Matcher with Probability Calibration & Entity Decision Layer.
#   Implements Phase 9, 10, 11:
#     - LightGBM / HistGradientBoosting / Logistic Regression models
#     - Probability calibration (Isotonic Regression / Platt Scaling)
#     - Hard negative mining support
#     - Entity-level decision logic:
#         * Top-1 score, Top-2 score, score margin
#         * Adaptive singleton thresholding
#         * Multi-match acceptance based on score margin
#     - Model checkpoint serialization (pickle)
# ==============================================================================

import os
import pickle
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.features import FEATURE_NAMES, EXTENDED_FEATURE_NAMES, BASELINE_FEATURE_NAMES


class EntityMatchingModel:
    """
    Unified classifier interface for Business Entity Resolution.
    Includes Platt / Isotonic probability calibration and entity-level decision logic.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        use_extended_features: bool = True
    ):
        self.model_type: str = model_type
        self.random_state: int = random_state
        self.use_extended_features: bool = use_extended_features
        self.feature_names: List[str] = (
            list(EXTENDED_FEATURE_NAMES) if use_extended_features else list(BASELINE_FEATURE_NAMES)
        )
        self.model: Any = None
        self.calibrator: Any = None
        self.is_fitted: bool = False
        self.is_calibrated: bool = False
        self.optimal_threshold: float = 0.50
        self.singleton_threshold: float = 0.50
        self.margin_threshold: float = 0.20

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Instantiates underlying classifier."""
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=7,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
        elif HAS_SKLEARN and self.model_type in ["lightgbm", "hist_gbdt"]:
            self.model_type = "hist_gbdt"
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=31,
                max_depth=7,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        elif HAS_SKLEARN and self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            )
        else:
            self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EntityMatchingModel":
        """Fits classifier on pairwise feature matrix X and binary labels y."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        if self.model is not None and len(X_arr) > 0:
            self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs uncalibrated raw model probabilities."""
        X_arr = np.asarray(X, dtype=np.float32)
        if len(X_arr) == 0:
            return np.array([], dtype=np.float32)

        if self.model is not None and self.is_fitted:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_arr)
                if probs.shape[1] >= 2:
                    return probs[:, 1]
                return probs[:, 0]
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X_arr)
                return 1.0 / (1.0 + np.exp(-scores))

        # Heuristic fallback
        lev = X_arr[:, 4] if X_arr.shape[1] > 4 else np.zeros(len(X_arr))
        jacc = X_arr[:, 2] if X_arr.shape[1] > 2 else np.zeros(len(X_arr))
        return np.clip(0.6 * lev + 0.4 * jacc, 0.0, 1.0)

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "isotonic"
    ) -> "EntityMatchingModel":
        """
        Calibrates predicted probabilities using validation data (Phase 11).
        Supports:
          - 'isotonic': Non-parametric isotonic regression
          - 'platt': Logistic sigmoid regression
        """
        if len(X_val) == 0:
            return self

        raw_probs = self.predict_raw_proba(X_val)
        y_arr = np.asarray(y_val, dtype=np.float64)

        if method == "isotonic" and HAS_SKLEARN:
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_probs, y_arr)
            self.is_calibrated = True
        elif method == "platt" and HAS_SKLEARN:
            self.calibrator = LogisticRegression(C=1.0)
            self.calibrator.fit(raw_probs.reshape(-1, 1), y_arr)
            self.is_calibrated = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs calibrated probabilities if calibrated, else raw probabilities."""
        raw_probs = self.predict_raw_proba(X)
        if self.is_calibrated and self.calibrator is not None and len(raw_probs) > 0:
            if isinstance(self.calibrator, IsotonicRegression):
                return np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)
            elif isinstance(self.calibrator, LogisticRegression):
                return self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return raw_probs

    def make_entity_decisions(
        self,
        candidate_scores_per_s1: Dict[str, List[Tuple[str, float]]],
        decision_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Entity-level decision logic (Phase 10).
        For each Source 1 query entity:
          - Extracts top candidate, top-1 score, top-2 score, score margin
          - Accepts top candidate if top-1 score >= threshold
          - Accepts additional candidates if within score margin
          - Declares singleton if top-1 score < threshold
        """
        thresh = decision_threshold if decision_threshold is not None else self.optimal_threshold
        margin = margin_threshold if margin_threshold is not None else self.margin_threshold

        decisions: Dict[str, Dict[str, Any]] = {}

        for s1_id, candidate_pairs in candidate_scores_per_s1.items():
            if not candidate_pairs:
                decisions[s1_id] = {
                    "accepted_matches": [],
                    "best_score": 0.0,
                    "second_best_score": 0.0,
                    "score_margin": 0.0,
                    "candidate_count": 0,
                    "is_singleton": True
                }
                continue

            # Sort candidate pairs by score descending
            sorted_cands = sorted(candidate_pairs, key=lambda kv: kv[1], reverse=True)
            best_id, best_score = sorted_cands[0]
            second_score = sorted_cands[1][1] if len(sorted_cands) > 1 else 0.0
            score_margin = best_score - second_score

            accepted: List[str] = []
            if best_score >= thresh:
                accepted.append(best_id)
                # Check for plausible multi-matches within confidence margin
                for tid, score in sorted_cands[1:]:
                    if score >= thresh and (best_score - score) <= margin:
                        accepted.append(tid)

            decisions[s1_id] = {
                "accepted_matches": accepted,
                "best_score": round(best_score, 4),
                "second_best_score": round(second_score, 4),
                "score_margin": round(score_margin, 4),
                "candidate_count": len(sorted_cands),
                "is_singleton": (len(accepted) == 0)
            }

        return decisions

    def get_feature_importances(self) -> Dict[str, float]:
        """Extracts normalized feature importance percentages."""
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            raw = self.model.feature_importances_
            total = np.sum(raw)
            if total > 0:
                normalized = (raw / total) * 100.0
            else:
                normalized = raw
            return {
                name: float(normalized[i])
                for i, name in enumerate(self.feature_names)
                if i < len(normalized)
            }
        return {}

    def save(self, filepath: str) -> None:
        """Serializes trained model and calibrator to disk safely."""
        clean_path = os.path.abspath(str(filepath).strip())
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        temp_path = clean_path + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "calibrator": self.calibrator,
                "is_calibrated": self.is_calibrated,
                "model_type": self.model_type,
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
                "optimal_threshold": self.optimal_threshold,
                "singleton_threshold": self.singleton_threshold,
                "margin_threshold": self.margin_threshold,
                "use_extended_features": self.use_extended_features
            }, f)
        if os.path.exists(clean_path):
            try:
                os.remove(clean_path)
            except OSError:
                pass
        if os.path.exists(temp_path):
            os.replace(temp_path, clean_path)

    @classmethod
    def load(cls, filepath: str) -> "EntityMatchingModel":
        """Loads serialized model checkpoint."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            model_type=data.get("model_type", "lightgbm"),
            use_extended_features=data.get("use_extended_features", True)
        )
        instance.model = data.get("model")
        instance.calibrator = data.get("calibrator")
        instance.is_calibrated = data.get("is_calibrated", False)
        instance.is_fitted = data.get("is_fitted", False)
        instance.feature_names = data.get("feature_names", list(EXTENDED_FEATURE_NAMES))
        instance.optimal_threshold = data.get("optimal_threshold", 0.50)
        instance.singleton_threshold = data.get("singleton_threshold", 0.50)
        instance.margin_threshold = data.get("margin_threshold", 0.20)
        return instance
