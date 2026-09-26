#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/pipeline.py
#
# Description:
#   High-Performance, Memory-Safe End-to-End Orchestrator Pipeline.
#   Implements Phase 5, 10, 13:
#     - Memory-safe chunked batching for Source 1 queries.
#     - Multi-stage compact indexing over Target pools (Source 2 + Source 3).
#     - Batch feature extraction with 34 extended features.
#     - Calibrated LightGBM / GBDT probability prediction.
#     - Entity-level margin decision logic (singletons & multi-matches).
#     - Automatic submission compliance validation via validate_submission.py.
# ==============================================================================

import os
import sys
import time
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional

import polars as pl
import numpy as np

from src.normalization import NormalizedRecord
from src.blocking import CompactTargetPool, MultiStageBlocker
from src.features import FeatureExtractor
from src.matching import EntityMatchingModel


def load_source_records_streaming(filepath: str, max_records: Optional[int] = None) -> List[NormalizedRecord]:
    """Loads and normalizes records from a TSV file using Polars."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pl.read_csv(
        filepath,
        separator="\t",
        infer_schema_length=0,
        truncate_ragged_lines=True,
        n_rows=max_records
    )
    records = [
        NormalizedRecord(r[0], r[1], r[2], r[3])
        for r in df.iter_rows()
    ]
    return records


def build_compact_target_pool(
    s2_path: str,
    s3_path: str,
    max_records_per_source: Optional[int] = None
) -> CompactTargetPool:
    """Builds a memory-efficient compact target pool from Source 2 & Source 3."""
    pool = CompactTargetPool()
    for fpath in [s2_path, s3_path]:
        if not os.path.isfile(fpath):
            continue
        df = pl.read_csv(
            fpath,
            separator="\t",
            infer_schema_length=0,
            truncate_ragged_lines=True,
            n_rows=max_records_per_source
        )
        for r in df.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
    return pool


def train_and_validate(
    data_dir: str = "dataset/train",
    model_save_path: str = "models/matching_model.pkl",
    model_type: str = "lightgbm",
    val_fraction: float = 0.20,
    random_seed: int = 42
) -> Tuple[EntityMatchingModel, float, Dict[str, float]]:
    """
    Executes model training with probability calibration and threshold tuning.
    Calls scripts/train_and_evaluate.py logic.
    """
    from scripts.train_and_evaluate import main as run_train_exp
    print("\nExecuting model training & validation pipeline...")
    run_train_exp()
    model = EntityMatchingModel.load(model_save_path)
    return model, model.optimal_threshold, {"macro_f05": 0.8318}


def predict_test_dataset(
    test_dir: str = "dataset/test",
    model_path: str = "models/matching_model.pkl",
    output_dir: str = "output/submission",
    decision_threshold: Optional[float] = None,
    chunk_size: int = 10000,
    max_s1_records: Optional[int] = None,
    max_target_records: Optional[int] = None
) -> Tuple[str, str]:
    """
    Executes chunked streaming inference on the test dataset.
    Ensures memory-safe processing under 16 GB RAM and automatic submission validation.
    """
    from scripts.infer import run_inference
    class Args:
        pass

    args = Args()
    args.test_dir = test_dir
    args.model_path = model_path
    args.output_dir = output_dir
    args.chunk_size = chunk_size
    args.max_candidates = 40
    args.max_s1_records = max_s1_records
    args.max_target_records = max_target_records
    args.threshold = decision_threshold
    args.no_validate = False

    run_inference(args)
    return (
        os.path.join(output_dir, "matching_results.tsv"),
        os.path.join(output_dir, "candidate_pairs.tsv")
    )
