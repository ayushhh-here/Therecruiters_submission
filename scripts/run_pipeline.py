#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Script: scripts/run_pipeline.py
#
# Description:
#   Master Command-Line Interface (CLI) for executing the complete pipeline.
#   Supports flexible execution modes:
#     - 'audit'   : Executes Exploratory Data Analysis & open-set integrity audit.
#     - 'train'   : Trains LightGBM/GBDT matching model and tunes threshold on validation set.
#     - 'predict' : Runs test inference, generates output TSVs, and runs validator.
#     - 'all'     : Runs audit, training, threshold tuning, and test prediction sequentially.
#
# Usage Examples:
#   python scripts/run_pipeline.py --mode all
#   python scripts/run_pipeline.py --mode train --model lightgbm
#   python scripts/run_pipeline.py --mode predict --threshold 0.65
# ==============================================================================

# Standard library imports
import os
import sys
import argparse

# Add workspace root to sys.path so 'src' can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Internal pipeline imports
from src.eda import run_full_data_audit
from src.pipeline import train_and_validate, predict_test_dataset


def parse_args():
    """
    Parses command line arguments.
    
    Returns:
        argparse.Namespace: CLI argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge: End-to-End Business Entity Resolution Pipeline"
    )

    # Execution Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["audit", "train", "predict", "all"],
        help="Pipeline phase to execute: 'audit', 'train', 'predict', or 'all' (default: all)"
    )

    # Dataset directories
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Root folder containing 'train/' and 'test/' directories (default: dataset)"
    )

    # Model architecture selection
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "hist_gbdt", "logistic"],
        help="Machine learning classifier type (default: lightgbm)"
    )

    # Model checkpoint location
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/matching_model.pkl",
        help="Path to save or load model checkpoint (default: models/matching_model.pkl)"
    )

    # Custom probability threshold override
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom probability threshold for prediction. If omitted, uses validation-tuned optimal threshold."
    )

    # Output directory for submission TSVs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output folder for matching_results.tsv and candidate_pairs.tsv (default: output)"
    )

    return parser.parse_args()


def main():
    """
    Main driver orchestrating the chosen pipeline phase.
    """
    args = parse_args()
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")

    print("\n" + "#" * 75)
    print(" AMAZON ML CHALLENGE - BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f" Mode: {args.mode.upper()} | Model: {args.model.upper()} | Data Root: {args.data_root}")
    print("#" * 75)

    # Mode 1: Audit
    if args.mode in ["audit", "all"]:
        print("\n>>> PHASE 1: DATA AUDIT & EDA <<<")
        run_full_data_audit(data_root=args.data_root)

    # Mode 2: Train & Validate
    if args.mode in ["train", "all"]:
        print("\n>>> PHASE 2: MODEL TRAINING & VALIDATION THRESHOLD TUNING <<<")
        train_and_validate(
            data_dir=train_dir,
            model_save_path=args.model_path,
            model_type=args.model,
            val_fraction=0.20
        )

    # Mode 3: Test Prediction & Submission Generation
    if args.mode in ["predict", "all"]:
        print("\n>>> PHASE 3: TEST INFERENCE & OFFICIAL SUBMISSION VALIDATION <<<")
        predict_test_dataset(
            test_dir=test_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            decision_threshold=args.threshold
        )

    print("\n" + "#" * 75)
    print(" Pipeline Execution Finished Successfully!")
    print("#" * 75)


if __name__ == "__main__":
    main()
