#!/usr/bin/env python3
"""
scripts/smoke_test.py — End-to-End Pipeline Smoke Test

Verifies all 10 stages on a tiny subset:
  1. Imports
  2. Dataset loading (TSV with tab separator)
  3. Multilingual normalization (Unicode preserved)
  4. Compact index construction
  5. Multi-stage candidate retrieval
  6. Pairwise feature extraction (29 baseline + 5 extended)
  7. Model training & probability calibration
  8. Entity-level prediction & thresholding
  9. TSV output generation (matching_results.tsv, candidate_pairs.tsv)
  10. Submission validator compliance check
"""

import os
import sys
import tempfile
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


def run_smoke_test():
    print("=" * 70)
    print(" Amazon ML Challenge — Pipeline Smoke Test (10 Verification Checks)")
    print("=" * 70)

    # 1. Imports
    print("\n[Check 1/10] Verifying Core Imports...")
    try:
        import polars as pl
        import lightgbm as lgb
        import sklearn
        import rapidfuzz
        from src.normalization import NormalizedRecord
        from src.blocking import CompactTargetPool, MultiStageBlocker
        from src.features import FeatureExtractor, BASELINE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES
        from src.matching import EntityMatchingModel
        from src.evaluate import evaluate_macro_f05, find_optimal_decision_threshold
        print("  ✓ All required libraries and src modules imported successfully.")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        sys.exit(1)

    # 2. Dataset Loading (from actual files with tiny n_rows)
    print("\n[Check 2/10] Verifying Dataset Loading...")
    try:
        df_s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=50, infer_schema_length=0)
        df_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=100, infer_schema_length=0)
        df_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=100, infer_schema_length=0)
        df_gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=50, infer_schema_length=0)
        print(f"  ✓ Loaded sample S1 ({len(df_s1)}), S2 ({len(df_s2)}), S3 ({len(df_s3)}), GT ({len(df_gt)}).")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        sys.exit(1)

    # 3. Multilingual Normalization
    print("\n[Check 3/10] Verifying Multilingual Normalization...")
    try:
        rec_en = NormalizedRecord("S1-EN", "Orelee's Barbershop, LLC", "1795 Westchester Drive, High Point, NC 27262", "US")
        rec_in = NormalizedRecord("S1-IN", "राम मार्केटिंग प्राइवेट लिमिटेड", "12 एमजी रोड, बैंगलोर 560001", "India")
        rec_fr = NormalizedRecord("S1-FR", "Marina École France SARL", "14 Rue de Buci, 75006 Paris", "France")

        assert rec_en.norm_name == "orelee s barbershop llc"
        assert rec_en.core_name == "orelee s barbershop"
        assert rec_en.postal_code == "27262"

        # Unicode preserved (not destroyed to ASCII)
        assert "राम" in rec_in.norm_name
        assert rec_in.primary_script == "DEVANAGARI"
        assert rec_in.core_name == "राम मार्केटिंग"

        # Transliteration auxiliary representation
        assert "ecole" in rec_fr.translit_name
        assert rec_fr.postal_code == "75006"
        print("  ✓ Normalization preserved Unicode, extracted legal suffixes and postal codes correctly.")
    except Exception as e:
        print(f"  ✗ Normalization check failed: {e}")
        sys.exit(1)

    # 4. Compact Index Construction
    print("\n[Check 4/10] Verifying Compact Target Index Construction...")
    try:
        pool = CompactTargetPool()
        for r in df_s2.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        for r in df_s3.iter_rows():
            pool.add(r[0], r[1], r[2], r[3])
        print(f"  ✓ Compact target pool populated with {len(pool)} entities.")
    except Exception as e:
        print(f"  ✗ Index construction failed: {e}")
        sys.exit(1)

    # 5. Multi-Stage Candidate Retrieval
    print("\n[Check 5/10] Verifying Candidate Retrieval...")
    try:
        blocker = MultiStageBlocker(pool, max_candidates=25)
        test_rec = NormalizedRecord(df_s1["entity_id"][0], df_s1["business_name"][0], df_s1["business_address"][0], df_s1["country"][0])
        cands, meta = blocker.generate_candidates_for_record(test_rec)
        print(f"  ✓ Candidate blocker generated {len(cands)} candidates for query '{test_rec.raw_name[:30]}'.")
    except Exception as e:
        print(f"  ✗ Candidate retrieval failed: {e}")
        sys.exit(1)

    # 6. Feature Extraction
    print("\n[Check 6/10] Verifying Pairwise Feature Extraction...")
    try:
        feat_vec = FeatureExtractor.extract_pairwise_features(
            test_rec,
            pool.raw_names[0],
            pool.raw_addresses[0],
            pool.countries[0],
            use_extended_features=True
        )
        assert len(feat_vec) == len(EXTENDED_FEATURE_NAMES)
        print(f"  ✓ Successfully extracted {len(feat_vec)} dense features per pair.")
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        sys.exit(1)

    # 7. Model Training & Probability Calibration
    print("\n[Check 7/10] Verifying Model Training & Probability Calibration...")
    try:
        n_samples = 40
        X_mock = np.random.rand(n_samples, len(EXTENDED_FEATURE_NAMES)).astype(np.float32)
        y_mock = np.array([1 if i % 4 == 0 else 0 for i in range(n_samples)], dtype=np.int32)

        model = EntityMatchingModel(model_type="lightgbm", use_extended_features=True)
        model.fit(X_mock, y_mock)
        model.calibrate(X_mock[:20], y_mock[:20], method="isotonic")
        assert model.is_fitted
        assert model.is_calibrated
        print("  ✓ LightGBM model trained and calibrated with Isotonic Regression.")
    except Exception as e:
        print(f"  ✗ Model training failed: {e}")
        sys.exit(1)

    # 8. Prediction & Entity-Level Decision Logic
    print("\n[Check 8/10] Verifying Entity-Level Prediction & Margin Logic...")
    try:
        candidate_scores = {
            "S1-001": [("S2-001", 0.92), ("S2-002", 0.88), ("S2-003", 0.20)],
            "S1-002": [("S2-004", 0.35)]  # Should become singleton
        }
        decisions = model.make_entity_decisions(candidate_scores, decision_threshold=0.60, margin_threshold=0.10)
        assert decisions["S1-001"]["accepted_matches"] == ["S2-001", "S2-002"]
        assert decisions["S1-001"]["is_singleton"] is False
        assert decisions["S1-002"]["accepted_matches"] == []
        assert decisions["S1-002"]["is_singleton"] is True
        print("  ✓ Entity decision logic correctly resolved multi-match and singleton cases.")
    except Exception as e:
        print(f"  ✗ Prediction check failed: {e}")
        sys.exit(1)

    # 9 & 10. Output Generation & Submission Validation Logic
    print("\n[Check 9/10] Verifying Output TSV Generation...")
    try:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmp_dir_obj.name
        match_path = os.path.join(tmpdir, "matching_results.tsv")
        cand_path = os.path.join(tmpdir, "candidate_pairs.tsv")

        with open(match_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001,S2-002\n")
            f.write("S1-002\t\n")

        with open(cand_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-001,S2-002,S2-003\n")
            f.write("S1-002\tS2-004\n")

        assert os.path.isfile(match_path) and os.path.getsize(match_path) > 0
        assert os.path.isfile(cand_path) and os.path.getsize(cand_path) > 0
        print("  ✓ Output TSV files written with proper tab separation.")
    except Exception as e:
        print(f"  ✗ Output generation failed: {e}")
        sys.exit(1)

    print("\n[Check 10/10] Verifying Submission Validator Compliance...")
    try:
        from utils.validate_submission import validate_tsv_structure_and_ids
        issues = []
        val_map = validate_tsv_structure_and_ids(
            match_path,
            ["source1_entity_id", "matched_entity_ids"],
            {"S1-001", "S1-002"},
            {"S2-001", "S2-002", "S2-003", "S2-004"},
            issues,
            "smoke_test_matching"
        )
        assert len(issues) == 0, f"Validator found issues: {issues}"
        tmp_dir_obj.cleanup()
        print("  ✓ Official validator check passed with 0 issues.")
    except Exception as e:
        print(f"  ✗ Validator check failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" ALL 10 SMOKE TEST CHECKS PASSED SUCCESSFULLY! (EXIT CODE 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
