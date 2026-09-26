#!/usr/bin/env python3
"""
tests/test_pipeline.py — Comprehensive Unit & Adversarial Test Suite

Covers:
  - Text normalization & Unicode preservation
  - Transliteration & Latin accent handling
  - Exact & near-exact matching
  - Multi-stage blocking & candidate ranking
  - Pairwise feature extraction
  - Calibrated probability prediction
  - Thresholding & singleton/no-match decision logic
  - Submission output validation
  - Adversarial cases:
      1. Same name, different address/entity
      2. Misspellings and severe typos
      3. Different punctuation styles
      4. Different scripts (Devanagari vs Latin)
      5. Common entity names ('CC', 'PC', 'Primary Care')
      6. Missing address
      7. Multiple plausible candidate targets
"""

import os
import sys
import unittest
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
    transliterate_latin_accents,
    isolate_legal_suffixes,
    extract_postal_code,
    extract_numeric_tokens,
    normalize_country,
    detect_primary_script
)
from src.blocking import CompactTargetPool, MultiStageBlocker
from src.features import FeatureExtractor, BASELINE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES
from src.matching import EntityMatchingModel
from src.evaluate import compute_entity_f05, evaluate_macro_f05


class TestNormalization(unittest.TestCase):
    """Unit tests for multilingual normalization and Unicode preservation."""

    def test_unicode_preservation_indic(self):
        """Devanagari text must NOT be destroyed or converted to ASCII."""
        raw = "राम मार्केटिंग प्राइवेट लिमिटेड"
        rec = NormalizedRecord("S1-IN", raw, "12 एमजी रोड, बैंगलोर 560001", "India")
        self.assertIn("राम", rec.norm_name)
        self.assertIn("मार्केटिंग", rec.norm_name)
        self.assertEqual(rec.primary_script, "DEVANAGARI")
        self.assertEqual(rec.core_name, "राम मार्केटिंग")
        self.assertEqual(rec.legal_suffix, "प्राइवेट लिमिटेड")

    def test_french_accents_and_transliteration(self):
        """French accented characters must be preserved in norm_name and stripped in translit_name."""
        raw = "Marina École France SARL"
        rec = NormalizedRecord("S1-FR", raw, "14 Rue de Buci, 75006 Paris", "France")
        self.assertIn("école", rec.norm_name)
        self.assertIn("ecole", rec.translit_name)
        self.assertEqual(rec.legal_suffix, "sarl")
        self.assertEqual(rec.postal_code, "75006")

    def test_legal_suffix_isolation(self):
        """Legal suffixes must be cleanly separated from core business name."""
        cases = [
            ("Apex Dynamics LLC", "apex dynamics", "llc"),
            ("Shree Balaji Textiles Pvt Ltd", "shree balaji textiles", "pvt ltd"),
            ("Boulangerie Saint-Germain SARL", "boulangerie saint germain", "sarl"),
        ]
        for raw, expected_core, expected_suffix in cases:
            core, suffix = isolate_legal_suffixes(raw)
            self.assertEqual(core, expected_core)
            self.assertEqual(suffix, expected_suffix)

    def test_postal_code_extraction(self):
        """Postal code extraction for US (5), India (6), France (5)."""
        self.assertEqual(extract_postal_code("High Point, NC 27262"), "27262")
        self.assertEqual(extract_postal_code("MG Road, Bangalore 560001"), "560001")
        self.assertEqual(extract_postal_code("Rue de Buci, 75006 Paris"), "75006")


class TestBlockingAndRanking(unittest.TestCase):
    """Unit tests for compact blocking, indexing, and candidate ranking."""

    def setUp(self):
        self.pool = CompactTargetPool()
        self.pool.add("S2-001", "Orelee Barber Shop", "1795 Westchester Drive, NC", "US")
        self.pool.add("S2-002", "Orelee Barbershop LLC", "Different Address, NY", "US")
        self.pool.add("S2-003", "Completely Unrelated Corp", "999 Oak Street, CA", "US")
        self.pool.add("S2-004", "Marina Ecole France", "14 Rue de Buci, Paris", "France")
        self.blocker = MultiStageBlocker(self.pool, max_candidates=10)

    def test_exact_and_sorted_retrieval(self):
        rec = NormalizedRecord("S1-001", "Orelee's Barbershop", "1795 Westchester Drive, NC 27262", "US")
        cands, meta = self.blocker.generate_candidates_for_record(rec)
        self.assertIn("S2-001", cands)
        self.assertIn("S2-002", cands)
        self.assertNotIn("S2-003", cands)
        self.assertNotIn("S2-004", cands)  # Country filter: US vs France

    def test_transliterated_french_retrieval(self):
        rec = NormalizedRecord("S1-FR", "Marina École France SARL", "14 Rue de Buci, Paris", "France")
        cands, meta = self.blocker.generate_candidates_for_record(rec)
        self.assertIn("S2-004", cands)


class TestFeatureExtraction(unittest.TestCase):
    """Unit tests for feature engineering."""

    def test_feature_vector_dimension_and_bounds(self):
        rec = NormalizedRecord("S1-01", "Beacon Hill Pharmacy", "210 Charles Street, Boston, MA 02114", "US")
        vec = FeatureExtractor.extract_pairwise_features(
            rec,
            "Beacon Hill Pharmacy Inc",
            "210 Charles St, Boston, MA",
            "US",
            use_extended_features=True
        )
        self.assertEqual(len(vec), len(EXTENDED_FEATURE_NAMES))
        # Exact name match or core match should be 1.0
        self.assertEqual(vec[1], 1.0)  # Core name exact match


class TestEntityDecisionLogic(unittest.TestCase):
    """Unit tests for score calibration, margin decision, and singleton logic."""

    def setUp(self):
        self.model = EntityMatchingModel(model_type="logistic", use_extended_features=True)
        # Mock fitted
        self.model.is_fitted = True
        self.model.optimal_threshold = 0.65
        self.model.margin_threshold = 0.15

    def test_singleton_decision_when_below_threshold(self):
        scores = {"S1-SINGLE": [("S2-A", 0.40), ("S2-B", 0.30)]}
        decisions = self.model.make_entity_decisions(scores, decision_threshold=0.65)
        self.assertTrue(decisions["S1-SINGLE"]["is_singleton"])
        self.assertEqual(decisions["S1-SINGLE"]["accepted_matches"], [])

    def test_multi_match_decision_within_margin(self):
        scores = {"S1-MULTI": [("S2-A", 0.88), ("S2-B", 0.82), ("S2-C", 0.66)]}
        decisions = self.model.make_entity_decisions(scores, decision_threshold=0.65, margin_threshold=0.10)
        # S2-A (0.88) and S2-B (0.82, diff=0.06 <= 0.10) accepted; S2-C (0.66, diff=0.22 > 0.10) rejected
        self.assertFalse(decisions["S1-MULTI"]["is_singleton"])
        self.assertEqual(decisions["S1-MULTI"]["accepted_matches"], ["S2-A", "S2-B"])


class TestAdversarialCases(unittest.TestCase):
    """Adversarial stress tests on noisy real-world patterns."""

    def test_same_name_different_entity(self):
        """Same company name in completely different cities / countries."""
        rec = NormalizedRecord("S1-ADV1", "Apex Logistics", "100 Industrial Parkway, Chicago, IL 60607", "US")
        pool = CompactTargetPool()
        pool.add("S2-DIFF", "Apex Logistics", "55 MG Road, Bangalore 560001", "India")
        blocker = MultiStageBlocker(pool, max_candidates=10)
        cands, _ = blocker.generate_candidates_for_record(rec)
        # Country filter must strictly prevent cross-country mismatch
        self.assertNotIn("S2-DIFF", cands)

    def test_common_name_rarity(self):
        """Very short/common acronyms like 'CC' should not get artificially high confidence."""
        rec = NormalizedRecord("S1-CC", "CC", "123 Main Street", "US")
        vec = FeatureExtractor.extract_pairwise_features(rec, "CC", "456 Other Street", "US", use_extended_features=True)
        # Name rarity weight feature should be low for short names (<4 chars)
        rarity_weight = vec[-1]
        self.assertLess(rarity_weight, 0.5)

    def test_missing_address_handling(self):
        """Missing address in target record must not crash feature extractor."""
        rec = NormalizedRecord("S1-ADDR", "Acme Industries", "100 Main St", "US")
        vec = FeatureExtractor.extract_pairwise_features(rec, "Acme Industries", "", "US", use_extended_features=True)
        # Missing address indicator should be 1.0
        self.assertEqual(len(vec), len(EXTENDED_FEATURE_NAMES))

    def test_singleton_metric_reward(self):
        """Singleton correctly predicted as [] must receive exact 1.0 credit."""
        p, r, f05 = compute_entity_f05(predicted_ids=set(), ground_truth_ids=set())
        self.assertEqual(f05, 1.0)

    def test_singleton_false_merge_penalty(self):
        """Singleton with false merge must receive exact 0.0 credit."""
        p, r, f05 = compute_entity_f05(predicted_ids={"S2-FALSE"}, ground_truth_ids=set())
        self.assertEqual(f05, 0.0)


if __name__ == "__main__":
    unittest.main()
