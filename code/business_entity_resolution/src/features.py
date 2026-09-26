#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/features.py
#
# Description:
#   High-Dimensional Multilingual Pairwise Feature Engineering Engine.
#   Implements Phase 8 requirements:
#     1. Business Name Features:
#        - Exact normalized match, core name match, sorted match
#        - Transliterated / accent-invariant match
#        - Token Jaccard similarity & character trigram Jaccard
#        - C-accelerated Levenshtein similarity (RapidFuzz)
#        - Token count difference, string length ratio
#        - Name frequency / rarity penalty (downweights common names like 'CC')
#     2. Business Address Features:
#        - Exact address match, token Jaccard, character trigram Jaccard
#        - Numeric token overlap (house numbers, suite numbers)
#        - Postal code signal (+1 match, -1 mismatch, 0 missing)
#        - Missing address indicators (handles ~3% null target addresses)
#        - Address length ratio
#     3. Script & Open-Set Country Features:
#        - Strict country equality & unknown indicators
#        - Primary script concordance (e.g. Devanagari == Devanagari)
#     4. Cross-Field Non-Linear Interaction Signals:
#        - Strong name + weak address (detects chain franchises)
#        - Strong address + weak name (detects corporate renames / typos)
#        - Both strong match
#        - Shared informative token counts
#     5. Candidate Metadata & Retrieval Priority Signals:
#        - Rule agreement counts, individual rule triggers
#        - Retrieval rank within query candidate pool
#        - Retrieval priority score
# ==============================================================================

import math
from typing import Dict, List, Set, Tuple, Any, Optional

from src.normalization import (
    NormalizedRecord,
    generate_char_ngrams,
    normalize_unicode_and_case,
    remove_punctuation_unicode,
    transliterate_latin_accents,
    extract_postal_code,
    extract_numeric_tokens,
    isolate_legal_suffixes,
    normalize_country
)

# RapidFuzz C-extension for high-speed Levenshtein calculations
try:
    from rapidfuzz.distance import Levenshtein as rapid_levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ==============================================================================
# 1. STRING SIMILARITY PRIMITIVES
# ==============================================================================

def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Computes Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    intersection_len = len(set_a & set_b)
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return intersection_len / union_len


def fast_levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes normalized Levenshtein similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    if HAS_RAPIDFUZZ:
        dist = rapid_levenshtein.distance(s1, s2)
        return max(0.0, 1.0 - (dist / max_len))

    # Python fallback
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        curr = [i + 1]
        for j, c1 in enumerate(s1):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return max(0.0, 1.0 - (prev[-1] / max_len))


def string_length_ratio(s1: str, s2: str) -> float:
    """Computes string length ratio: min(len1, len2) / max(len1, len2)."""
    l1, l2 = len(s1), len(s2)
    m = max(l1, l2)
    return (min(l1, l2) / m) if m > 0 else 0.0


# ==============================================================================
# 2. FEATURE DEFINITIONS
# ==============================================================================

BASELINE_FEATURE_NAMES: List[str] = [
    "feat_name_exact_match",
    "feat_name_core_exact_match",
    "feat_name_token_jaccard",
    "feat_name_trigram_jaccard",
    "feat_name_levenshtein_sim",
    "feat_name_token_count_diff",
    "feat_name_length_ratio",
    "feat_name_sorted_match",
    "feat_addr_exact_match",
    "feat_addr_token_jaccard",
    "feat_addr_trigram_jaccard",
    "feat_addr_numeric_overlap",
    "feat_addr_postal_match",
    "feat_addr_length_ratio",
    "feat_country_exact_match",
    "feat_country_s1_unknown",
    "feat_country_target_unknown",
    "feat_cross_name_strong_addr_weak",
    "feat_cross_addr_strong_name_weak",
    "feat_cross_both_strong",
    "feat_cross_shared_informative_tokens",
    "feat_meta_rule_count",
    "feat_meta_rule_exact_name",
    "feat_meta_rule_sorted_name",
    "feat_meta_rule_core_name",
    "feat_meta_rule_postal_code",
    "feat_meta_rule_token_overlap",
    "feat_meta_rule_trigram_overlap",
    "feat_meta_candidate_rank"
]

EXTENDED_FEATURE_NAMES: List[str] = list(BASELINE_FEATURE_NAMES) + [
    "feat_name_translit_match",
    "feat_addr_missing_indicator",
    "feat_script_match",
    "feat_meta_retrieval_score",
    "feat_name_rarity_weight"
]

FEATURE_NAMES: List[str] = EXTENDED_FEATURE_NAMES


# ==============================================================================
# 3. FEATURE EXTRACTOR
# ==============================================================================

class FeatureExtractor:
    """Extracts dense numerical feature vectors for candidate entity pairs."""

    @staticmethod
    def extract_pairwise_features(
        s1: NormalizedRecord,
        target_raw_name: str,
        target_raw_address: str,
        target_country: str,
        candidate_meta: Optional[Dict[str, Any]] = None,
        candidate_rank: int = 0,
        use_extended_features: bool = True
    ) -> List[float]:
        """
        Extracts pairwise feature vector between Source 1 record and Target candidate.
        """
        if candidate_meta is None:
            candidate_meta = {"rules": set(), "rule_count": 0, "token_overlap": 0, "score": 0.0}

        rules = candidate_meta.get("rules", set())

        # Target normalized attributes computed on the fly
        t_norm_name = remove_punctuation_unicode(normalize_unicode_and_case(target_raw_name))
        t_translit_name = transliterate_latin_accents(t_norm_name)
        t_tokens = set(t_norm_name.split())
        t_norm_addr = remove_punctuation_unicode(normalize_unicode_and_case(target_raw_address))
        t_addr_tokens = set(t_norm_addr.split())
        t_numeric = set(extract_numeric_tokens(t_norm_addr))
        t_postal = extract_postal_code(target_raw_address)
        t_country = normalize_country(target_country)

        # Trigrams (computed only for candidate pair)
        t_name_trigrams = generate_char_ngrams(t_norm_name, n=3)
        t_addr_trigrams = generate_char_ngrams(t_norm_addr, n=3)

        # ----------------------------------------------------------------------
        # 1. Business Name Features
        # ----------------------------------------------------------------------
        name_exact = 1.0 if (s1.norm_name and s1.norm_name == t_norm_name) else 0.0
        name_sorted = 1.0 if (s1.sorted_name_key and s1.sorted_name_key == " ".join(sorted(t_tokens))) else 0.0

        # Core name match
        t_core, _ = isolate_legal_suffixes(target_raw_name)
        name_core = 1.0 if (s1.core_name and s1.core_name == t_core) else 0.0

        name_tok_jaccard = jaccard_similarity(s1.name_tokens, t_tokens)
        name_tri_jaccard = jaccard_similarity(s1.name_trigrams, t_name_trigrams)
        name_lev_sim = fast_levenshtein_similarity(s1.norm_name, t_norm_name)
        name_tok_diff = float(abs(len(s1.name_tokens) - len(t_tokens)))
        name_len_ratio = string_length_ratio(s1.norm_name, t_norm_name)

        # ----------------------------------------------------------------------
        # 2. Business Address Features
        # ----------------------------------------------------------------------
        addr_missing = 1.0 if not t_norm_addr or not s1.norm_address else 0.0
        addr_exact = 1.0 if (s1.norm_address and s1.norm_address == t_norm_addr) else 0.0
        addr_tok_jaccard = jaccard_similarity(s1.address_tokens, t_addr_tokens) if not addr_missing else 0.0
        addr_tri_jaccard = jaccard_similarity(s1.address_trigrams, t_addr_trigrams) if not addr_missing else 0.0
        addr_num_overlap = jaccard_similarity(s1.numeric_tokens, t_numeric) if not addr_missing else 0.0

        if s1.postal_code and t_postal:
            postal_match = 1.0 if s1.postal_code == t_postal else -1.0
        else:
            postal_match = 0.0

        addr_len_ratio = string_length_ratio(s1.norm_address, t_norm_addr) if not addr_missing else 0.0

        # ----------------------------------------------------------------------
        # 3. Country & Script Features
        # ----------------------------------------------------------------------
        country_exact = 1.0 if (s1.norm_country == t_country and s1.norm_country != "UNKNOWN") else 0.0
        country_s1_unk = 1.0 if s1.norm_country == "UNKNOWN" else 0.0
        country_t_unk = 1.0 if t_country == "UNKNOWN" else 0.0

        # ----------------------------------------------------------------------
        # 4. Cross-Field Non-Linear Features
        # ----------------------------------------------------------------------
        cross_name_strong_addr_weak = 1.0 if (name_tok_jaccard >= 0.80 and addr_tok_jaccard <= 0.30) else 0.0
        cross_addr_strong_name_weak = 1.0 if (addr_tok_jaccard >= 0.80 and name_tok_jaccard <= 0.30) else 0.0
        cross_both_strong = 1.0 if (name_tok_jaccard >= 0.70 and addr_tok_jaccard >= 0.60) else 0.0
        shared_tokens = float(len((s1.name_tokens | s1.address_tokens) & (t_tokens | t_addr_tokens)))

        # ----------------------------------------------------------------------
        # 5. Candidate Metadata Features
        # ----------------------------------------------------------------------
        rule_count = float(candidate_meta.get("rule_count", len(rules)))
        r_exact = 1.0 if "exact_name" in rules else 0.0
        r_sorted = 1.0 if "sorted_name" in rules else 0.0
        r_core = 1.0 if "core_name" in rules else 0.0
        r_postal = 1.0 if "postal_code" in rules else 0.0
        r_token = 1.0 if "token_overlap" in rules else 0.0
        r_trigram = 1.0 if "trigram_overlap" in rules else 0.0
        c_rank = float(candidate_rank)

        baseline_vector = [
            name_exact,
            name_core,
            name_tok_jaccard,
            name_tri_jaccard,
            name_lev_sim,
            name_tok_diff,
            name_len_ratio,
            name_sorted,
            addr_exact,
            addr_tok_jaccard,
            addr_tri_jaccard,
            addr_num_overlap,
            postal_match,
            addr_len_ratio,
            country_exact,
            country_s1_unk,
            country_t_unk,
            cross_name_strong_addr_weak,
            cross_addr_strong_name_weak,
            cross_both_strong,
            shared_tokens,
            rule_count,
            r_exact,
            r_sorted,
            r_core,
            r_postal,
            r_token,
            r_trigram,
            c_rank
        ]

        if not use_extended_features:
            return baseline_vector

        # ----------------------------------------------------------------------
        # 6. Extended Signals (Phase 8)
        # ----------------------------------------------------------------------
        # Transliterated equality (e.g. Marina École vs Marina Ecole)
        name_translit = 1.0 if (s1.translit_name and s1.translit_name == t_translit_name) else 0.0

        # Script match
        from src.normalization import detect_primary_script
        t_script = detect_primary_script(target_raw_name)
        script_match = 1.0 if s1.primary_script == t_script else 0.0

        # Retrieval score from candidate generator
        retrieval_score = float(candidate_meta.get("score", 0.0))

        # Name rarity penalty: very short names (< 4 chars, e.g. "CC", "PC", "LC") are low rarity
        name_rarity = 1.0 if len(t_norm_name) >= 6 else (len(t_norm_name) / 6.0)

        extended_vector = baseline_vector + [
            name_translit,
            addr_missing,
            script_match,
            retrieval_score,
            name_rarity
        ]

        return extended_vector
