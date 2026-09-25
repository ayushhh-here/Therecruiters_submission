#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/features.py
#
# Description:
#   High-Dimensional Pairwise Feature Engineering Engine.
#   Implements the feature groups defined in Section 6 of the practical guide:
#     1. Business Name Features:
#        - Normalized exact match (binary)
#        - Core name exact match (without legal suffixes)
#        - Token Jaccard similarity (word set overlap)
#        - Character trigram Jaccard similarity (sub-word typo tolerance)
#        - Levenshtein / Normalized Edit Distance similarity
#        - Token count difference and string length ratio
#     2. Business Address Features:
#        - Normalized exact address match
#        - Address token Jaccard similarity
#        - Address character trigram Jaccard similarity
#        - Numeric token overlap (house numbers, unit numbers)
#        - Postal / PIN code match signal (1 = match, -1 = mismatch, 0 = missing)
#        - Address length ratio
#     3. Open-Set Country Features:
#        - Strict country equality (1.0 = equal, 0.0 = different)
#        - Missing country indicators (handles open-set French and unseen records)
#     4. Cross-Field Non-Linear Interaction Signals:
#        - Name strong + address weak (e.g., chain franchise with different locations)
#        - Address strong + name weak (e.g., typo or rebranding at exact same premise)
#        - Both name and address strong (strongest match signal)
#        - Shared informative token count
#     5. Candidate Metadata Features:
#        - Total number of blocking rules that retrieved the pair
#        - Individual blocking rule flags (exact, sorted, postal, token, trigram)
#        - Candidate rank index within the query entity's candidate pool
# ==============================================================================

# Standard library imports
import math
from typing import Dict, List, Set, Tuple, Any, Optional

# Import normalized record representation
from src.normalization import NormalizedRecord

# Attempt to import RapidFuzz for high-speed C-optimized Levenshtein calculations.
# If unavailable, fall back to pure-Python dynamic programming to guarantee zero-crash execution.
try:
    from rapidfuzz.distance import Levenshtein as rapid_levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ==============================================================================
# 1. MATHEMATICAL & STRING SIMILARITY PRIMITIVES
# ==============================================================================

def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """
    Computes the Jaccard similarity index: |A ∩ B| / |A ∪ B|.
    
    Args:
        set_a (Set): First set of tokens or n-grams.
        set_b (Set): Second set of tokens or n-grams.
        
    Returns:
        float: Similarity between 0.0 (disjoint) and 1.0 (identical sets).
    """
    if not set_a and not set_b:
        return 0.0
    intersection_len = len(set_a & set_b)
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return intersection_len / union_len


def pure_python_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Levenshtein edit distance using dynamic programming with O(min(N, M)) space.
    Used as an unshakeable fallback when rapidfuzz is not installed.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        int: Minimum number of single-character edits (insertions, deletions, substitutions).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Ensure s1 is the shorter string to minimize space
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    previous_row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        current_row = [i + 1]
        for j, c1 in enumerate(s1):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Computes normalized string similarity based on Levenshtein distance:
        similarity = 1.0 - (edit_distance / max(len(s1), len(s2)))
        
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    if not s1 and not s2:
        return 0.0
    if s1 == s2:
        return 1.0
        
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    if HAS_RAPIDFUZZ:
        dist = rapid_levenshtein.distance(s1, s2)
    else:
        dist = pure_python_levenshtein_distance(s1, s2)

    return max(0.0, 1.0 - (dist / max_len))


def string_length_ratio(s1: str, s2: str) -> float:
    """
    Computes length ratio: min(len1, len2) / max(len1, len2).
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        float: Ratio between 0.0 and 1.0.
    """
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 0.0
    return min(len1, len2) / max_len


# ==============================================================================
# 2. FEATURE EXTRACTOR CLASS
# ==============================================================================

# Explicit list of feature column names for documentation and model interpretability
FEATURE_NAMES: List[str] = [
    # --- Name Features ---
    "feat_name_exact_match",
    "feat_name_core_exact_match",
    "feat_name_token_jaccard",
    "feat_name_trigram_jaccard",
    "feat_name_levenshtein_sim",
    "feat_name_token_count_diff",
    "feat_name_length_ratio",
    "feat_name_sorted_match",
    
    # --- Address Features ---
    "feat_addr_exact_match",
    "feat_addr_token_jaccard",
    "feat_addr_trigram_jaccard",
    "feat_addr_numeric_overlap",
    "feat_addr_postal_match",
    "feat_addr_length_ratio",
    
    # --- Country Features (Open-Set) ---
    "feat_country_exact_match",
    "feat_country_s1_unknown",
    "feat_country_target_unknown",
    
    # --- Cross-Field & Non-Linear Features ---
    "feat_cross_name_strong_addr_weak",
    "feat_cross_addr_strong_name_weak",
    "feat_cross_both_strong",
    "feat_cross_shared_informative_tokens",
    
    # --- Candidate Metadata Features ---
    "feat_meta_rule_count",
    "feat_meta_rule_exact_name",
    "feat_meta_rule_sorted_name",
    "feat_meta_rule_core_name",
    "feat_meta_rule_postal_code",
    "feat_meta_rule_token_overlap",
    "feat_meta_rule_trigram_overlap",
    "feat_meta_candidate_rank"
]


class FeatureExtractor:
    """
    Extracts dense numeric feature vectors for candidate entity pairs.
    Precomputed properties on NormalizedRecord objects allow for ultra-fast calculation.
    """

    @staticmethod
    def extract_pairwise_features(
        s1: NormalizedRecord,
        target: NormalizedRecord,
        candidate_meta: Optional[Dict[str, Any]] = None,
        candidate_rank: int = 0
    ) -> List[float]:
        """
        Extracts all pairwise features for a single (Source 1, Target) candidate pair.
        
        Args:
            s1 (NormalizedRecord): Normalized Source 1 entity.
            target (NormalizedRecord): Normalized Source 2 or Source 3 entity.
            candidate_meta (dict, optional): Metadata from blocking stage.
            candidate_rank (int): Rank of candidate in retrieval order (0-indexed).
            
        Returns:
            List[float]: Numeric feature vector matching FEATURE_NAMES.
        """
        if candidate_meta is None:
            candidate_meta = {"rules": set(), "rule_count": 0, "token_overlap": 0}

        rules = candidate_meta.get("rules", set())

        # ----------------------------------------------------------------------
        # 1. Business Name Features
        # ----------------------------------------------------------------------
        name_exact = 1.0 if (s1.norm_name and s1.norm_name == target.norm_name) else 0.0
        name_core_exact = 1.0 if (s1.core_name and s1.core_name == target.core_name) else 0.0
        name_sorted_exact = 1.0 if (s1.sorted_name_key and s1.sorted_name_key == target.sorted_name_key) else 0.0

        name_tok_jaccard = jaccard_similarity(s1.name_tokens, target.name_tokens)
        name_tri_jaccard = jaccard_similarity(s1.name_trigrams, target.name_trigrams)
        name_lev_sim = normalized_levenshtein_similarity(s1.norm_name, target.norm_name)

        name_tok_diff = float(abs(len(s1.name_tokens) - len(target.name_tokens)))
        name_len_ratio = string_length_ratio(s1.norm_name, target.norm_name)

        # ----------------------------------------------------------------------
        # 2. Business Address Features
        # ----------------------------------------------------------------------
        addr_exact = 1.0 if (s1.norm_address and s1.norm_address == target.norm_address) else 0.0
        addr_tok_jaccard = jaccard_similarity(s1.address_tokens, target.address_tokens)
        addr_tri_jaccard = jaccard_similarity(s1.address_trigrams, target.address_trigrams)
        addr_num_overlap = jaccard_similarity(s1.numeric_tokens, target.numeric_tokens)

        # Postal code signal:
        # +1.0 = both non-empty and matching
        # -1.0 = both non-empty but mismatching
        #  0.0 = either missing
        if s1.postal_code and target.postal_code:
            postal_match = 1.0 if s1.postal_code == target.postal_code else -1.0
        else:
            postal_match = 0.0

        addr_len_ratio = string_length_ratio(s1.norm_address, target.norm_address)

        # ----------------------------------------------------------------------
        # 3. Open-Set Country Features
        # ----------------------------------------------------------------------
        # Strictly compares strings without hard-coding {US, India}
        country_exact = 1.0 if (s1.norm_country == target.norm_country and s1.norm_country != "UNKNOWN") else 0.0
        country_s1_unk = 1.0 if s1.norm_country == "UNKNOWN" else 0.0
        country_t_unk = 1.0 if target.norm_country == "UNKNOWN" else 0.0

        # ----------------------------------------------------------------------
        # 4. Cross-Field Non-Linear Features
        # ----------------------------------------------------------------------
        cross_name_strong_addr_weak = 1.0 if (name_tok_jaccard >= 0.80 and addr_tok_jaccard <= 0.30) else 0.0
        cross_addr_strong_name_weak = 1.0 if (addr_tok_jaccard >= 0.80 and name_tok_jaccard <= 0.30) else 0.0
        cross_both_strong = 1.0 if (name_tok_jaccard >= 0.70 and addr_tok_jaccard >= 0.60) else 0.0
        shared_tokens = float(len((s1.name_tokens | s1.address_tokens) & (target.name_tokens | target.address_tokens)))

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

        # Assembled feature vector
        return [
            name_exact,
            name_core_exact,
            name_tok_jaccard,
            name_tri_jaccard,
            name_lev_sim,
            name_tok_diff,
            name_len_ratio,
            name_sorted_exact,
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
