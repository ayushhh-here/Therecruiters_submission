#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/blocking.py
#
# Description:
#   High-Recall, Multi-Stage Candidate Generation (Blocking) Engine.
#   As detailed in Section 5 of the practical guide:
#     - Directly comparing all Source 1 records against all Source 2 and Source 3
#       records is a massive Cartesian product (O(N * M)) that is computationally
#       intractable.
#     - Blocking acts as the first filtering sieve, selecting a high-quality candidate
#       pool for each Source 1 business entity.
#     - The challenge guidelines emphasize:
#         1. Blocking determines the UPPER BOUND on recall: if a true match is missed
#            during blocking, downstream ML models can never recover it.
#         2. Rule 1: Country-aware blocking (strictly OPEN-SET; handles France, US,
#            India, and any unseen future labels).
#         3. Rule 2: Exact/near-exact normalized keys (sorted name, exact name, postal code).
#         4. Rule 3: Informative token inverted index (excluding pathological high-frequency words).
#         5. Rule 4: Character 3-gram retrieval for typo and transliteration resilience.
#         6. Rule 5: Multiple-block union to combine complementary signals.
#         7. Rule 6: Pathological block capping to prune runaway word blocks (e.g., 'hotel', 'restaurant').
# ==============================================================================

# Standard library imports
import collections
from typing import Dict, List, Set, Tuple, Any, Optional

# Import normalized record representation from our normalization module
from src.normalization import NormalizedRecord


# ==============================================================================
# 1. PATHOLOGICAL FREQUENT WORDS TO FILTER FROM INVERTED INDEXES
# ==============================================================================

# Extremely common words that appear across thousands of businesses.
# Indexing these creates giant candidate blocks that degrade speed without providing
# discriminative value.
STOP_TOKENS: Set[str] = {
    # Generic entity descriptors
    "restaurant", "hotel", "cafe", "dhaba", "bistro", "bar", "grill",
    "store", "shop", "mart", "bazaar", "center", "centre", "mall",
    "services", "solutions", "enterprises", "enterprise", "agency",
    "consulting", "associates", "group", "holdings", "company", "co",
    # Generic location words
    "india", "usa", "us", "state", "city", "road", "street", "avenue",
    "cross", "main", "near", "opposite", "behind", "floor", "building",
    "nagar", "colony", "market", "complex", "sector", "lane", "block"
}


# ==============================================================================
# 2. INVERTED INDEX DATA STRUCTURE
# ==============================================================================

class InvertedIndex:
    """
    Fast in-memory inverted index mapping informative tokens/keys to lists of target entity IDs.
    Allows O(1) candidate lookup per token during retrieval.
    """
    def __init__(self, max_token_frequency: int = 500):
        """
        Initializes the inverted index.
        
        Args:
            max_token_frequency (int): Maximum allowed entity count per token.
                Tokens exceeding this frequency are capped or treated as non-informative.
        """
        # Dictionary mapping token -> list of target entity IDs
        self.index: Dict[str, List[str]] = collections.defaultdict(list)
        # Threshold to cap pathological blocks
        self.max_token_frequency: int = max_token_frequency

    def add(self, key: str, entity_id: str) -> None:
        """
        Adds a single key -> entity_id association.
        
        Args:
            key (str): Token or normalized string key.
            entity_id (str): Target entity ID (S2 or S3).
        """
        # Skip empty keys
        if not key:
            return
        self.index[key].append(entity_id)

    def query(self, key: str) -> List[str]:
        """
        Retrieves all entity IDs associated with the given key.
        Applies pathological block capping.
        
        Args:
            key (str): Token or key to lookup.
            
        Returns:
            List[str]: List of matching entity IDs, or empty list if capped/not found.
        """
        if not key or key not in self.index:
            return []
            
        results = self.index[key]
        # Rule 6: Cap pathological blocks if token is overly frequent
        if len(results) > self.max_token_frequency:
            return []
            
        return results


# ==============================================================================
# 3. MULTI-STAGE CANDIDATE GENERATOR
# ==============================================================================

class MultiStageBlocker:
    """
    Orchestrates candidate generation across multiple complementary blocking rules:
      - Rule 1: Country-aware filtering (open-set)
      - Rule 2: Exact normalized name and sorted name matching
      - Rule 3: Informative token inverted indexing on names
      - Rule 4: Character 3-gram indexing for typo recovery
      - Rule 5: Postal code + address token blocking
    """

    def __init__(
        self,
        target_records: Dict[str, NormalizedRecord],
        max_candidates_per_source: int = 60,
        max_token_freq: int = 300
    ):
        """
        Builds all inverted indexes over Source 2 and Source 3 records once.
        
        Args:
            target_records (Dict[str, NormalizedRecord]):
                Pre-normalized dictionary of all Source 2 and Source 3 records.
            max_candidates_per_source (int):
                Maximum number of candidates retained per Source 1 entity.
            max_token_freq (int):
                Maximum frequency threshold for inverted token blocking.
        """
        self.target_records: Dict[str, NormalizedRecord] = target_records
        self.max_candidates_per_source: int = max_candidates_per_source

        # Inverted index for exact normalized names
        self.exact_name_index = InvertedIndex(max_token_frequency=max_token_freq)
        
        # Inverted index for sorted name keys (handles token order variations)
        self.sorted_name_index = InvertedIndex(max_token_frequency=max_token_freq)
        
        # Inverted index for core names (stripped of legal suffixes)
        self.core_name_index = InvertedIndex(max_token_frequency=max_token_freq)
        
        # Inverted index for postal codes
        self.postal_code_index = InvertedIndex(max_token_frequency=max_token_freq)
        
        # Inverted index for individual informative name tokens
        self.name_token_index = InvertedIndex(max_token_frequency=max_token_freq)
        
        # Inverted index for character trigrams
        self.trigram_index = InvertedIndex(max_token_frequency=max_token_freq * 2)

        # Populate all indexes with target records (S2 and S3)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """
        Iterates over all target records and registers them into the respective inverted indexes.
        Precomputing this once provides massive speedups during batch inference.
        """
        for tid, record in self.target_records.items():
            # 1. Exact normalized name
            if record.norm_name:
                self.exact_name_index.add(record.norm_name, tid)
                
            # 2. Sorted token name key
            if record.sorted_name_key:
                self.sorted_name_index.add(record.sorted_name_key, tid)
                
            # 3. Core name (without legal suffixes)
            if record.core_name:
                self.core_name_index.add(record.core_name, tid)
                
            # 4. Postal / PIN code
            if record.postal_code:
                self.postal_code_index.add(record.postal_code, tid)
                
            # 5. Informative name tokens (filtering out common stop words)
            for token in record.name_tokens:
                if len(token) >= 3 and token not in STOP_TOKENS:
                    self.name_token_index.add(token, tid)
                    
            # 6. Character trigrams from name
            for trigram in record.name_trigrams:
                self.trigram_index.add(trigram, tid)

    def generate_candidates_for_record(
        self,
        s1_record: NormalizedRecord
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Generates candidate target entity IDs (S2/S3) for a single Source 1 record.
        
        Args:
            s1_record (NormalizedRecord): The normalized Source 1 query entity.
            
        Returns:
            Tuple[List[str], Dict[str, Dict[str, Any]]]:
                - candidate_list: Ranked list of target IDs (up to max_candidates_per_source).
                - candidate_metadata: Dictionary mapping target_id -> metadata dict:
                    - 'rule_count': Number of distinct blocking rules that retrieved this candidate.
                    - 'rules': Set of rule names that triggered.
                    - 'token_overlap': Number of shared tokens.
        """
        # Dictionary to track metadata for each candidate:
        # target_id -> {"rules": set_of_rules, "score": weighted_priority}
        candidates_meta: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {"rules": set(), "score": 0.0, "token_overlap": 0}
        )

        s1_country = s1_record.norm_country

        # Helper function to check open-set country compatibility
        def is_country_compatible(target_id: str) -> bool:
            """
            Checks if country labels are compatible.
            Complies with Open-Set requirements:
              - If either country is 'UNKNOWN', allow comparison (safe fallback).
              - If both countries are known strings, they must match exactly.
              - Never filters out France or any unseen country label.
            """
            target_record = self.target_records.get(target_id)
            if not target_record:
                return False
            t_country = target_record.norm_country
            if s1_country == "UNKNOWN" or t_country == "UNKNOWN":
                return True
            return s1_country == t_country

        # ----------------------------------------------------------------------
        # Rule 2A: Exact Normalized Name Match
        # ----------------------------------------------------------------------
        if s1_record.norm_name:
            for tid in self.exact_name_index.query(s1_record.norm_name):
                if is_country_compatible(tid):
                    candidates_meta[tid]["rules"].add("exact_name")
                    candidates_meta[tid]["score"] += 10.0

        # ----------------------------------------------------------------------
        # Rule 2B: Sorted Name Key Match (Handles word order variations)
        # ----------------------------------------------------------------------
        if s1_record.sorted_name_key:
            for tid in self.sorted_name_index.query(s1_record.sorted_name_key):
                if is_country_compatible(tid):
                    candidates_meta[tid]["rules"].add("sorted_name")
                    candidates_meta[tid]["score"] += 8.0

        # ----------------------------------------------------------------------
        # Rule 2C: Core Name Match (Stripped of legal suffixes)
        # ----------------------------------------------------------------------
        if s1_record.core_name and s1_record.core_name != s1_record.norm_name:
            for tid in self.core_name_index.query(s1_record.core_name):
                if is_country_compatible(tid):
                    candidates_meta[tid]["rules"].add("core_name")
                    candidates_meta[tid]["score"] += 6.0

        # ----------------------------------------------------------------------
        # Rule 2D: Postal Code + Name Overlap
        # ----------------------------------------------------------------------
        if s1_record.postal_code:
            for tid in self.postal_code_index.query(s1_record.postal_code):
                if is_country_compatible(tid):
                    # Check if there is also at least some name overlap to avoid false zip merges
                    t_record = self.target_records.get(tid)
                    if t_record and (s1_record.name_tokens & t_record.name_tokens):
                        candidates_meta[tid]["rules"].add("postal_code")
                        candidates_meta[tid]["score"] += 5.0

        # ----------------------------------------------------------------------
        # Rule 3: Informative Name Token Inverted Index
        # ----------------------------------------------------------------------
        token_hits: Dict[str, int] = collections.defaultdict(int)
        for token in s1_record.name_tokens:
            if len(token) >= 3 and token not in STOP_TOKENS:
                for tid in self.name_token_index.query(token):
                    if is_country_compatible(tid):
                        token_hits[tid] += 1

        # Keep candidates sharing at least 1 significant token
        for tid, count in token_hits.items():
            candidates_meta[tid]["rules"].add("token_overlap")
            candidates_meta[tid]["score"] += count * 2.0
            candidates_meta[tid]["token_overlap"] = count

        # ----------------------------------------------------------------------
        # Rule 4: Character Trigram Retrieval (Typo / Transliteration Recovery)
        # ----------------------------------------------------------------------
        # Only query trigrams if we have relatively few candidates so far
        if len(candidates_meta) < self.max_candidates_per_source // 2:
            trigram_hits: Dict[str, int] = collections.defaultdict(int)
            for trigram in s1_record.name_trigrams:
                for tid in self.trigram_index.query(trigram):
                    if is_country_compatible(tid):
                        trigram_hits[tid] += 1
            
            # Select targets sharing a significant number of trigrams (>= 3)
            for tid, count in trigram_hits.items():
                if count >= 3:
                    candidates_meta[tid]["rules"].add("trigram_overlap")
                    candidates_meta[tid]["score"] += count * 0.2

        # ----------------------------------------------------------------------
        # Rule 5: Multiple-Block Union & Priority Ranking
        # ----------------------------------------------------------------------
        # Sort candidates by descending priority score and rule count
        sorted_candidates = sorted(
            candidates_meta.keys(),
            key=lambda tid: (
                len(candidates_meta[tid]["rules"]),
                candidates_meta[tid]["score"]
            ),
            reverse=True
        )

        # Cap candidates per Source 1 entity
        top_candidates = sorted_candidates[:self.max_candidates_per_source]

        # Record final metadata for top candidates
        final_meta = {}
        for tid in top_candidates:
            info = candidates_meta[tid]
            final_meta[tid] = {
                "rule_count": len(info["rules"]),
                "rules": info["rules"],
                "score": info["score"],
                "token_overlap": info["token_overlap"]
            }

        return top_candidates, final_meta


# ==============================================================================
# 4. BLOCKING EVALUATION & EXPORT UTILITIES
# ==============================================================================

def evaluate_blocking_quality(
    candidates_dict: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    total_target_count: int
) -> Dict[str, float]:
    """
    Computes blocking quality metrics as specified in the practical guide:
      1. Candidate Recall: Percentage of all true positive matches captured in candidates.
         (Determines the upper-bound ceiling for the ML model).
      2. Average Candidates Per S1: Average candidate count per query entity.
      3. Reduction Ratio: Percentage reduction compared to full Cartesian product.
      
    Args:
        candidates_dict (Dict[str, List[str]]): Generated candidates {s1_id: [s2_s3_ids]}.
        ground_truth (Dict[str, List[str]]): True matches {s1_id: [true_matches]}.
        total_target_count (int): Total number of target entities (len(S2) + len(S3)).
        
    Returns:
        Dict[str, float]: Dictionary of computed metric values.
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_candidates_generated = 0
    num_s1 = len(candidates_dict)

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands_set = set(candidates_dict.get(s1_id, []))
        for match_id in true_matches:
            total_true_matches += 1
            if match_id in cands_set:
                captured_true_matches += 1

    for cands in candidates_dict.values():
        total_candidates_generated += len(cands)

    candidate_recall = (
        (captured_true_matches / total_true_matches)
        if total_true_matches > 0 else 1.0
    )
    avg_candidates = total_candidates_generated / max(num_s1, 1)

    total_possible_pairs = num_s1 * total_target_count
    reduction_ratio = (
        (1.0 - (total_candidates_generated / total_possible_pairs))
        if total_possible_pairs > 0 else 1.0
    )

    return {
        "candidate_recall": candidate_recall,
        "captured_matches": captured_true_matches,
        "total_true_matches": total_true_matches,
        "avg_candidates_per_s1": avg_candidates,
        "reduction_ratio": reduction_ratio
    }


def export_candidate_pairs_tsv(
    candidates_dict: Dict[str, List[str]],
    output_filepath: str,
    all_s1_ids: List[str]
) -> None:
    """
    Exports candidates to the official 'output/candidate_pairs.tsv' format.
    Ensures:
      - Tab-separated values: 'source1_entity_id\tcandidate_entity_ids'.
      - Exactly one row per Source 1 entity in all_s1_ids.
      - Comma-separated candidate IDs without quotes.
      - Empty string for entities with 0 candidates.
      
    Args:
        candidates_dict (Dict[str, List[str]]): Mapping of s1_id to list of candidate IDs.
        output_filepath (str): Destination TSV file path.
        all_s1_ids (List[str]): Full list of all Source 1 entity IDs.
    """
    with open(output_filepath, "w", encoding="utf-8") as f:
        # Write exact required header
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates_dict.get(s1_id, [])
            # Format comma-separated list of target IDs
            cands_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")
