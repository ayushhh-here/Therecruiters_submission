#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Script: utils/validate_submission.py
#
# Description:
#   Submission validation script using ONLY the Python Standard Library.
#   It verifies that:
#     1. Both output files (matching_results.tsv and candidate_pairs.tsv) exist
#        and follow strict tab-separated format (.tsv).
#     2. Headers match the required column names exactly.
#     3. Every Source 1 entity in the test set appears EXACTLY ONCE (no missing,
#        no duplicates).
#     4. IDs in the match/candidate lists are valid Source 2 (S2-) or Source 3 (S3-)
#        IDs present in the test set.
#     5. No self-references (S1- IDs) are present in the matched lists.
#     6. No duplicate IDs exist within any single comma-separated list.
#     7. Every matched entity in matching_results.tsv is a subset of candidates in
#        candidate_pairs.tsv.
#
# Usage:
#   python utils/validate_submission.py \
#       --matching output/matching_results.tsv \
#       --candidate output/candidate_pairs.tsv \
#       --test-dir dataset/test
# ==============================================================================

# Import standard library modules (no third-party dependencies required)
import argparse
import os
import sys


def parse_arguments():
    """
    Parses command-line arguments for the submission validator.
    
    Returns:
        argparse.Namespace: Object containing parsed CLI arguments:
            - matching: Path to output/matching_results.tsv
            - candidate: Path to output/candidate_pairs.tsv
            - test_dir: Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv
    """
    # Create the argument parser with detailed help description
    parser = argparse.ArgumentParser(
        description="Validate Amazon ML Challenge submission files against official constraints."
    )
    
    # Define required file path arguments
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        help="Path to matching_results.tsv (final entity match predictions)"
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate_pairs.tsv (blocking candidates fed to model)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Directory containing test_source1.tsv, test_source2.tsv, and test_source3.tsv"
    )
    
    # Parse and return command line arguments
    return parser.parse_args()


def load_valid_test_ids(test_dir):
    """
    Loads all valid test entity IDs from test_source1.tsv, test_source2.tsv, and test_source3.tsv.
    
    Args:
        test_dir (str): Path to directory containing test set TSV files.
        
    Returns:
        tuple (set, set):
            - source1_ids: Set of valid S1 test entity IDs.
            - valid_s2_s3_ids: Set of valid S2 and S3 test entity IDs.
    """
    # Check that test directory exists
    if not os.path.isdir(test_dir):
        print(f"[ERROR] Test directory does not exist: {test_dir}")
        sys.exit(1)
        
    # Define expected file names inside test directory
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    # Verify that all 3 test files exist
    for fpath in [s1_file, s2_file, s3_file]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] Required test file missing: {fpath}")
            sys.exit(1)
            
    # Sets to accumulate valid entity identifiers
    source1_ids = set()
    valid_s2_s3_ids = set()
    
    # Helper function to read the first column ('entity_id') from a TSV file
    def extract_ids_from_tsv(filepath, target_set):
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            # Confirm that the first column is entity_id
            if not header or header[0] != "entity_id":
                print(f"[WARNING] Expected first column to be 'entity_id' in {filepath}")
            # Read subsequent lines and store entity_id
            for line_num, line in enumerate(f, start=2):
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                parts = stripped.split("\t")
                target_set.add(parts[0].strip())

    # Extract IDs from each respective source file
    extract_ids_from_tsv(s1_file, source1_ids)
    extract_ids_from_tsv(s2_file, valid_s2_s3_ids)
    extract_ids_from_tsv(s3_file, valid_s2_s3_ids)
    
    return source1_ids, valid_s2_s3_ids


def validate_tsv_structure_and_ids(filepath, expected_header, valid_s1_ids, valid_target_ids, issues, label):
    """
    Validates structural formatting, header, entity presence, uniqueness, and ID validity
    for either matching_results.tsv or candidate_pairs.tsv.
    
    Args:
        filepath (str): Path to the TSV file to check.
        expected_header (list): Expected list of header column names.
        valid_s1_ids (set): Set of expected Source 1 entity IDs.
        valid_target_ids (set): Set of valid S2 and S3 entity IDs.
        issues (list): Accumulator list of detected validation errors.
        label (str): Human-readable name for logging ("matching_results" or "candidate_pairs").
        
    Returns:
        dict: Mapping of {s1_id: set_of_target_ids} for downstream cross-checks.
    """
    # Check if the file exists on disk
    if not os.path.isfile(filepath):
        issues.append(f"File not found: '{filepath}'.")
        return {}

    mapping = {}
    seen_s1_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        # Read header line
        header_line = f.readline()
        if not header_line:
            issues.append(f"[{label}] File is completely empty.")
            return {}
            
        header = header_line.rstrip("\r\n").split("\t")
        if header != expected_header:
            issues.append(
                f"[{label}] Header mismatch. Expected: {expected_header}, but got: {header}. "
                "Ensure columns are tab-separated."
            )
            
        # Parse data rows line-by-line
        for line_num, line in enumerate(f, start=2):
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue
                
            parts = line_clean.split("\t")
            # If tab separator was omitted, there will be only 1 part or excessive parts
            if len(parts) > 2:
                issues.append(
                    f"[{label}] Line {line_num} has {len(parts)} columns instead of 2. "
                    "Make sure entries do not contain extra tabs."
                )
                continue
                
            s1_id = parts[0].strip()
            # If there's a second column, extract IDs; otherwise empty list
            target_ids_raw = parts[1].strip() if len(parts) > 1 else ""

            # Check 1: Duplicate Source 1 entity row
            if s1_id in seen_s1_ids:
                issues.append(f"[{label}] Duplicate row found for Source 1 ID '{s1_id}' on line {line_num}.")
            seen_s1_ids.add(s1_id)

            # Check 2: Unrecognized Source 1 ID not present in test_source1.tsv
            if s1_id not in valid_s1_ids:
                issues.append(
                    f"[{label}] Line {line_num}: Source 1 ID '{s1_id}' is not in test_source1.tsv."
                )

            # Parse comma-separated target IDs
            if target_ids_raw:
                id_tokens = [tok.strip() for tok in target_ids_raw.split(",") if tok.strip()]
                # Check 3: Check for duplicates within a single entity's list
                if len(id_tokens) != len(set(id_tokens)):
                    issues.append(
                        f"[{label}] Line {line_num}: Duplicate IDs in list for '{s1_id}': {id_tokens}"
                    )

                # Check 4: Verify each ID belongs to S2 or S3 test sets
                current_targets = set()
                for tid in id_tokens:
                    # Reject self-matches to Source 1
                    if tid.startswith("S1-"):
                        issues.append(
                            f"[{label}] Line {line_num}: Invalid self-match '{tid}' for '{s1_id}'. "
                            "Only S2 and S3 IDs are allowed."
                        )
                    # Check existence in test sets
                    elif tid not in valid_target_ids:
                        issues.append(
                            f"[{label}] Line {line_num}: ID '{tid}' does not exist in test_source2.tsv or test_source3.tsv."
                        )
                    current_targets.add(tid)
                    
                mapping[s1_id] = current_targets
            else:
                # Singleton (no matching or candidate records)
                mapping[s1_id] = set()

    # Check 5: Verify that every test Source 1 entity is present
    missing_s1 = valid_s1_ids - seen_s1_ids
    if missing_s1:
        issues.append(
            f"[{label}] Missing {len(missing_s1)} Source 1 test entities. "
            f"Examples: {list(missing_s1)[:5]}. Every Source 1 entity must have exactly one row."
        )

    return mapping


def main():
    """
    Main execution entrypoint for validation.
    Runs all tests and exits with code 0 if PASS, or code 1 with detailed failure reasons.
    """
    # Parse command line arguments
    args = parse_arguments()

    print("=" * 70)
    print(" Amazon ML Challenge - Submission Validator")
    print("=" * 70)
    print(f" Checking Matching File : {args.matching}")
    print(f" Checking Candidate File: {args.candidate}")
    print(f" Test Set Directory     : {args.test_dir}")
    print("-" * 70)

    # Accumulator list for any detected problems
    issues = []

    # Step 1: Load valid IDs from test sources
    valid_s1_ids, valid_s2_s3_ids = load_valid_test_ids(args.test_dir)
    print(f" Loaded {len(valid_s1_ids)} valid Source 1 test entities.")
    print(f" Loaded {len(valid_s2_s3_ids)} valid Source 2 and Source 3 target entities.")

    # Step 2: Validate matching_results.tsv
    matching_header = ["source1_entity_id", "matched_entity_ids"]
    matching_map = validate_tsv_structure_and_ids(
        args.matching, matching_header, valid_s1_ids, valid_s2_s3_ids, issues, "matching_results.tsv"
    )

    # Step 3: Validate candidate_pairs.tsv
    candidate_header = ["source1_entity_id", "candidate_entity_ids"]
    candidate_map = validate_tsv_structure_and_ids(
        args.candidate, candidate_header, valid_s1_ids, valid_s2_s3_ids, issues, "candidate_pairs.tsv"
    )

    # Step 4: Cross-file consistency check: every final match MUST be a candidate
    if matching_map and candidate_map:
        for s1_id, matches in matching_map.items():
            candidates = candidate_map.get(s1_id, set())
            unmatched = matches - candidates
            if unmatched:
                issues.append(
                    f"Pipeline consistency violation for '{s1_id}': "
                    f"Matched ID(s) {list(unmatched)} do not appear in candidate_pairs.tsv! "
                    "Every final match must be generated during the blocking stage."
                )

    # Step 5: Report outcome
    print("-" * 70)
    if not issues:
        print(" PASS: All submission constraints satisfied!")
        print(" Files are valid, properly formatted, and ready for leaderboard upload.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f" FAIL: Detected {len(issues)} validation issue(s):")
        for idx, issue in enumerate(issues[:30], start=1):
            print(f"   {idx}. {issue}")
        if len(issues) > 30:
            print(f"   ... and {len(issues) - 30} more issues.")
        print("=" * 70)
        sys.exit(1)


# Standard Python boilerplate to execute main when script is run directly
if __name__ == "__main__":
    main()
