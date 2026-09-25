#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/normalization.py
#
# Description:
#   Comprehensive text normalization and canonical representation pipeline.
#   In accordance with the official challenge guidelines:
#     - We keep BOTH raw and normalized representations so downstream models
#       can leverage exact lexical match features alongside clean normalized keys.
#     - Normalization handles:
#         1. Unicode normalization (NFKD) and case folding.
#         2. Whitespace collapsing and noise stripping.
#         3. Business legal suffix isolation/removal (Pvt, Ltd, LLC, Inc, Corp, etc.)
#            to prevent corporate structure tokens from distorting name similarity.
#         4. Address abbreviation standardizations (Rd -> Road, St -> Street, etc.).
#         5. Dedicated extraction of numeric tokens (house numbers, postal/PIN codes).
#         6. Character n-gram generation (2-4 grams) for typo and transliteration recovery.
#         7. Open-set country standardization (stripping and uppercase, without hard-coding
#            any country domain).
# ==============================================================================

# Standard library imports for regular expressions, unicode handling, and typing
import re
import unicodedata
from typing import Dict, List, Set, Tuple, Any, Optional


# ==============================================================================
# 1. DICTIONARIES & PATTERNS FOR CLEANING
# ==============================================================================

# Common business legal suffixes across US, India, and European (France) jurisdictions.
# Removing or isolating these prevents two companies with different legal forms from
# looking falsely dissimilar, or two unrelated "LLC" companies from matching falsely.
LEGAL_SUFFIXES: Set[str] = {
    # English / International legal suffixes
    "pvt", "private", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llc", "llp", "pllc",
    "gmbh", "ag", "sa", "sarl", "sas", "bv", "nv", "spa", "srl",
    "enterprises", "enterprise", "holdings", "holding", "group",
    "services", "solutions", "technologies", "consulting", "industries"
}

# Common address abbreviations mapped to canonical expansions.
# Ensures that "123 Main St." matches "123 Main Street" accurately.
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    # Road / Street types
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "sq": "square",
    "cir": "circle",
    "ter": "terrace",
    "way": "way",
    "aly": "alley",
    
    # Unit / Sub-location types
    "apt": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "bldg": "building",
    "rm": "room",
    "dept": "department",
    "sec": "sector",
    "blk": "block",
    "plot": "plot",
    
    # Directionals
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    
    # Indian / International landmark prefixes
    "opp": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "jn": "junction",
    "po": "post office"
}

# Pre-compiled regular expressions for speed during repeated execution
RE_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")  # Strip punctuation but keep alphanumeric
RE_MULTIPLE_WHITESPACE = re.compile(r"\s+")      # Collapse multiple spaces, tabs, newlines
RE_NUMBERS = re.compile(r"\b\d+\b")              # Extract standalone numeric tokens


# ==============================================================================
# 2. STRING NORMALIZATION UTILITIES
# ==============================================================================

def clean_unicode_and_case(text: Optional[str]) -> str:
    """
    Standardizes unicode characters into ASCII compatibility decomposition (NFKD),
    converts text to lowercase, and strips leading/trailing whitespace.
    
    Args:
        text (str or None): The raw input string.
        
    Returns:
        str: Cleaned, lowercased, and normalized unicode string.
    """
    # Return empty string if input is None or NaN
    if text is None or not isinstance(text, str):
        return ""
        
    # NFKD normalizes accents (e.g., 'Café' -> 'Cafe') and full-width characters
    normalized = unicodedata.normalize("NFKD", text)
    
    # Encode to ASCII bytes ignoring unknown characters, then decode back to UTF-8
    ascii_clean = normalized.encode("ascii", "ignore").decode("utf-8")
    
    # Convert to lowercase and strip whitespace
    return ascii_clean.lower().strip()


def remove_punctuation(text: str) -> str:
    """
    Replaces all non-alphanumeric characters with a single space to avoid merging tokens
    (e.g., 'Walmart/Target' becomes 'Walmart Target').
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Text containing only lowercase alphanumeric tokens and single spaces.
    """
    # Replace symbols/punctuation with spaces
    spaced = RE_NON_ALPHANUMERIC.sub(" ", text)
    
    # Collapse multiple consecutive whitespace characters into a single space
    return RE_MULTIPLE_WHITESPACE.sub(" ", spaced).strip()


def normalize_address_text(address_text: Optional[str]) -> str:
    """
    Applies domain-specific address normalization:
      1. Cleans unicode and lowers case.
      2. Replaces punctuation with whitespace.
      3. Expands standard abbreviations (rd -> road, st -> street, etc.).
      4. Collapses whitespace.
      
    Args:
        address_text (str): Raw business address string.
        
    Returns:
        str: Canonical normalized address string.
    """
    # Baseline cleaning
    base = clean_unicode_and_case(address_text)
    punct_cleaned = remove_punctuation(base)
    
    # Tokenize by whitespace
    tokens = punct_cleaned.split()
    
    # Expand recognized abbreviations
    expanded_tokens = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    
    # Reassemble tokens with single spaces
    return " ".join(expanded_tokens)


def isolate_legal_suffixes(name_text: Optional[str]) -> Tuple[str, str]:
    """
    Separates the core business name from corporate legal structure identifiers.
    
    Args:
        name_text (str): Raw business entity name.
        
    Returns:
        Tuple[str, str]:
            - core_name: Business name stripped of trailing legal suffixes.
            - suffix_found: The extracted suffix (or empty string if none).
    """
    # Base cleaning
    clean_name = remove_punctuation(clean_unicode_and_case(name_text))
    tokens = clean_name.split()
    
    if not tokens:
        return "", ""
        
    # Check trailing tokens for legal suffixes (e.g. ['apple', 'computer', 'inc'])
    suffix_tokens = []
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        suffix_tokens.insert(0, tokens.pop())
        
    core_name = " ".join(tokens)
    extracted_suffix = " ".join(suffix_tokens)
    
    # If stripping removed everything (e.g. company called 'Enterprises LLC'), retain original
    if not core_name:
        core_name = clean_name
        
    return core_name, extracted_suffix


def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """
    Extracts all standalone digit sequences from a text string.
    Useful for comparing building numbers, suite numbers, and postal PIN codes.
    
    Args:
        text (str): Input text (typically address).
        
    Returns:
        List[str]: List of extracted digit strings in order of occurrence.
    """
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def extract_postal_code(address_text: Optional[str]) -> str:
    """
    Extracts potential postal or PIN codes from an address string.
    Supports US 5-digit zip codes and Indian 6-digit PIN codes and European 5-digit codes.
    
    Args:
        address_text (str): Normalized or raw address text.
        
    Returns:
        str: The most plausible postal code string, or empty string if none found.
    """
    if not address_text:
        return ""
    # Find all 5-digit or 6-digit numeric sequences
    candidates = re.findall(r"\b\d{5,6}\b", address_text)
    if candidates:
        # Return the last candidate (standard postal code position at end of address)
        return candidates[-1]
    return ""


def generate_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Generates a set of character n-grams from a string.
    Character n-grams provide robustness against typos, misspellings, and transliterations.
    
    Args:
        text (str): Input string.
        n (int): Length of each character n-gram (default: 3 for trigrams).
        
    Returns:
        Set[str]: Set of n-gram substrings.
    """
    # Compact string without spaces for n-gram comparison
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def normalize_country(country_text: Optional[str]) -> str:
    """
    Standardizes country string in an OPEN-SET compliant manner.
    CRITICAL: Never hard-codes allowed countries. Strips and uppercases any country label
    (e.g., 'us' -> 'US', 'india' -> 'INDIA', 'france' -> 'FRANCE', etc.).
    
    Args:
        country_text (str or None): Raw country label.
        
    Returns:
        str: Cleaned uppercase country label, or 'UNKNOWN' if missing.
    """
    if country_text is None or not isinstance(country_text, str):
        return "UNKNOWN"
    cleaned = country_text.strip().upper()
    return cleaned if cleaned else "UNKNOWN"


# ==============================================================================
# 3. UNIFIED RECORD NORMALIZATION CONTAINER
# ==============================================================================

class NormalizedRecord:
    """
    A rich data structure encapsulating both raw and normalized fields for an entity.
    Precomputes token sets, character n-grams, and postal codes once to avoid
    redundant calculations across thousands of pairwise comparisons.
    """
    __slots__ = (
        "entity_id",
        "raw_name",
        "raw_address",
        "raw_country",
        "norm_name",
        "core_name",
        "legal_suffix",
        "norm_address",
        "norm_country",
        "name_tokens",
        "sorted_name_key",
        "address_tokens",
        "numeric_tokens",
        "postal_code",
        "name_trigrams",
        "address_trigrams"
    )

    def __init__(
        self,
        entity_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str]
    ):
        """
        Initializes and precomputes all normalized attributes for a business entity.
        
        Args:
            entity_id (str): Unique entity ID (e.g., 'S1-00001').
            business_name (str): Raw name of the business entity.
            business_address (str): Raw address of the business entity.
            country (str): Country label.
        """
        # Store original raw attributes
        self.entity_id: str = entity_id.strip() if entity_id else ""
        self.raw_name: str = business_name if isinstance(business_name, str) else ""
        self.raw_address: str = business_address if isinstance(business_address, str) else ""
        self.raw_country: str = country if isinstance(country, str) else ""

        # Normalize country open-set
        self.norm_country: str = normalize_country(self.raw_country)

        # Normalize business name
        cleaned_name = remove_punctuation(clean_unicode_and_case(self.raw_name))
        self.norm_name: str = cleaned_name
        self.core_name, self.legal_suffix = isolate_legal_suffixes(self.raw_name)

        # Name token collections and word-order invariant key
        tokens_list = self.norm_name.split()
        self.name_tokens: Set[str] = set(tokens_list)
        # Sorted name key allows matching "Walmart Supercenter" with "Supercenter Walmart"
        self.sorted_name_key: str = " ".join(sorted(tokens_list))

        # Normalize business address
        self.norm_address: str = normalize_address_text(self.raw_address)
        self.address_tokens: Set[str] = set(self.norm_address.split())

        # Extract numeric tokens and postal codes from address
        self.numeric_tokens: Set[str] = set(extract_numeric_tokens(self.norm_address))
        self.postal_code: str = extract_postal_code(self.norm_address)

        # Precompute character trigrams for fast string similarity
        self.name_trigrams: Set[str] = generate_char_ngrams(self.norm_name, n=3)
        self.address_trigrams: Set[str] = generate_char_ngrams(self.norm_address, n=3)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<NormalizedRecord id='{self.entity_id}' "
            f"name='{self.norm_name[:25]}' country='{self.norm_country}'>"
        )
