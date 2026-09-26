#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Module: src/normalization.py
#
# Description:
#   High-Performance, Multilingual, Script-Aware Text Normalization.
#   Implements Phase 4 of the architectural requirements:
#     - Preserves original Unicode text (NEVER destructively converts to ASCII).
#     - Maintains multiple representations:
#         1. Raw Unicode string
#         2. Unicode-normalized (NFKC) & case-folded string
#         3. Punctuation/whitespace-normalized Unicode string
#         4. Script-aware transliteration (e.g., Latin accent stripping)
#     - Multi-script support: Latin, Extended Latin (French), Devanagari,
#       Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Cyrillic, Arabic.
#     - Multi-lingual legal suffix stripping (English, French, Devanagari).
#     - Multi-lingual address abbreviation standardization.
#     - Script-aware postal code and numeric extraction (handling Indic numerals).
#     - Character 3-gram generation preserving Unicode code points.
# ==============================================================================

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Any, Optional

# ==============================================================================
# 1. DICTIONARIES & PATTERNS FOR CLEANING
# ==============================================================================

# Multilingual business legal suffixes: English, French, and Devanagari
LEGAL_SUFFIXES: Set[str] = {
    # English / International legal suffixes
    "pvt", "private", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llc", "llp", "pllc",
    "gmbh", "ag", "enterprises", "enterprise", "holdings", "holding", "group",
    "services", "solutions", "technologies", "consulting", "industries",
    # French legal structures (observed in challenge test set)
    "sarl", "sas", "sa", "eurl", "sci", "snc", "ste", "societe", "ets",
    # Devanagari legal suffixes (observed in challenge train/test data)
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "कंपनी", "इंटरप्राइजेज", "कॉरपोरेशन",
    "उद्योग", "ट्रेडर्स"
}

# Multilingual address abbreviations mapped to canonical forms
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    # English Road / Street types
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

    # French Address types
    "r": "rue",
    "rue": "rue",
    "bd": "boulevard",
    "bld": "boulevard",
    "av.": "avenue",
    "pl.": "place",
    "rte": "route",
    "all": "allee",
    "imp": "impasse",
    "che": "chemin",
    "crs": "cours",

    # Unit / Sub-location types
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "rm": "room",
    "dept": "department",
    "sec": "sector",
    "blk": "block",
    "plot": "plot",

    # Indian Landmark prefixes
    "opp": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "jn": "junction",
    "po": "post office"
}

# Precomputed Unicode punctuation & symbol translation table (maps all P* and S* categories to space)
# Preserves letters (L*), combining marks/matras (M*), and numbers (N*)
PUNCT_SYMBOLS_TRANSLATE: Dict[int, str] = {
    i: " " for i in range(0x10000)
    if unicodedata.category(chr(i)).startswith(("P", "S"))
}
RE_MULTIPLE_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
RE_NUMBERS = re.compile(r"\b\d+\b")

# Indic digit translation map to ASCII 0-9
INDIC_DIGITS_TABLE = str.maketrans(
    "०१२३४५६७८९"  # Devanagari
    "০১২৩৪৫৬৭৮৯"  # Bengali
    "੦੧੨੩੪੫੬੭੮੯"  # Gurmukhi
    "૦૧૨૩૪૫૬૭૮૯"  # Gujarati
    "୦୧୨୩୪୫୬୭୮୯"  # Oriya
    "௦௧௨௩௪௫௬௭௮௯"  # Tamil
    "౦౧౨౩౪౫౬౭౮౯"  # Telugu
    "೦೧೨೩೪೫೬೭೮೯"  # Kannada
    "൦൧൨൩൪൫൬൭൮൯", # Malayalam
    "0123456789" * 9
)


# ==============================================================================
# 2. SCRIPT DETECTION & NORMALIZATION UTILITIES
# ==============================================================================

def detect_primary_script(text: Optional[str]) -> str:
    """Detects the primary Unicode script family of a string."""
    if not text:
        return "UNKNOWN"
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "DEVANAGARI"
        elif 0x0B80 <= cp <= 0x0BFF:
            return "TAMIL"
        elif 0x0C00 <= cp <= 0x0C7F:
            return "TELUGU"
        elif 0x0C80 <= cp <= 0x0CFF:
            return "KANNADA"
        elif 0x0980 <= cp <= 0x09FF:
            return "BENGALI"
        elif 0x0A80 <= cp <= 0x0AFF:
            return "GUJARATI"
        elif 0x0D00 <= cp <= 0x0D7F:
            return "MALAYALAM"
        elif 0x0600 <= cp <= 0x06FF:
            return "ARABIC"
        elif 0x0400 <= cp <= 0x04FF:
            return "CYRILLIC"
    return "LATIN"


def normalize_unicode_and_case(text: Optional[str]) -> str:
    """
    Normalizes Unicode via NFKC and applies Unicode-aware case folding.
    PRESERVES all scripts and code points (no destructive ASCII casting).
    """
    if not text or not isinstance(text, str):
        return ""
    # NFKC normalizes full-width and compatibility characters safely
    nfkc = unicodedata.normalize("NFKC", text)
    # casefold() is Unicode-aware (handles German ß, French accents, Greek, etc.)
    return nfkc.casefold().strip()


def remove_punctuation_unicode(text: str) -> str:
    """
    Replaces punctuation and symbols with spaces while retaining alphanumeric tokens
    and combining marks across ALL Unicode scripts.
    """
    if not text:
        return ""
    spaced = text.translate(PUNCT_SYMBOLS_TRANSLATE)
    return RE_MULTIPLE_WHITESPACE.sub(" ", spaced).strip()


def transliterate_latin_accents(text: str) -> str:
    """
    Strips diacritical marks from Latin characters (e.g., 'é' -> 'e', 'ô' -> 'o')
    WITHOUT corrupting non-Latin scripts (Devanagari, Tamil, etc.).
    Used as an auxiliary representation for accent-invariant matching.
    """
    if not text:
        return ""
    res = []
    for c in text:
        cp = ord(c)
        # Latin block and Latin Extended
        if cp < 0x0370 or (0x1E00 <= cp <= 0x1EFF):
            for sub in unicodedata.normalize("NFKD", c):
                if unicodedata.category(sub) != "Mn":
                    res.append(sub)
        else:
            # Keep non-Latin characters and matras intact
            res.append(c)
    return "".join(res)


def normalize_address_text(address_text: Optional[str]) -> Tuple[str, str]:
    """
    Applies multilingual address normalization.
    Returns:
        Tuple[str, str]: (unicode_normalized_address, transliterated_address)
    """
    base = normalize_unicode_and_case(address_text)
    # Translate Indic digits to ASCII numbers
    base = base.translate(INDIC_DIGITS_TABLE)
    cleaned = remove_punctuation_unicode(base)

    # Expand recognized abbreviations
    tokens = cleaned.split()
    expanded_tokens = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    norm_address = " ".join(expanded_tokens)
    translit_address = transliterate_latin_accents(norm_address)

    return norm_address, translit_address


def isolate_legal_suffixes(name_text: Optional[str]) -> Tuple[str, str]:
    """
    Separates the core business name from corporate legal structure identifiers
    across English, French, and Indic scripts.
    """
    clean_name = remove_punctuation_unicode(normalize_unicode_and_case(name_text))
    tokens = clean_name.split()

    if not tokens:
        return "", ""

    suffix_tokens = []
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        suffix_tokens.insert(0, tokens.pop())

    core_name = " ".join(tokens)
    extracted_suffix = " ".join(suffix_tokens)

    if not core_name:
        core_name = clean_name

    return core_name, extracted_suffix


def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """Extracts standalone numeric tokens (translates Indic digits first)."""
    if not text:
        return []
    translated = text.translate(INDIC_DIGITS_TABLE)
    return RE_NUMBERS.findall(translated)


def extract_postal_code(address_text: Optional[str]) -> str:
    """
    Extracts potential postal or PIN codes from an address string.
    Supports US (5-digit), France (5-digit), and India (6-digit).
    """
    if not address_text:
        return ""
    translated = address_text.translate(INDIC_DIGITS_TABLE)
    candidates = re.findall(r"\b\d{5,6}\b", translated)
    if candidates:
        return candidates[-1]
    return ""


def generate_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Generates a set of character n-grams from a Unicode string.
    Works natively on Latin, Devanagari, Tamil, etc.
    """
    compact = re.sub(r"\s+", "", text, flags=re.UNICODE)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def normalize_country(country_text: Optional[str]) -> str:
    """
    Standardizes country string in an OPEN-SET compliant manner.
    Never hardcodes allowed country sets.
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
    Memory-compact representation of a business entity record.
    Maintains multiple representations (Raw, Unicode Clean, Transliterated)
    without destructive loss.
    """
    __slots__ = (
        "entity_id",
        "raw_name",
        "raw_address",
        "raw_country",
        "norm_name",
        "translit_name",
        "core_name",
        "legal_suffix",
        "norm_address",
        "translit_address",
        "norm_country",
        "name_tokens",
        "sorted_name_key",
        "address_tokens",
        "numeric_tokens",
        "postal_code",
        "name_trigrams",
        "address_trigrams",
        "primary_script"
    )

    def __init__(
        self,
        entity_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str]
    ):
        self.entity_id: str = entity_id.strip() if entity_id else ""
        self.raw_name: str = str(business_name) if business_name else ""
        self.raw_address: str = str(business_address) if business_address else ""
        self.raw_country: str = str(country) if country else ""

        # Open-set country normalization
        self.norm_country: str = normalize_country(self.raw_country)

        # Primary script detection
        self.primary_script: str = detect_primary_script(self.raw_name)

        # Name: Unicode normalization & case-folding
        cleaned_name = remove_punctuation_unicode(normalize_unicode_and_case(self.raw_name))
        self.norm_name: str = cleaned_name
        self.translit_name: str = transliterate_latin_accents(cleaned_name)
        self.core_name, self.legal_suffix = isolate_legal_suffixes(self.raw_name)

        # Name tokens and word-order invariant key
        tokens_list = self.norm_name.split()
        self.name_tokens: Set[str] = set(tokens_list)
        self.sorted_name_key: str = " ".join(sorted(tokens_list))

        # Address: Unicode normalization, abbreviation expansion, and transliteration
        self.norm_address, self.translit_address = normalize_address_text(self.raw_address)
        self.address_tokens: Set[str] = set(self.norm_address.split())

        # Numeric tokens and postal codes
        self.numeric_tokens: Set[str] = set(extract_numeric_tokens(self.norm_address))
        self.postal_code: str = extract_postal_code(self.norm_address)

        # Precomputed character trigrams over Unicode string
        self.name_trigrams: Set[str] = generate_char_ngrams(self.norm_name, n=3)
        self.address_trigrams: Set[str] = generate_char_ngrams(self.norm_address, n=3)

    def __repr__(self) -> str:
        return (
            f"<NormalizedRecord id='{self.entity_id}' "
            f"name='{self.norm_name[:25]}' script='{self.primary_script}' "
            f"country='{self.norm_country}'>"
        )
