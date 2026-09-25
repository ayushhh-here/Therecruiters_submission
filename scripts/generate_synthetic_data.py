#!/usr/bin/env python3
# ==============================================================================
# Amazon ML Challenge - Business Entity Resolution
# Script: scripts/generate_synthetic_data.py
#
# Description:
#   Generates realistic synthetic datasets matching the exact schema, file structure,
#   and noise characteristics described in the official challenge statement:
#     - Tab-separated values (.tsv) with explicit headers.
#     - Training Set:
#         * dataset/train/train_source1.tsv (Reference source)
#         * dataset/train/train_source2.tsv
#         * dataset/train/train_source3.tsv
#         * dataset/train/train_ground_truth.tsv
#     - Test Set:
#         * dataset/test/test_source1.tsv
#         * dataset/test/test_source2.tsv
#         * dataset/test/test_source3.tsv
#     - Noise patterns:
#         * Name abbreviations (Corp, Inc, Pvt Ltd, Co)
#         * Transliteration & spelling variations
#         * Address abbreviations (Rd, St, Ave, Blvd, Opp, Nr)
#         * Missing PIN / postal codes
#         * Landmark-based addresses ("Near Metro Station")
#         * Component reordering
#     - Open-Set Country Validation:
#         * Training contains: US, India
#         * Test set contains: France, US, India (open-set country test)
#     - Match Multiplicity:
#         * Singletons (0 matches)
#         * 1-to-1 matches
#         * 1-to-Many matches across both Source 2 and Source 3
# ==============================================================================

# Standard library imports
import os
import random
from typing import List, Tuple, Dict, Any


def ensure_dirs(base_dir: str = "dataset") -> Tuple[str, str]:
    """
    Creates the required directory structure for train and test splits.
    
    Args:
        base_dir (str): Base dataset root.
        
    Returns:
        Tuple[str, str]: (train_dir, test_dir)
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir


# Sample realistic business templates
SAMPLE_BUSINESSES = [
    # US Businesses
    ("Blue Ridge Mountain Bakery", "452 Pine Ridge Road, Suite 101, Asheville, NC 28804", "US"),
    ("Apex Dynamics Global", "1024 Silicon Parkway, Building B, Austin, TX 78701", "US"),
    ("Emerald Bay Dental Clinic", "789 Coastal Highway, Santa Monica, CA 90401", "US"),
    ("Silverline Logistics LLC", "55 Industrial Boulevard, Chicago, IL 60607", "US"),
    ("Starlight Cafe & Bistro", "12 Maple Street, Cambridge, MA 02138", "US"),
    ("Summit Peak Outdoor Gear", "300 Mountain View Ave, Denver, CO 80202", "US"),
    ("Pinnacle Financial Advisory", "500 Wall Street, Floor 14, New York, NY 10005", "US"),
    ("Beacon Hill Pharmacy", "210 Charles Street, Boston, MA 02114", "US"),
    ("Golden Gate Tech Solutions", "88 Market St, San Francisco, CA 94105", "US"),
    ("Prairie Wind Renewable Energy", "910 Windmill Lane, Des Moines, IA 50309", "US"),
    
    # India Businesses
    ("Kaveri Sweet Delights", "14 MG Road, Opposite Metro Station, Indiranagar, Bangalore 560038", "India"),
    ("Shree Balaji Textiles Pvt Ltd", "88 Ring Road, Near Surat Railway Station, Surat 395002", "India"),
    ("Sai Krupa Multi Specialty Hospital", "24/B Sector 17, Vashi, Navi Mumbai 400703", "India"),
    ("Sri Krishna Electricals", "5/10 Anna Salai, Mount Road, Chennai 600002", "India"),
    ("Royal Heritage Palace Hotel", "Jaleb Chowk, Near Hawa Mahal, Jaipur 302002", "India"),
    ("Apex Diagnostic & Path Lab", "Plot 45, Sector 18, Gurgaon, Haryana 122001", "India"),
    ("Ganga Valley Ayurvedic Care", "Tapovan, Badrinath Road, Rishikesh, Uttarakhand 249192", "India"),
    ("Deccan Auto Spares", "Shop 12, Rani Gunj, Secunderabad, Hyderabad 500003", "India"),
    ("Surya Bio Organic Fertilizers", "Industrial Estate, Phase 2, Peenya, Bangalore 560058", "India"),
    ("Punjab Grill Delight Dhaba", "GT Road, Near Haveli, Jalandhar, Punjab 144001", "India"),
    
    # France Businesses (Appearing in Test Set for Open-Set Country Validation)
    ("Boulangerie Patisserie Saint-Germain", "14 Rue de Buci, Saint-Germain-des-Prés, 75006 Paris", "France"),
    ("Cabinet Dentaire du Marais", "35 Rue des Francs-Bourgeois, 75004 Paris", "France"),
    ("Bistro Le Petit Chatelet", "39 Rue de la Bûcherie, 75005 Paris", "France"),
    ("Atelier de Couture Parisienne", "12 Rue de la Paix, 75002 Paris", "France"),
    ("Pharmacie de la Mairie", "8 Place de la République, 69002 Lyon", "France"),
    ("Optique Lafayette", "24 Boulevard Haussmann, 75009 Paris", "France"),
    ("Librairie Les Arpenteurs", "9 Rue Marcadet, 75018 Paris", "France"),
    ("Garage Automobile Provençal", "150 Avenue du Prado, 13008 Marseille", "France")
]


def corrupt_business_name(name: str) -> str:
    """
    Applies realistic perturbations to a business name (abbreviations, typos, suffixes).
    
    Args:
        name (str): Original business name.
        
    Returns:
        str: Perturbed business name.
    """
    transformations = [
        # Suffix variation
        lambda s: s.replace(" Pvt Ltd", " Private Limited").replace(" LLC", " Corp"),
        lambda s: s + " & Co." if not ("LLC" in s or "Pvt" in s) else s,
        # Typos
        lambda s: s.replace("Baker", "Bakery").replace("Clinic", "Clinik"),
        lambda s: s.replace("Technologies", "Tech").replace("Solutions", "Solns"),
        # Symbol substitution
        lambda s: s.replace("&", "and").replace("and", "&"),
        # Case / Whitespace
        lambda s: s.upper(),
        lambda s: s.lower()
    ]
    fn = random.choice(transformations)
    return fn(name)


def corrupt_address(address: str) -> str:
    """
    Applies realistic perturbations to an address (abbreviations, missing postal code, reordering).
    
    Args:
        address (str): Original address string.
        
    Returns:
        str: Perturbed address string.
    """
    transformations = [
        # Abbreviation swaps
        lambda a: a.replace(" Road", " Rd.").replace(" Street", " St."),
        lambda a: a.replace(" Avenue", " Ave").replace(" Boulevard", " Blvd"),
        lambda a: a.replace("Near ", "Nr. ").replace("Opposite ", "Opp. "),
        lambda a: a.replace("Suite ", "Ste #").replace("Floor ", "Fl. "),
        # Dropping postal code
        lambda a: " ".join([tok for tok in a.split() if not (tok.isdigit() and len(tok) in (5, 6))]),
        # Landmark additions
        lambda a: a + ", Near Old SBI Branch",
        # Case variation
        lambda a: a.lower()
    ]
    fn = random.choice(transformations)
    return fn(address)


def generate_dataset_split(
    businesses_list: List[Tuple[str, str, str]],
    split_name: str,
    output_dir: str,
    is_train: bool = True
) -> None:
    """
    Generates Source 1, Source 2, Source 3, and Ground Truth TSVs.
    
    Args:
        businesses_list: Pool of template businesses.
        split_name: 'train' or 'test'.
        output_dir: Target output folder.
        is_train: Boolean indicating whether to produce ground truth.
    """
    s1_rows = []
    s2_rows = []
    s3_rows = []
    ground_truth_rows = []

    s2_counter = 1
    s3_counter = 1

    for idx, (name, addr, country) in enumerate(businesses_list, start=1):
        s1_id = f"S1-{idx:05d}"
        s1_rows.append((s1_id, name, addr, country))

        # Determine match multiplicity:
        # 25% singletons (0 matches)
        # 50% 1 match (either in S2 or S3)
        # 25% multi-match (in both S2 and S3)
        rand_val = random.random()
        matched_ids = []

        if rand_val >= 0.25:  # Not a singleton
            # Match in Source 2
            if random.random() < 0.8:
                s2_id = f"S2-{s2_counter:05d}"
                s2_counter += 1
                s2_name = corrupt_business_name(name)
                s2_addr = corrupt_address(addr)
                s2_rows.append((s2_id, s2_name, s2_addr, country))
                matched_ids.append(s2_id)

            # Match in Source 3 (multi-match)
            if random.random() < 0.6 or len(matched_ids) == 0:
                s3_id = f"S3-{s3_counter:05d}"
                s3_counter += 1
                s3_name = corrupt_business_name(name)
                s3_addr = corrupt_address(addr)
                s3_rows.append((s3_id, s3_name, s3_addr, country))
                matched_ids.append(s3_id)

        # Also generate some distractor records in S2 and S3 that do not match S1
        if random.random() < 0.3:
            s2_id = f"S2-{s2_counter:05d}"
            s2_counter += 1
            s2_rows.append((s2_id, f"Unrelated Enterprise {s2_counter}", "100 Random Industrial Park", country))

        if random.random() < 0.3:
            s3_id = f"S3-{s3_counter:05d}"
            s3_counter += 1
            s3_rows.append((s3_id, f"Different Business {s3_counter}", "500 Another Bypass Road", country))

        if is_train:
            matches_str = ",".join(matched_ids)
            ground_truth_rows.append((s1_id, matches_str))

    # Shuffle S2 and S3 rows to test robust retrieval
    random.shuffle(s2_rows)
    random.shuffle(s3_rows)

    # Write files with tab-separation
    def write_tsv(filepath: str, rows: List[Tuple[str, str, str, str]]):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for eid, b_name, b_addr, ctry in rows:
                f.write(f"{eid}\t{b_name}\t{b_addr}\t{ctry}\n")

    s1_path = os.path.join(output_dir, f"{split_name}_source1.tsv")
    s2_path = os.path.join(output_dir, f"{split_name}_source2.tsv")
    s3_path = os.path.join(output_dir, f"{split_name}_source3.tsv")

    write_tsv(s1_path, s1_rows)
    write_tsv(s2_path, s2_rows)
    write_tsv(s3_path, s3_rows)
    print(f" Saved: {s1_path} ({len(s1_rows)} rows)")
    print(f" Saved: {s2_path} ({len(s2_rows)} rows)")
    print(f" Saved: {s3_path} ({len(s3_rows)} rows)")

    if is_train:
        gt_path = os.path.join(output_dir, "train_ground_truth.tsv")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, m_ids in ground_truth_rows:
                f.write(f"{s1_id}\t{m_ids}\n")
        print(f" Saved: {gt_path} ({len(ground_truth_rows)} rows)")


def generate_unique_business_catalog(n_entities: int = 120) -> List[Tuple[str, str, str]]:
    """
    Generates a catalog of unique, realistic businesses across US, India, and France.
    """
    prefixes = [
        "Apex", "Summit", "Beacon", "Pinnacle", "Golden", "Silver", "Emerald",
        "Blue Ridge", "Pacific", "Atlantic", "Metro", "Royal", "Imperial",
        "Horizon", "Vanguard", "Zenith", "Sunrise", "Evergreen", "Cascade", "Crest"
    ]
    industries = [
        "Tech Solutions", "Healthcare", "Dental Clinic", "Logistics", "Bakery",
        "Cafe & Bistro", "Textiles", "Hospital", "Electricals", "Auto Spares",
        "Consulting Group", "Pharma Labs", "Organic Foods", "Financial Advisory",
        "Hardware Store", "Fitness Studio", "Legal Services", "Security Systems"
    ]
    cities_us = [("New York", "NY", "10001"), ("Austin", "TX", "78701"), ("Seattle", "WA", "98101"), ("Chicago", "IL", "60601")]
    cities_in = [("Bangalore", "Karnataka", "560001"), ("Mumbai", "Maharashtra", "400001"), ("Delhi", "Delhi", "110001"), ("Hyderabad", "Telangana", "500001")]
    cities_fr = [("Paris", "IDF", "75001"), ("Lyon", "ARA", "69001"), ("Marseille", "PAC", "13001"), ("Bordeaux", "NAQ", "33000")]

    businesses = []
    idx = 0
    while len(businesses) < n_entities:
        p = prefixes[idx % len(prefixes)]
        ind = industries[(idx // len(prefixes)) % len(industries)]
        num = (idx * 37 + 101) % 999 + 1
        
        # Distribute countries: 45% US, 45% India, 10% France
        mod_country = idx % 10
        if mod_country in [0, 1, 2, 3]:
            city, state, pin = cities_us[idx % len(cities_us)]
            name = f"{p} {ind} LLC"
            addr = f"{num} Main Street, Suite {idx % 20 + 1}, {city}, {state} {pin}"
            country = "US"
        elif mod_country in [4, 5, 6, 7]:
            city, state, pin = cities_in[idx % len(cities_in)]
            name = f"{p} {ind} Pvt Ltd"
            addr = f"Plot {num}, MG Road, Near Metro Pillar {idx % 50}, {city} {pin}"
            country = "India"
        else:
            city, state, pin = cities_fr[idx % len(cities_fr)]
            name = f"{p} {ind} SARL"
            addr = f"{num} Rue de la Paix, {pin} {city}"
            country = "France"

        businesses.append((name, addr, country))
        idx += 1

    return businesses


def main():
    """Main data generation script."""
    random.seed(42)
    train_dir, test_dir = ensure_dirs("dataset")

    print("=" * 70)
    print(" Amazon ML Challenge - Generating Synthetic Datasets")
    print("=" * 70)

    all_catalog = generate_unique_business_catalog(150)

    # Filter templates: Training contains US and India only
    train_pool = [b for b in all_catalog if b[2] in ("US", "India")][:100]
    print("\n--- Generating Training Dataset (US, India) ---")
    generate_dataset_split(train_pool, "train", train_dir, is_train=True)

    # Test set contains US, India, AND France (open-set!)
    test_pool = all_catalog[50:134]  # 84 test entities including France
    print("\n--- Generating Test Dataset (US, India, France) ---")
    generate_dataset_split(test_pool, "test", test_dir, is_train=False)

    print("\n" + "=" * 70)
    print(" Synthetic datasets successfully generated in 'dataset/'!")
    print("=" * 70)


if __name__ == "__main__":
    main()
