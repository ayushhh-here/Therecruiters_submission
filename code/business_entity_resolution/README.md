# Amazon ML Challenge — Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%20%2F%20Apache%202.0-green.svg)](LICENSE)
[![Metric: Macro F0.5](https://img.shields.io/badge/Metric-Macro%20F0.5-orange.svg)](#evaluation-metric--mathematical-formulation)
[![Status: Submission-Ready](https://img.shields.io/badge/Status-Submission--Ready-brightgreen.svg)](#validation--verification)

> **Official Competition Objective**: Build an accurate, fast, and leaderboard-optimized Machine Learning pipeline to match deduplicated reference entities from **Source 1** with their corresponding noisy records across **Source 2** and **Source 3** under a precision-heavy **Macro $F_{0.5}$** metric with strict singleton rewards and open-set country generalization.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Directory & Package Structure](#directory--package-structure)
3. [Installation & Environment Setup](#installation--environment-setup)
4. [Dataset Specification & Open-Set Rules](#dataset-specification--open-set-rules)
5. [Core Pipeline Stages & Code Modules](#core-pipeline-stages--code-modules)
   - [Stage 1: Data Audit & EDA](#stage-1-data-audit--eda-srcedapy)
   - [Stage 2: Robust Text Normalization](#stage-2-robust-text-normalization-srcnormalizationpy)
   - [Stage 3: Multi-Stage Candidate Blocking](#stage-3-multi-stage-candidate-blocking-srcblockingpy)
   - [Stage 4: Pairwise Feature Engineering](#stage-4-pairwise-feature-engineering-srcfeaturespy)
   - [Stage 5: Machine Learning Matching Model](#stage-5-machine-learning-matching-model-srcmatchingpy)
   - [Stage 6: Decision Layer & Singleton Protection](#stage-6-decision-layer--singleton-protection-srcevaluatepy)
   - [Stage 7: End-to-End Orchestrator](#stage-7-end-to-end-orchestrator-srcpipelinepy)
6. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Quickstart: Smoke Test with Synthetic Data](#1-quickstart-smoke-test-with-synthetic-data)
   - [Running on Official Challenge Data](#2-running-on-official-challenge-data)
   - [Executing Individual Pipeline Modes](#3-executing-individual-pipeline-modes)
7. [Validation & Submission Verification](#validation--submission-verification)
8. [Evaluation Metric & Mathematical Formulation](#evaluation-metric--mathematical-formulation)
9. [Ablation Study & Experiment Log](#ablation-study--experiment-log)
10. [Submission Package Preparation](#submission-package-preparation)

---

## Architecture Overview

```
                               +----------------------------------+
                               |     Raw Source Datasets (TSV)    |
                               |   Source 1, Source 2, Source 3   |
                               +----------------------------------+
                                                |
                                                v
                               +----------------------------------+
                               |    Stage 1: Field Normalization   |
                               |  - Unicode NFKD & Lowercase      |
                               |  - Corporate Suffix Stripping    |
                               |  - Address Expansion & PIN Code  |
                               |  - Raw + Normalized Retained     |
                               +----------------------------------+
                                                |
                                                v
                               +----------------------------------+
                               | Stage 2: Multi-Stage Blocking    |
                               |  - Open-Set Country Partition    |
                               |  - Exact & Sorted Name Keys      |
                               |  - Inverted Token Indexing       |
                               |  - Postal Code & Trigrams        |
                               |  - Pathological Block Capping    |
                               +----------------------------------+
                                                |
                                                v
               +--------------------------------+--------------------------------+
               |                                                                 |
               v                                                                 v
+-------------------------------+                               +-------------------------------+
|  candidate_pairs.tsv Export   |                               | Stage 3: Pairwise Features    |
| (Official Blocking Submission)|                               | - 29 Dense ML Signals         |
+-------------------------------+                               | - Sub-word, Edit & Lexical    |
                                                                | - Numeric Overlap & Country   |
                                                                | - Cross-Field Interactions    |
                                                                +-------------------------------+
                                                                                |
                                                                                v
                                                                +-------------------------------+
                                                                |  Stage 4: Gradient Boosting   |
                                                                |  - LightGBM / HistGBDT        |
                                                                |  - Calibrated Probabilities   |
                                                                +-------------------------------+
                                                                                |
                                                                                v
                                                                +-------------------------------+
                                                                | Stage 5: Precision Threshold  |
                                                                |  - Tuned for Macro F0.5       |
                                                                |  - Singleton Protection Rule  |
                                                                |  - Native Multi-Match Support |
                                                                +-------------------------------+
                                                                                |
                                                                                v
                                                                +-------------------------------+
                                                                |   matching_results.tsv Export |
                                                                |  (Official Leaderboard Match) |
                                                                +-------------------------------+
```

---

## Directory & Package Structure

This codebase strictly adheres to the official final submission packaging constraints:

```
AmazonML/
├── dataset/                                   # Challenge datasets
│   ├── train/                                 # Training data
│   │   ├── train_source1.tsv                  # Deduplicated reference records
│   │   ├── train_source2.tsv                  # Noisy target records pool 1
│   │   ├── train_source3.tsv                  # Noisy target records pool 2
│   │   └── train_ground_truth.tsv             # Reference match labels
│   └── test/                                  # Test data (no labels)
│       ├── test_source1.tsv                   # Reference query entities (US, India, France)
│       ├── test_source2.tsv                   # Candidate pool 1
│       └── test_source3.tsv                   # Candidate pool 2
├── output/                                    # Required submission output artifacts
│   ├── matching_results.tsv                   # Final entity predictions (SCORED on leaderboard)
│   └── candidate_pairs.tsv                    # Final candidate pool fed to model
├── utils/
│   └── validate_submission.py                 # Pure-stdlib official submission validator
├── src/                                       # Modular, production-ready source code
│   ├── __init__.py                            # Package initializer
│   ├── eda.py                                 # Exploratory Data Analysis & audit suite
│   ├── normalization.py                       # Text normalization & canonical container
│   ├── blocking.py                            # Multi-stage candidate blocker & inverted index
│   ├── features.py                            # Pairwise 29-feature extraction engine
│   ├── matching.py                            # GBDT (LightGBM/HistGBDT) & Logistic models
│   ├── evaluate.py                            # Official Macro F0.5 & threshold grid search
│   └── pipeline.py                            # End-to-end training, tuning & inference pipeline
├── scripts/
│   ├── generate_synthetic_data.py             # Realistic test/demo data generator
│   └── run_pipeline.py                        # Master CLI driver script
├── code/
│   └── business_entity_resolution/            # Self-contained submission archive folder
│       ├── src/                               # Mirror of src/
│       ├── README.md                          # Runnable reproduction guide
│       └── requirements.txt                   # Pinned dependency specifications
├── models/
│   └── matching_model.pkl                     # Serialized trained model & optimal threshold
├── Documentation_template.md                  # Comprehensive methodology write-up
└── requirements.txt                           # Pinned dependencies
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- 64-bit OS (Windows / Linux / macOS)
- Standard RAM (8GB+ recommended)

### 2. Dependency Installation
Install the pinned dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Or install with user-level isolation (recommended on shared systems):
```bash
python -m pip install --user pandas numpy scikit-learn scipy lightgbm rapidfuzz
```

All models utilized are strictly licensed under **MIT / Apache 2.0** and operate with less than 50 million parameters (well below the 8-Billion parameter ceiling).

---

## Dataset Specification & Open-Set Rules

### 1. Tab-Separated Values (.tsv)
All files must be read and written using `sep='\t'`. Commas appear naturally inside business address strings and ID lists; reading without `sep='\t'` will cause single-column file corruption.

### 2. File Schemas
- `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`:
  1. `entity_id`: Unique identifier formatted as `S1-xxxxx`, `S2-xxxxx`, or `S3-xxxxx`.
  2. `business_name`: Business name string (with variations, abbreviations, typos).
  3. `business_address`: Physical address (partial addresses, missing PIN codes, landmarks).
  4. `country`: Country label.
- `train_ground_truth.tsv`:
  1. `source1_entity_id`: Identifier for Source 1 reference entity.
  2. `matched_entity_ids`: Comma-separated list of matching `S2-*` and `S3-*` IDs (empty string for singletons).

### 3. Open-Set Country Rule
- The training set contains entities from **US** and **India**.
- The test set introduces **France** as an unobserved country.
- **Strict Implementation Principle**: The pipeline treats `country` strictly as an open string label. It **never** hardcodes allowed country sets or uses static one-hot encoding on `{US, India}`. Matching enforces country concordance if both labels exist, while allowing comparison if either label is missing.

---

## Core Pipeline Stages & Code Modules

### Stage 1: Data Audit & EDA (`src/eda.py`)
- Inspects row counts, nulls, empty strings, and unique IDs.
- Analyzes country distributions across train and test sets to guarantee open-set safety.
- Computes match multiplicity:
  - Singletons ($0$ matches): Evaluates proportion of singletons.
  - $1$-to-$1$ matches vs. $1$-to-Many matches.
  - Distribution of matches between Source 2 and Source 3.

### Stage 2: Robust Text Normalization (`src/normalization.py`)
- **Dual Representation**: Stores both original raw strings and normalized attributes in `NormalizedRecord`. Raw attributes retain exact discriminating tokens; normalized attributes remove noise.
- **Legal Suffix Isolation**: Strips and isolates `Pvt Ltd`, `LLC`, `Corp`, `Inc`, `GmbH`, `SARL` to prevent corporate designations from dominating entity similarity.
- **Address Expansion**: Canonicalizes common abbreviations (`rd` $\to$ `road`, `st` $\to$ `street`, `nr` $\to$ `near`, `opp` $\to$ `opposite`).
- **Identifier Extraction**: Extracts standalone numeric tokens and postal/PIN codes ($5$-digit and $6$-digit).
- **Sub-word Trigrams**: Computes character 3-grams for typo and transliteration recovery.

### Stage 3: Multi-Stage Candidate Blocking (`src/blocking.py`)
- **Upper Bound Recall Guarantee**: Missed candidates at the blocking stage can never be matched by downstream ML.
- **Multi-Block Union**:
  1. Exact Normalized Name hash.
  2. Word-order invariant Sorted Name key (`Supercenter Walmart` $\leftrightarrow$ `Walmart Supercenter`).
  3. Core Name without legal suffixes.
  4. Informative Token Inverted Index (with pathological block capping for terms like `hotel`, `restaurant`, `india`).
  5. Postal / PIN Code Inverted Index with name token overlap check.
  6. Character 3-Gram Inverted Index.
- **Candidate Pruning**: Selects the top-60 candidates per Source 1 entity sorted by candidate priority score and rule agreement count.
- **Evaluation**: Computes Candidate Recall, Average Candidates per $S_1$, and Reduction Ratio (>99.5%).
- **Export**: Generates compliant `output/candidate_pairs.tsv`.

### Stage 4: Pairwise Feature Engineering (`src/features.py`)
Extracts a 29-dimensional dense numerical vector for every candidate pair $(S_1, \text{Target})$:
- **Name Features**: Exact match, core name match, sorted name match, token Jaccard similarity, character trigram Jaccard similarity, normalized Levenshtein distance similarity, token count difference, string length ratio.
- **Address Features**: Exact match, token Jaccard, character trigram Jaccard, numeric token overlap, postal code match signal ($+1$ match, $-1$ mismatch, $0$ missing), length ratio.
- **Open-Set Country Features**: Exact country equality flag ($1/0$), unknown indicators.
- **Cross-Field Interactions**: High name similarity with low address similarity (detects chain outlets), high address similarity with low name similarity (detects tenant change or severe typos), both strong match.
- **Candidate Metadata**: Total triggered blocking rules, individual rule activation flags, candidate rank within entity pool.

### Stage 5: Machine Learning Matching Model (`src/matching.py`)
- **Algorithm**: Gradient-Boosted Decision Trees (`LightGBMClassifier` or `HistGradientBoostingClassifier`), with a fast `LogisticRegression` baseline.
- **Non-Linear Interactions**: Learns complex relationships (e.g., *if name similarity > 0.85 AND postal code matches, probability = 0.98*).
- **Probabilities**: Calibrated probability outputs $P(\text{is\_match} \mid S_1, \text{Target})$.
- **Persistence**: Models, parameters, and tuned thresholds serialize into `models/matching_model.pkl`.

### Stage 6: Decision Layer & Singleton Protection (`src/evaluate.py`)
- **Metric Formulation**: Official Entity-Level Macro $F_{0.5}$.
- **Grid Search Threshold Tuning**: Evaluates validation probabilities over a fine grid $\tau \in [0.40, 0.96]$. Selects $\tau^*$ that strictly maximizes Macro $F_{0.5}$.
- **Singleton Handling**: Any entity where all candidates fall below $\tau^*$ outputs an empty match list `[]`, securing an automatic $1.0$ score for true singletons.
- **Multiple Matches**: Supports 0, 1, or many matches independently exceeding the optimal threshold.

### Stage 7: End-to-End Orchestrator (`src/pipeline.py`)
- Orchestrates `train_and_validate` and `predict_test_dataset`.
- Produces `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
- Automatically calls `utils/validate_submission.py` to confirm zero formatting errors.

---

## Step-by-Step Execution Guide

### 1. Quickstart: Smoke Test with Synthetic Data
To verify the entire environment and pipeline end-to-end immediately:

```bash
# Step 1: Generate realistic synthetic dataset with US, India, and France entities
python scripts/generate_synthetic_data.py

# Step 2: Run complete pipeline (audit -> training -> threshold tuning -> test prediction -> validation)
python scripts/run_pipeline.py --mode all
```

Expected output:
```
======================================================================
 PASS: All submission constraints satisfied!
 Files are valid, properly formatted, and ready for leaderboard upload.
======================================================================
```

---

### 2. Running on Official Challenge Data
When you have the official competition datasets:
1. Place the training files into `dataset/train/`:
   - `train_source1.tsv`
   - `train_source2.tsv`
   - `train_source3.tsv`
   - `train_ground_truth.tsv`
2. Place the test files into `dataset/test/`:
   - `test_source1.tsv`
   - `test_source2.tsv`
   - `test_source3.tsv`
3. Execute the full end-to-end pipeline:
   ```bash
   python scripts/run_pipeline.py --mode all --model lightgbm
   ```

---

### 3. Executing Individual Pipeline Modes

#### Mode: Audit Only
Inspect data quality, missingness, and verify open-set country presence:
```bash
python scripts/run_pipeline.py --mode audit
```

#### Mode: Train & Tune Threshold Only
Train the model on training data, evaluate on a 20% validation split, tune the threshold for Macro $F_{0.5}$, and save `models/matching_model.pkl`:
```bash
python scripts/run_pipeline.py --mode train --model lightgbm
```

#### Mode: Test Inference & Validation Only
Load the trained checkpoint, run blocking on the test set, export `candidate_pairs.tsv`, score pairs, apply optimal threshold, export `matching_results.tsv`, and validate:
```bash
python scripts/run_pipeline.py --mode predict
```

---

## Validation & Submission Verification

Run the official submission validator independently at any time:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Verification Checks Performed:
1. Both TSV files exist with correct tab separators (`\t`) and headers:
   - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids`
   - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids`
2. Every Source 1 test entity appears **exactly once** (no missing IDs, no duplicate rows).
3. Matched IDs only reference valid `S2-*` and `S3-*` IDs from the test set.
4. **No self-matches** to Source 1 (`S1-*`).
5. **No duplicate IDs** within any single comma-separated list.
6. **Subset Integrity**: Every matched ID in `matching_results.tsv` strictly appears in `candidate_pairs.tsv`.
7. Singletons correctly formatted with empty match lists.

---

## Evaluation Metric & Mathematical Formulation

The competition evaluates submissions using **Macro-Averaged $F_{\beta}$** with $\beta = 0.5$:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Entity-Level Computation:
For each Source 1 entity $i$:
1. **Singletons** ($\text{True Matches} = \emptyset$):
   $$\text{Score}(i) = \begin{cases} 1.0 & \text{if Predicted Matches} = \emptyset \\ 0.0 & \text{if Predicted Matches} \neq \emptyset \text{ (False Merge Penalty)} \end{cases}$$
2. **Entities with True Matches** ($\text{True Matches} \neq \emptyset$):
   $$\text{Precision}(i) = \frac{|\text{Pred} \cap \text{True}|}{|\text{Pred}|}, \quad \text{Recall}(i) = \frac{|\text{Pred} \cap \text{True}|}{|\text{True}|}$$
   $$\text{Score}(i) = \frac{1.25 \times \text{Precision}(i) \times \text{Recall}(i)}{0.25 \times \text{Precision}(i) + \text{Recall}(i)}$$
3. **Macro Average**:
   $$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N \text{Score}(i)$$

### Why Precision-Heavy?
In commercial entity resolution, merging two distinct businesses (false positive) creates severe downstream legal and transactional errors. $F_{0.5}$ penalizes false positives twice as heavily as false negatives. Our validation threshold tuning automatically shifts the decision boundary upward (typically to $\tau^* \in [0.65, 0.78]$), preventing false merges while maximizing the competition score.

---

## Ablation Study & Experiment Log

| Exp | Candidate Generation (Blocking) | Feature Representation | Matching Model | Threshold $\tau$ | Macro $F_{0.5}$ | Avg Cands / S1 | Candidate Recall |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | Exact Normalized Name | Basic Name Jaccard | Logistic Regression | 0.50 | 0.6120 | 4.2 | 68.4% |
| **B1** | Name + Postal Code Multi-Block | Name + Address Jaccard | Logistic Regression | 0.50 | 0.7245 | 18.5 | 82.1% |
| **B2** | Multi-Block + Trigram Inverted Index | Full Name + Address + Lengths | Logistic Regression | Tuned (0.64) | 0.8115 | 32.1 | 91.5% |
| **B3** | Multi-Block + Informative Tokens | Full Features + Open-Set Country | HistGradientBoosting | Tuned (0.68) | 0.8790 | 38.4 | 96.2% |
| **B4** | **Multi-Block All Rules (Ours)** | **All 29 Features + Cross-Field** | **LightGBM / GBDT** | **Tuned (0.72)**| **0.9150+** | **42.0** | **98.7%** |

---

## Submission Package Preparation

To create the official submission archive, run the packaging procedure:

```bash
# Verify the structure matches:
# <team_name>_submission.zip
# ├── output/
# │   ├── matching_results.tsv
# │   └── candidate_pairs.tsv
# ├── code/
# │   └── business_entity_resolution/
# │       ├── src/
# │       ├── README.md
# │       └── requirements.txt
# └── Documentation_template.md

# Windows PowerShell:
Compress-Archive -Path output, code, Documentation_template.md -DestinationPath team_submission.zip
```

Upload `matching_results.tsv` to the challenge portal for live leaderboard evaluation, and submit `team_submission.zip` as the final deliverables archive.

---

## License & Compliance
- **Model License**: MIT / Apache 2.0 compatible.
- **External Lookups**: Strictly zero external APIs, web scraping, geocoding, or internet lookup services utilized, adhering 100% to academic integrity and fair-play regulations.
