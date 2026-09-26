# Final Experiment Summary — Business Entity Resolution

## 1. Challenge Dataset Profile (Measured Real Data)

| Dataset File | Role | Total Records | File Size | Countries Observed | Missing Names | Missing Addresses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | Train Query (S1) | **2,206,821** | 200.3 MB | US (60%), India (40%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2.tsv` | Train Target (S2) | **5,034,616** | 466.6 MB | US (60%), India (40%) | 0 (0.00%) | 168,967 (3.36%) |
| `train_source3.tsv` | Train Target (S3) | **5,285,603** | 480.4 MB | US (60%), India (40%) | 0 (0.00%) | 175,916 (3.33%) |
| `train_ground_truth.tsv` | Ground Truth | **2,206,821** | 121.1 MB | N/A | 0 (0.00%) | 123,247 Singletons (5.58%) |
| `test_source1.tsv` | Test Query (S1) | **1,732,544** | 166.9 MB | India (47%), US (38%), France (15%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2.tsv` | Test Target (S2) | **4,887,273** | 485.9 MB | India (47%), US (38%), France (14%) | 0 (0.00%) | 129,408 (2.65%) |
| `test_source3.tsv` | Test Target (S3) | **5,082,316** | 482.6 MB | India (47%), US (38%), France (14%) | 0 (0.00%) | 136,098 (2.68%) |

### Critical Multilingual Findings:
- Over 180,000 Devanagari tokens and tens of thousands of Telugu, Kannada, Tamil, Bengali, and Gujarati tokens occur in Target records.
- Over 14,000 French accented characters occur in Test files (`é`, `è`, `ô`, `ç`).
- The previous prototype contained an ASCII-casting bug (`.encode('ascii', 'ignore')`) that completely stripped these characters. Our rewritten `src/normalization.py` preserves full Unicode while extracting script-aware transliterated auxiliary representations.

---

## 2. Measured Blocking Recall Curve

Evaluated on 1,907 validation queries against a 206,785 target pool (6,930 true targets + background distractors):

| Metric | Measured Value | Analysis / Decision Rationale |
| :--- | :--- | :--- |
| **Recall@1** | **23.25%** | Exact matches captured at rank 1 |
| **Recall@5** | **65.06%** | Sharp jump capturing high-confidence variants |
| **Recall@10** | **69.57%** | Captures majority of primary matches |
| **Recall@20** | **71.33%** | Captures multi-entity clusters |
| **Recall@50** | **74.36%** | Diminishing returns inflection point |
| **Recall@100** | **77.53%** | High candidate cap |
| **Recall@200** | **79.03%** | Ceiling recall |
| **Median Candidates / Query** | **49.0** | Stable candidate distribution |
| **Zero-Candidate Query Rate** | **1.47%** | Only 1.47% of queries yielded 0 candidates |
| **Mean True Match Rank** | **9.65** | True matches appear on average in top 10 |

**Candidate Budget Decision**:
An adaptive budget of **40 candidates** (reduced to 15 for high-confidence exact matches, expanded up to 50 for ambiguous tokens) strikes the optimal trade-off between recall (>72%) and feature computation throughput (>300 queries/sec).

---

## 3. Train / Validation Experiment Comparison

Deterministic split: 80% train / 20% validation saved in `splits/val_entity_ids.tsv`.
Logged to `reports/experiments.csv`:

| Evaluation Metric | Baseline Model (`EXP-01`) | Improved Model (`EXP-02`) | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **Feature Set** | 29 Baseline Features | **34 Extended Features** (Rarity, Translit, Script) | +5 Features |
| **Negative Sampling** | Standard Retrieval Negatives | **Hard Negative Mining** (High-Lexical Non-Matches) | Boundary Reinforced |
| **Score Calibration** | None (Raw Tree Probs) | **Isotonic Probability Calibration** | Calibrated Probabilities |
| **Decision Rule** | Fixed Threshold | **Entity Margin Logic** ($\tau = 0.72, \Delta \le 0.20$) | Adaptive Multi-Match |
| **Optimal Threshold** | 0.48 | **0.72** | Shifted to High Precision |
| **Macro F0.5 (Challenge Metric)** | **0.8304** | **0.8318** | **+0.0014 (+0.14%)** |
| **Macro Precision** | **0.9073** | **0.9092** | **+0.0019 (+0.19%)** |
| **Macro Recall** | **0.7028** | **0.7030** | **+0.0002** |
| **Singleton F0.5** | **0.8971** | **0.9118** | **+0.0147 (+1.47%)** |
| **Non-Singleton F0.5** | **0.8272** | **0.8280** | **+0.0008** |
| **Training Pairs** | 62,713 | 64,992 | +2,279 Hard Negatives |
| **Training Runtime** | 13.37 seconds | 12.66 seconds | Fast & Stable |

---

## 4. Submission Validation Status

The final inference output was validated using the official competition script `utils/validate_submission.py`:
- Matching Output: `output/submission/matching_results.tsv`
- Candidate Output: `output/submission/candidate_pairs.tsv`
- **Validation Result**: **PASS (Exit Code 0)**
- **Integrity**: 0 missing entities, 0 duplicates, 0 invalid target IDs, 100% strict tab separation.
