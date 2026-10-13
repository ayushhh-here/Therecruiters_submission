# Amazon ML Challenge — Business Entity Resolution Pipeline

A high-performance, multilingual, memory-safe entity resolution submission pipeline designed to match business entity records between query datasets (Source 1) and massive multi-million record target reference pools (Source 2 and Source 3) under strict 16 GB RAM hardware constraints.

---

## 1. Executive Summary & Measured Achievements (Phase 19 Updated)

| Milestone | Status | Key Measured Result |
| :--- | :---: | :--- |
| **Real Challenge Dataset Profiling** | **DONE** | 26.4 million records profiled across 7 TSV files in 19.75s (< 1.2 GB RAM). |
| **Reproducible Environment** | **DONE** | Pinned lockfile (`requirements-lock.txt`), smoke test passing 10/10 checks. |
| **Multilingual Normalization** | **DONE** | 100% Unicode preservation (Devanagari, Tamil, Telugu, French accents). |
| **Measured Blocking System (Phase 19)** | **DONE** | **Recall@20 = 88.30%**, **Recall@50 = 89.78%**, **Recall@200 = 91.49%** (0.00% zero-candidate queries). |
| **Leak-Free 4-Way Data Partition** | **DONE** | Train (60%), Calibration (15%), Dev/Threshold (15%), Final Holdout (10%). |
| **Feature Engineering & Rarity** | **DONE** | 34 features (RapidFuzz C-Levenshtein, transliteration, script match, rarity). |
| **Hard Negative Mining** | **DONE** | Boundary reinforced on high-similarity non-matching candidates. |
| **Probability Calibration** | **DONE** | Isotonic Regression calibration mapping scores to true probabilities. |
| **Entity Decision Logic** | **DONE** | Score margin ($\Delta \le 0.20$) and singleton thresholding ($\tau = 0.72$). |
| **Final Holdout Evaluation Score** | **DONE** | **Macro F0.5 = 0.9342** (Precision: 0.9712, Recall: 0.8621, Singleton: 0.9559). |
| **Memory-Safe Streaming Inference** | **DONE** | Chunked batching processing at **180–280 queries/sec** under 600 MB RAM. |
| **Official Submission Validator** | **DONE** | **PASS (Exit Code 0)** on `utils/validate_submission.py`. |

---

## 2. Dataset Scale & Architecture Overview

The Amazon ML Challenge dataset consists of real-world multi-national business entities:
- **Query Pool (Source 1)**: 2,206,821 train records / 1,732,544 test records.
- **Target Pool (Source 2 + Source 3)**: 10,320,219 train records / 9,969,589 test records (~10M targets).
- **Match Multiplicity**: Ground truth exhibits a mean of 3.46 matches per entity (median 3, max 11), with 5.58% singletons (0 matches).
- **Country Distribution**: Open-set setting covering United States, India, and France.

```
Source 1 Streaming Chunk (e.g., 1,000 - 10,000 queries)
                  │
                  ▼
   Multilingual Normalization (Unicode Preserved)
                  │
                  ▼
   Multi-Stage Orchestrated Retrieval (9 complementary signals):
     1. Exact Name  2. Sorted Name  3. Core Name  4. Translit Name
     5. Exact Address  6. Street Component (Num + Token)
     7. Postal + Name Token  8. Soft-Capped Token Index  9. Trigrams
                  │
                  ▼
   High-Speed Pairwise Feature Extraction (34 dense features via C-RapidFuzz)
                  │
                  ▼
   LightGBM Classifier + Isotonic Probability Calibration
                  │
                  ▼
   Entity-Level Margin Decision Logic (multi-match & singleton resolution)
                  │
                  ▼
   Incremental Output Write (output/submission/matching_results.tsv)
```

---

## 3. Verified Validation Experiments & Factorial Study

### Factorial 2x2 Evaluation on Untouched Final Holdout Split:

| Configuration | Macro F0.5 | Precision | Recall | Non-Singleton F0.5 | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Old Blocking + Baseline Matcher** | 0.8297 | 0.8986 | 0.7116 | 0.8189 | 0.56 |
| **Old Blocking + Improved Matcher** | 0.8274 | 0.9022 | 0.6998 | 0.8164 | 0.52 |
| **New Blocking + Baseline Matcher** | **0.9267** | 0.9585 | 0.8634 | 0.9295 | 0.58 |
| **New Blocking + Improved Matcher** | **0.9342** | **0.9712** | **0.8621** | **0.9332** | **0.72** |

> **Key Architectural Insight**: Moving from Old Blocking to New Multi-Signal Blocking drove a **+0.0970 to +0.1045 absolute gain in Macro F0.5** (jumping from 0.83 to 0.934), proving that retrieval recall was the primary performance bottleneck.

---

## 4. Repository Structure

```
Therecruiters_submission/
├── dataset/                        # Challenge data directory
│   ├── train/                      # train_source{1,2,3}.tsv & train_ground_truth.tsv
│   ├── test/                       # test_source{1,2,3}.tsv
│   └── test_sample/                # 2,000 S1 / 50,000 target verified test sample
├── models/
│   └── matching_model.pkl          # Trained calibrated LightGBM model checkpoint
├── output/
│   ├── demo/                       # [DEMO ONLY] Mock test outputs from synthetic demo
│   ├── experiments/                # Intermediate validation experiments
│   └── submission/                 # Official submission TSVs (matching_results.tsv)
├── reports/
│   ├── dataset_profile.json        # Machine-readable profile of all 26.4M records
│   ├── dataset_profile_summary.txt # Human-readable data profiling summary
│   ├── blocking_evaluation.json    # Measured Recall@K curve (K=1 to 200)
│   ├── true_match_rank_distribution.csv # Per-pair true match rank breakdown
│   ├── blocking_failure_analysis.json   # Diagnostic categorization of missed matches
│   ├── blocking_ablation.csv       # Measured ablation of each retrieval signal
│   ├── matrix_2x2_evaluation.csv   # Factorial 2x2 evaluation results
│   ├── experiments.csv             # Full experiment comparison log
│   └── performance_benchmarks.md   # Stage-by-stage throughput & RAM benchmarks
├── scripts/
│   ├── profile_dataset.py          # Fast streaming dataset profiler
│   ├── smoke_test.py               # 10-point end-to-end pipeline verification check
│   ├── diagnose_blocking.py        # True match rank distribution and failure categorization
│   ├── ablate_blocking.py          # Independent retrieval signal ablation runner
│   ├── evaluate_blocking.py        # Measured blocking recall evaluator
│   ├── run_2x2_experiment.py       # Factorial 2x2 experiment runner
│   ├── train_and_evaluate.py       # Baseline vs Improved model experiment runner
│   ├── infer.py                    # Memory-safe chunked streaming inference pipeline
│   └── run_pipeline.py             # Master CLI orchestrator
├── splits/                         # Leak-free 4-way data partition files
│   ├── split_train_ids.tsv         # 60% Train partition
│   ├── split_calib_ids.tsv         # 15% Calibration partition
│   ├── split_dev_ids.tsv           # 15% Dev/Threshold partition
│   └── split_holdout_ids.tsv       # 10% Final Holdout partition
├── src/
│   ├── normalization.py            # Multilingual Unicode normalization & script detection
│   ├── blocking.py                 # Multi-signal orchestrated candidate generation engine
│   ├── features.py                 # 34-feature pairwise extractor (RapidFuzz C-backend)
│   ├── matching.py                 # EntityMatchingModel (LightGBM, calibration, margin logic)
│   ├── evaluate.py                 # Challenge Macro F0.5 evaluation metric & grid search
│   └── pipeline.py                 # Unified pipeline orchestrator
├── tests/
│   └── test_pipeline.py            # 14 unit and adversarial test cases
├── utils/
│   └── validate_submission.py      # Official competition submission validator
├── requirements.txt                # Abstract package dependencies
└── requirements-lock.txt           # Pinned exact reproducible dependencies
```

---

## 5. Quickstart & Execution Commands

### Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1. Run Pipeline Smoke Test (10 Verification Checks)
```powershell
python scripts/smoke_test.py
```

### 2. Run Comprehensive Unit & Adversarial Tests (14 Tests)
```powershell
python -m unittest tests/test_pipeline.py
```

### 3. Profile Dataset (26.4M Records)
```powershell
python scripts/profile_dataset.py dataset
```

### 4. Run Blocking Diagnosis & True-Match Rank Distribution
```powershell
python scripts/diagnose_blocking.py
```

### 5. Run Blocking Strategy Ablation Study
```powershell
python scripts/ablate_blocking.py
```

### 6. Run 2x2 Factorial Leak-Free Experiment
```powershell
python scripts/run_2x2_experiment.py
```

### 7. Train & Evaluate Models
```powershell
python scripts/train_and_evaluate.py
```

### 8. Execute Streaming Test Inference & Validate
```powershell
python scripts/infer.py --test-dir dataset/test_sample --output-dir output/submission --chunk-size 500
```

---

## SageMaker custom training container

The SageMaker entry point is [train.py](train.py). It preserves the production
34-feature entity-resolution flow: multi-stage blocking, hard-negative mining,
LightGBM training, isotonic calibration, and validation F0.5 threshold tuning.
It writes `matching_model.pkl` and `training_manifest.json` to `/opt/ml/model`;
it also writes a training summary to `/opt/ml/output`.

SageMaker must provide four named input channels. Files may have arbitrary
names (and a channel may contain multiple TSV files), but their tab-separated
headers must be:

| SageMaker channel | Mount path | Required columns |
| --- | --- | --- |
| `ground_truth` | `/opt/ml/input/data/ground_truth` | `source1_entity_id`, `matched_entity_ids` |
| `source1` | `/opt/ml/input/data/source1` | `entity_id`, `business_name`, `business_address`, `country` |
| `source2` | `/opt/ml/input/data/source2` | `entity_id`, `business_name`, `business_address`, `country` |
| `source3` | `/opt/ml/input/data/source3` | `entity_id`, `business_name`, `business_address`, `country` |

The image contains code and dependencies only—never training data. The default
record caps match the existing experiment runner (3,000 train and 1,500
validation Source 1 entities). Pass `--max-train-records 0 --max-validation-records 0`
to train on every eligible Source 1 record.

### Windows PowerShell: build and local test

Run these commands from the repository root. Docker Desktop must be running.

```powershell
# Check Docker
docker version
docker info

# Build the SageMaker training image
docker build --tag entity-resolution-training:latest .

# Prepare SageMaker-like local channel mounts (copies are intentionally outside the image)
$localTest = Join-Path $PWD "sagemaker-local-test"
New-Item -ItemType Directory -Force "$localTest\input\ground_truth", "$localTest\input\source1", "$localTest\input\source2", "$localTest\input\source3", "$localTest\model", "$localTest\output"
Copy-Item .\dataset\train\train_ground_truth.tsv "$localTest\input\ground_truth\ground_truth.tsv"
Copy-Item .\dataset\train\train_source1.tsv "$localTest\input\source1\source1.tsv"
Copy-Item .\dataset\train\train_source2.tsv "$localTest\input\source2\source2.tsv"
Copy-Item .\dataset\train\train_source3.tsv "$localTest\input\source3\source3.tsv"

# Run the container (use small caps first; SageMaker will pass `train` automatically)
docker run --rm `
  -v "${localTest}\input:/opt/ml/input/data:ro" `
  -v "${localTest}\model:/opt/ml/model" `
  -v "${localTest}\output:/opt/ml/output" `
  entity-resolution-training:latest train `
  --max-train-records 100 --max-validation-records 50 --distractor-records-per-source 1000

# Verify the local artifacts
Get-ChildItem "$localTest\model"
Get-Content "$localTest\output\training_summary.json"
```

### Windows PowerShell: tag and push to ECR

Replace the placeholders with your AWS account ID and region. Create the ECR
repository once in AWS before pushing.

```powershell
$awsAccountId = "123456789012"
$awsRegion = "us-east-1"
$ecrRegistry = "$awsAccountId.dkr.ecr.$awsRegion.amazonaws.com"

# Tag the local image for ECR
docker tag entity-resolution-training:latest "$ecrRegistry/entity-resolution-training:latest"

# Authenticate Docker to ECR
aws ecr get-login-password --region $awsRegion | docker login --username AWS --password-stdin $ecrRegistry

# Push the image
docker push "$ecrRegistry/entity-resolution-training:latest"
```
"# parth" 
"# parth" 
