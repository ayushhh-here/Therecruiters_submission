#!/usr/bin/env python3
"""SageMaker training entry point for the business entity resolution model.

SageMaker mounts each input channel below ``/opt/ml/input/data``.  This entry
point deliberately discovers files by their tabular schema, so S3 object names
do not need to match the names used during local development.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import polars as pl

from src.blocking import CompactTargetPool, MultiStageBlocker
from src.evaluate import evaluate_macro_f05, find_optimal_decision_threshold
from src.features import EXTENDED_FEATURE_NAMES, FeatureExtractor
from src.matching import EntityMatchingModel
from src.normalization import NormalizedRecord


DEFAULT_INPUT_ROOT = "/opt/ml/input/data"
DEFAULT_MODEL_DIR = "/opt/ml/model"
DEFAULT_OUTPUT_DIR = "/opt/ml/output"
RECORD_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
GROUND_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}
GROUND_TRUTH_COLUMN_ORDER = ["source1_entity_id", "matched_entity_ids"]
RECORD_COLUMN_ORDER = ["entity_id", "business_name", "business_address", "country"]


@dataclass
class GroundTruthStats:
    anchors: int = 0
    singleton_anchors: int = 0
    valid_relationships: int = 0
    unresolved_relationships: int = 0
    duplicate_anchor_rows: int = 0


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("entity_resolution.train")


def discover_tsvs(channel_dir: Path, expected_columns: Set[str], channel_name: str) -> List[Path]:
    """Find TSV/CSV files in a channel, accepting arbitrary S3 object names."""
    if not channel_dir.is_dir():
        raise FileNotFoundError(f"Required SageMaker channel '{channel_name}' is missing: {channel_dir}")

    candidates = sorted(
        path for path in channel_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tsv", ".csv", ".txt"}
    )
    matches: List[Path] = []
    for path in candidates:
        try:
            header = set(pl.read_csv(path, separator="\t", n_rows=0).columns)
        except Exception as exc:
            raise ValueError(f"Cannot read candidate data file '{path}': {exc}") from exc
        if expected_columns.issubset(header):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No file with columns {sorted(expected_columns)} found in channel '{channel_name}' ({channel_dir}). "
            f"Files inspected: {[str(path) for path in candidates]}"
        )
    return matches


def read_channel(
    channel_dir: Path,
    expected_columns: Set[str],
    channel_name: str,
    column_order: Sequence[str],
) -> pl.DataFrame:
    files = discover_tsvs(channel_dir, expected_columns, channel_name)
    frames = [pl.read_csv(path, separator="\t", infer_schema_length=0) for path in files]
    frame = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    # Schema detection uses a set, but the selected order is semantic.  In
    # particular, ground truth must remain (source1_entity_id, matched_entity_ids).
    return frame.select(list(column_order))


def stable_partition(entity_id: str, validation_fraction: float) -> bool:
    """Returns true when an ID belongs to the deterministic validation split."""
    bucket = int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    return bucket < int(validation_fraction * 10_000)


def normalize_id(value: object) -> str:
    """Convert nullable IDs to canonical strings without turning null into 'None'."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def parse_ground_truth(frame: pl.DataFrame) -> Tuple[Dict[str, List[str]], GroundTruthStats]:
    """Map each Source 1 anchor to its zero-or-more Source 2/3 target IDs."""
    result: Dict[str, List[str]] = {}
    stats = GroundTruthStats()
    for source_id_raw, raw_matches in frame.iter_rows():
        source_id = normalize_id(source_id_raw)
        if not source_id:
            raise ValueError("Ground truth contains a missing source1_entity_id.")
        if source_id in result:
            stats.duplicate_anchor_rows += 1
        matches = [normalize_id(item) for item in normalize_id(raw_matches).split(",")]
        matches = [item for item in matches if item]
        # Union duplicate anchor rows safely if a channel was split across files.
        result[source_id] = list(dict.fromkeys([*result.get(source_id, []), *matches]))

    stats.anchors = len(result)
    stats.singleton_anchors = sum(not matches for matches in result.values())
    return result, stats


def validate_ground_truth_relationships(
    ground_truth: Dict[str, List[str]],
    source1_ids: Set[str],
    source2_ids: Set[str],
    source3_ids: Set[str],
    stats: GroundTruthStats,
) -> None:
    """Reject only genuinely inconsistent anchors or target relationships."""
    missing_anchors = sorted(set(ground_truth) - source1_ids)
    unresolved: List[Tuple[str, str]] = []
    for source_id, targets in ground_truth.items():
        for target_id in targets:
            if target_id in source2_ids or target_id in source3_ids:
                stats.valid_relationships += 1
            else:
                stats.unresolved_relationships += 1
                unresolved.append((source_id, target_id))

    log = logging.getLogger("entity_resolution.train")
    log.info(
        "Ground-truth validation: anchors=%d, singleton anchors=%d, valid relationships=%d, "
        "unresolved relationships=%d, duplicate anchor rows=%d, missing Source 1 anchors=%d",
        stats.anchors, stats.singleton_anchors, stats.valid_relationships,
        stats.unresolved_relationships, stats.duplicate_anchor_rows, len(missing_anchors),
    )
    if missing_anchors or unresolved:
        details = []
        if missing_anchors:
            details.append(f"{len(missing_anchors)} ground-truth anchors absent from source1, e.g. {missing_anchors[:3]}")
        if unresolved:
            details.append(f"{len(unresolved)} targets absent from source2/source3, e.g. {unresolved[:3]}")
        raise RuntimeError("Ground-truth/source schema inconsistency: " + "; ".join(details))


def build_pair_dataset(
    records: Sequence[NormalizedRecord],
    blocker: MultiStageBlocker,
    ground_truth: Dict[str, List[str]],
    mine_hard_negatives: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    pairs: List[Tuple[str, str]] = []
    for index, source_record in enumerate(records, start=1):
        true_matches = set(ground_truth.get(source_record.entity_id, []))
        candidates, metadata = blocker.generate_candidates_for_record(source_record)
        for rank, target_id in enumerate(candidates):
            target = metadata.get(target_id, {})
            features = FeatureExtractor.extract_pairwise_features(
                s1=source_record,
                target_raw_name=target.get("target_raw_name", ""),
                target_raw_address=target.get("target_raw_address", ""),
                target_country=target.get("target_country", ""),
                candidate_meta=target,
                candidate_rank=rank,
                use_extended_features=True,
            )
            is_match = int(target_id in true_matches)
            if mine_hard_negatives and not is_match and (
                target.get("score", 0.0) >= 8.0 or target.get("rule_count", 0) >= 2
            ):
                rows.append(features)
                labels.append(0)
                pairs.append((source_record.entity_id, target_id))
            rows.append(features)
            labels.append(is_match)
            pairs.append((source_record.entity_id, target_id))
        if index % 500 == 0:
            logging.getLogger("entity_resolution.train").info("Generated candidate features for %d source records", index)

    if not rows:
        raise RuntimeError("No training candidate pairs were generated; check channel contents and blocking configuration.")
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), pairs


def scores_by_entity(pairs: Iterable[Tuple[str, str]], probabilities: Iterable[float], records: Sequence[NormalizedRecord]):
    scores = collections.defaultdict(list)
    for (source_id, target_id), probability in zip(pairs, probabilities):
        scores[source_id].append((target_id, float(probability)))
    for record in records:
        scores.setdefault(record.entity_id, [])
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the entity-resolution model in SageMaker.")
    parser.add_argument("mode", nargs="?", default="train", choices=["train"], help=argparse.SUPPRESS)
    parser.add_argument("--input-root", default=os.environ.get("SM_CHANNEL_ROOT", DEFAULT_INPUT_ROOT))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-records", type=int, default=3000,
                        help="Cap train Source 1 records; use 0 for all records.")
    parser.add_argument("--max-validation-records", type=int, default=1500,
                        help="Cap validation Source 1 records; use 0 for all records.")
    parser.add_argument("--distractor-records-per-source", type=int, default=100000)
    parser.add_argument("--max-candidates", type=int, default=50)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    log = configure_logging()
    started = time.time()
    root = Path(args.input_root)
    model_dir, output_dir = Path(args.model_dir), Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting SageMaker-compatible entity-resolution training")
    log.info("Input root=%s, model dir=%s, output dir=%s", root, model_dir, output_dir)

    ground_truth, ground_truth_stats = parse_ground_truth(read_channel(
        root / "ground_truth", GROUND_TRUTH_COLUMNS, "ground_truth", GROUND_TRUTH_COLUMN_ORDER
    ))
    source1 = read_channel(root / "source1", RECORD_COLUMNS, "source1", RECORD_COLUMN_ORDER)
    source2 = read_channel(root / "source2", RECORD_COLUMNS, "source2", RECORD_COLUMN_ORDER)
    source3 = read_channel(root / "source3", RECORD_COLUMNS, "source3", RECORD_COLUMN_ORDER)
    log.info("Loaded rows: ground_truth=%d source1=%d source2=%d source3=%d", len(ground_truth), len(source1), len(source2), len(source3))

    train_ids, validation_ids = [], []
    for source_id in ground_truth:
        (validation_ids if stable_partition(source_id, args.validation_fraction) else train_ids).append(source_id)
    train_ids.sort(); validation_ids.sort()
    if args.max_train_records:
        train_ids = train_ids[:args.max_train_records]
    if args.max_validation_records:
        validation_ids = validation_ids[:args.max_validation_records]
    if not train_ids or not validation_ids:
        raise RuntimeError("Deterministic split produced an empty train or validation set.")
    selected_ids = set(train_ids) | set(validation_ids)
    selected_truth = {source_id: ground_truth[source_id] for source_id in selected_ids}
    needed_targets = {target_id for matches in selected_truth.values() for target_id in matches}
    log.info("Split: train=%d validation=%d required targets=%d", len(train_ids), len(validation_ids), len(needed_targets))

    # Validate the relationships that this job will train on.  Filtering first
    # avoids materializing tens of millions of Python IDs solely for validation.
    selected_source1 = source1.filter(pl.col("entity_id").is_in(selected_ids))
    source1_by_id = {normalize_id(row[0]): row for row in selected_source1.iter_rows() if normalize_id(row[0])}
    matched_source2 = source2.filter(pl.col("entity_id").is_in(needed_targets))
    matched_source3 = source3.filter(pl.col("entity_id").is_in(needed_targets))
    selected_stats = GroundTruthStats(
        anchors=len(selected_truth),
        singleton_anchors=sum(not matches for matches in selected_truth.values()),
        duplicate_anchor_rows=ground_truth_stats.duplicate_anchor_rows,
    )
    validate_ground_truth_relationships(
        selected_truth,
        set(source1_by_id),
        {normalize_id(entity_id) for entity_id in matched_source2["entity_id"].to_list()},
        {normalize_id(entity_id) for entity_id in matched_source3["entity_id"].to_list()},
        selected_stats,
    )

    pool = CompactTargetPool()
    for label, frame, matched in (("source2", source2, matched_source2), ("source3", source3, matched_source3)):
        distractors = frame.head(args.distractor_records_per_source).filter(~pl.col("entity_id").is_in(needed_targets))
        for entity_id, name, address, country in pl.concat([matched, distractors]).iter_rows():
            pool.add(entity_id, name, address, country)
        log.info("Added %d matched and %d distractor %s records to pool", len(matched), len(distractors), label)
    if not len(pool):
        raise RuntimeError("Target pool is empty.")

    def make_records(ids: Sequence[str]) -> List[NormalizedRecord]:
        return [NormalizedRecord(*source1_by_id[source_id]) for source_id in ids]
    train_records, validation_records = make_records(train_ids), make_records(validation_ids)
    blocker = MultiStageBlocker(pool, max_candidates=args.max_candidates, enable_adaptive_budget=True)

    log.info("Generating training features with the existing 34-feature pipeline and hard-negative mining")
    x_train, y_train, _ = build_pair_dataset(train_records, blocker, selected_truth, mine_hard_negatives=True)
    log.info("Generating validation features")
    x_validation, y_validation, validation_pairs = build_pair_dataset(validation_records, blocker, selected_truth, mine_hard_negatives=False)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
        raise RuntimeError("Training and validation samples must each contain both matches and non-matches.")
    log.info("Feature matrices: train=%s positives=%d; validation=%s positives=%d", x_train.shape, int(y_train.sum()), x_validation.shape, int(y_validation.sum()))

    model = EntityMatchingModel(model_type="lightgbm", use_extended_features=True, random_state=42)
    model.fit(x_train, y_train)
    model.calibrate(x_validation, y_validation, method="isotonic")
    validation_scores = scores_by_entity(validation_pairs, model.predict_proba(x_validation), validation_records)
    validation_truth = {source_id: selected_truth[source_id] for source_id in validation_ids}
    threshold, macro_f05, grid = find_optimal_decision_threshold(validation_scores, validation_truth)
    model.optimal_threshold = threshold
    model_path = model_dir / "matching_model.pkl"
    model.save(str(model_path))

    metrics = grid[threshold]
    manifest = {
        "model_file": model_path.name,
        "framework": "LightGBM with isotonic calibration",
        "feature_count": len(EXTENDED_FEATURE_NAMES),
        "optimal_threshold": threshold,
        "validation_macro_f05": macro_f05,
        "validation_metrics": metrics,
        "train_pairs": int(len(y_train)),
        "validation_pairs": int(len(y_validation)),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    for destination in (model_dir / "training_manifest.json", output_dir / "training_summary.json"):
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Saved model to %s", model_path)
    log.info("Training completed successfully: macro F0.5=%.4f threshold=%.2f elapsed=%.2fs", macro_f05, threshold, manifest["elapsed_seconds"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Training failed")
        raise
