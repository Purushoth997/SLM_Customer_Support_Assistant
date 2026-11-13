"""
Dataset Splitting Module for SLM Customer Support Assistant.
Splits the cleaned dataset into reproducible, leakage-free Train (70%), Validation (15%),
and Test (15%) splits with category stratification and exhaustive cross-split verification.
"""

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import normalize_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_stratified(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split dataset using category stratification to maintain consistent category distribution.
    Guarantees zero cross-split question leakage and mutually exclusive record IDs.
    """
    if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
        raise ValueError("Ratios must sum to 1.0")

    random.seed(seed)

    # Group by category
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in data:
        by_category[record["category"]].append(record)

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    split_distribution: Dict[str, Dict[str, int]] = {}

    for category, records in sorted(by_category.items()):
        # Deterministic shuffle within category
        shuffled = list(records)
        random.seed(seed + hash(category) % 10000)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test

        cat_train = shuffled[:n_train]
        cat_val = shuffled[n_train:n_train + n_val]
        cat_test = shuffled[n_train + n_val:]

        train_set.extend(cat_train)
        val_set.extend(cat_val)
        test_set.extend(cat_test)

        split_distribution[category] = {
            "total": n,
            "train": len(cat_train),
            "validation": len(cat_val),
            "test": len(cat_test),
        }

    # Deterministic final order
    train_set.sort(key=lambda r: r["id"])
    val_set.sort(key=lambda r: r["id"])
    test_set.sort(key=lambda r: r["id"])

    # =========================================================================
    # LEAKAGE & INTEGRITY VERIFICATION
    # =========================================================================
    train_ids = set(r["id"] for r in train_set)
    val_ids = set(r["id"] for r in val_set)
    test_ids = set(r["id"] for r in test_set)

    # 1. ID uniqueness across splits
    overlap_train_val_ids = train_ids & val_ids
    overlap_train_test_ids = train_ids & test_ids
    overlap_val_test_ids = val_ids & test_ids

    if overlap_train_val_ids or overlap_train_test_ids or overlap_val_test_ids:
        raise ValueError(
            f"ID Leakage detected between splits! "
            f"Train/Val: {overlap_train_val_ids}, Train/Test: {overlap_train_test_ids}, Val/Test: {overlap_val_test_ids}"
        )

    # 2. Question text leakage verification
    train_questions = set(normalize_question(r["question"]) for r in train_set)
    val_questions = set(normalize_question(r["question"]) for r in val_set)
    test_questions = set(normalize_question(r["question"]) for r in test_set)

    overlap_train_val_q = train_questions & val_questions
    overlap_train_test_q = train_questions & test_questions
    overlap_val_test_q = val_questions & test_questions

    if overlap_train_val_q or overlap_train_test_q or overlap_val_test_q:
        raise ValueError(
            f"Question leakage detected between splits! "
            f"Train/Val: {len(overlap_train_val_q)}, Train/Test: {len(overlap_train_test_q)}, Val/Test: {len(overlap_val_test_q)}"
        )

    # 3. Total record count conservation
    total_split_records = len(train_set) + len(val_set) + len(test_set)
    if total_split_records != len(data):
        raise ValueError(f"Record count mismatch: original {len(data)}, splits sum to {total_split_records}")

    summary = {
        "seed": seed,
        "total_records": len(data),
        "train_count": len(train_set),
        "validation_count": len(val_set),
        "test_count": len(test_set),
        "train_percentage": round(len(train_set) / len(data) * 100, 2),
        "validation_percentage": round(len(val_set) / len(data) * 100, 2),
        "test_percentage": round(len(test_set) / len(data) * 100, 2),
        "cross_split_id_overlap": 0,
        "cross_split_question_overlap": 0,
        "stratification_distribution": split_distribution,
    }

    logger.info("Stratified splitting verified: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
                len(train_set), summary["train_percentage"],
                len(val_set), summary["validation_percentage"],
                len(test_set), summary["test_percentage"])

    return train_set, val_set, test_set, summary


def run_split_pipeline(
    cleaned_path: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.json"),
    train_path: str = str(PROJECT_ROOT / "data" / "splits" / "train.json"),
    val_path: str = str(PROJECT_ROOT / "data" / "splits" / "validation.json"),
    test_path: str = str(PROJECT_ROOT / "data" / "splits" / "test.json"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute reading cleaned data, splitting, verifying, and saving splits."""
    src = Path(cleaned_path)
    if not src.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {src}")

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_set, val_set, test_set, summary = split_stratified(
        data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    for path_str, split_data in [(train_path, train_set), (val_path, val_set), (test_path, test_set)]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved split (%d records) to: %s", len(split_data), p)

    return summary


if __name__ == "__main__":
    run_split_pipeline()
