"""
Dataset Cleaning Module for SLM Customer Support Assistant.
Loads raw synthetic records, normalizes whitespace, standardizes categorical keys,
filters corrupt or invalid entries, prevents duplicates, and saves the cleaned dataset.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import (
    validate_record,
    normalize_question,
    ALLOWED_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace, remove invisible Unicode artifacts, and strip edges."""
    if not isinstance(text, str):
        return ""
    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Normalize multiple whitespaces and newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def standardize_category(raw_cat: str) -> str:
    """Standardize category naming convention to snake_case."""
    if not isinstance(raw_cat, str):
        return ""
    clean = raw_cat.strip().lower().replace("-", "_").replace(" ", "_")
    return clean


def clean_dataset(raw_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Clean the dataset while tracking all modifications, removals, and deduplications.
    Returns:
        cleaned_records: List of sanitized, valid records
        cleaning_stats: Detailed statistics dictionary
    """
    original_count = len(raw_data)
    cleaned_records: List[Dict[str, Any]] = []
    removed_records: List[Dict[str, Any]] = []

    seen_ids = set()
    seen_normalized_questions = set()
    duplicate_count = 0
    invalid_count = 0

    for idx, raw_record in enumerate(raw_data):
        if not isinstance(raw_record, dict):
            invalid_count += 1
            removed_records.append({
                "index": idx,
                "reason": "Not a valid dictionary object",
                "raw": str(raw_record),
            })
            continue

        # Standardize and sanitize fields
        cleaned_rec = {
            "id": clean_text(str(raw_record.get("id", ""))),
            "category": standardize_category(str(raw_record.get("category", ""))),
            "question": clean_text(str(raw_record.get("question", ""))),
            "context": clean_text(str(raw_record.get("context", ""))),
            "expected_answer": clean_text(str(raw_record.get("expected_answer", ""))),
            "requires_escalation": bool(raw_record.get("requires_escalation", False)),
            "difficulty": clean_text(str(raw_record.get("difficulty", "medium"))).lower(),
        }

        # Check validity against schema
        is_valid, issues = validate_record(cleaned_rec, idx)
        if not is_valid:
            invalid_count += 1
            removed_records.append({
                "index": idx,
                "id": cleaned_rec.get("id", f"UNKNOWN_{idx}"),
                "reason": f"Validation failure: {'; '.join(issues)}",
                "record": cleaned_rec,
            })
            continue

        # Check duplicate ID
        if cleaned_rec["id"] in seen_ids:
            duplicate_count += 1
            removed_records.append({
                "index": idx,
                "id": cleaned_rec["id"],
                "reason": f"Duplicate record ID: {cleaned_rec['id']}",
                "record": cleaned_rec,
            })
            continue

        # Check duplicate question content
        norm_q = normalize_question(cleaned_rec["question"])
        if norm_q in seen_normalized_questions:
            duplicate_count += 1
            removed_records.append({
                "index": idx,
                "id": cleaned_rec["id"],
                "reason": f"Duplicate question text: {cleaned_rec['question']}",
                "record": cleaned_rec,
            })
            continue

        seen_ids.add(cleaned_rec["id"])
        seen_normalized_questions.add(norm_q)
        cleaned_records.append(cleaned_rec)

    stats = {
        "original_record_count": original_count,
        "valid_record_count": len(cleaned_records),
        "duplicate_count": duplicate_count,
        "invalid_record_count": invalid_count,
        "removed_record_count": len(removed_records),
        "final_cleaned_count": len(cleaned_records),
        "removed_details": removed_records,
    }

    logger.info("Cleaning finished: %d raw -> %d cleaned (%d duplicates, %d invalid removed).",
                original_count, len(cleaned_records), duplicate_count, invalid_count)
    return cleaned_records, stats


def run_clean_pipeline(
    raw_path: str = str(PROJECT_ROOT / "data" / "raw" / "customer_support_dataset.json"),
    output_path: str = str(PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.json"),
) -> Tuple[Path, Dict[str, Any]]:
    """Execute the end-to-end cleaning pipeline from file to file."""
    src = Path(raw_path)
    dest = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {src}")

    with open(src, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data, stats = clean_dataset(raw_data)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    logger.info("Cleaned dataset saved successfully to: %s", dest)
    return dest, stats


if __name__ == "__main__":
    run_clean_pipeline()
