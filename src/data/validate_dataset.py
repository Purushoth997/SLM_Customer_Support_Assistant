"""
Dataset Validation Module for SLM Customer Support Assistant.
Performs rigorous structural, semantic, and categorical validation on synthetic datasets.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "order_status",
    "shipping_delivery",
    "payment_issues",
    "refund_requests",
    "return_policies",
    "product_information",
    "account_support",
    "customer_complaints",
    "escalation_requests",
    "general_support",
}

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "context",
    "expected_answer",
    "requires_escalation",
    "difficulty",
}


def normalize_question(q: str) -> str:
    """Normalize a question string for duplicate detection."""
    clean = re.sub(r"[^\w\s]", "", q.lower())
    return " ".join(clean.split())


def validate_record(record: Any, index: int) -> Tuple[bool, List[str]]:
    """Validate a single record against the customer support schema."""
    issues = []

    if not isinstance(record, dict):
        return False, [f"Record at index {index} is not a valid JSON object/dictionary."]

    # Check for missing fields
    missing_fields = REQUIRED_FIELDS - set(record.keys())
    if missing_fields:
        issues.append(f"Missing required fields: {sorted(list(missing_fields))}")

    # ID validation
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        issues.append("Field 'id' must be a non-empty string.")

    # Category validation
    category = record.get("category")
    if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
        issues.append(f"Invalid category '{category}'. Allowed: {sorted(list(ALLOWED_CATEGORIES))}")

    # Question validation
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        issues.append("Field 'question' must be a non-empty string.")
    elif len(question.strip()) < 5:
        issues.append(f"Question too short ({len(question.strip())} chars; min 5 required).")
    elif len(question.strip()) > 600:
        issues.append(f"Question exceeds maximum length of 600 characters.")

    # Context validation
    context = record.get("context")
    if not isinstance(context, str) or not context.strip():
        issues.append("Field 'context' must be a non-empty string.")

    # Expected answer validation
    expected_answer = record.get("expected_answer")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        issues.append("Field 'expected_answer' must be a non-empty string.")
    elif len(expected_answer.strip()) < 10:
        issues.append(f"Expected answer too short ({len(expected_answer.strip())} chars; min 10 required).")
    elif len(expected_answer.strip()) > 1200:
        issues.append(f"Expected answer exceeds maximum length of 1200 characters.")

    # Requires escalation validation (strictly boolean)
    escalation = record.get("requires_escalation")
    if not isinstance(escalation, bool):
        issues.append(f"Field 'requires_escalation' must be a boolean (True/False), got {type(escalation).__name__}.")

    # Difficulty validation
    difficulty = record.get("difficulty")
    if not isinstance(difficulty, str) or difficulty not in ALLOWED_DIFFICULTIES:
        issues.append(f"Invalid difficulty '{difficulty}'. Allowed: {sorted(list(ALLOWED_DIFFICULTIES))}")

    return (len(issues) == 0), issues


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate an entire dataset and produce a comprehensive diagnostic report."""
    total_records = len(data)
    valid_records = 0
    invalid_records = 0
    errors: List[Dict[str, Any]] = []

    seen_ids: Dict[str, int] = {}
    seen_questions: Dict[str, int] = {}
    duplicate_records_count = 0
    duplicate_questions_count = 0

    category_counts: Dict[str, int] = {cat: 0 for cat in ALLOWED_CATEGORIES}
    difficulty_counts: Dict[str, int] = {diff: 0 for diff in ALLOWED_DIFFICULTIES}
    escalation_counts: Dict[str, int] = {"True": 0, "False": 0}

    for idx, record in enumerate(data):
        is_valid, record_issues = validate_record(record, idx)

        # Check ID duplicates
        record_id = record.get("id")
        if record_id:
            if record_id in seen_ids:
                is_valid = False
                record_issues.append(f"Duplicate ID '{record_id}' (first seen at index {seen_ids[record_id]}).")
                duplicate_records_count += 1
            else:
                seen_ids[record_id] = idx

        # Check question duplicates
        question = record.get("question")
        if question and isinstance(question, str):
            norm_q = normalize_question(question)
            if norm_q in seen_questions:
                is_valid = False
                record_issues.append(f"Duplicate question text (first seen at index {seen_questions[norm_q]}).")
                duplicate_questions_count += 1
            else:
                seen_questions[norm_q] = idx

        if is_valid:
            valid_records += 1
            category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
            difficulty_counts[record["difficulty"]] = difficulty_counts.get(record["difficulty"], 0) + 1
            esc_key = "True" if record["requires_escalation"] else "False"
            escalation_counts[esc_key] = escalation_counts.get(esc_key, 0) + 1
        else:
            invalid_records += 1
            errors.append({
                "index": idx,
                "id": record.get("id", f"UNKNOWN_{idx}"),
                "issues": record_issues,
            })

    report = {
        "is_valid": (invalid_records == 0 and duplicate_records_count == 0 and duplicate_questions_count == 0),
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "duplicate_records": duplicate_records_count,
        "duplicate_questions": duplicate_questions_count,
        "category_distribution": category_counts,
        "difficulty_distribution": difficulty_counts,
        "escalation_distribution": escalation_counts,
        "errors": errors,
    }
    return report


def validate_file(file_path: str) -> Dict[str, Any]:
    """Load and validate a JSON dataset file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in %s: %s", path, e)
            return {
                "is_valid": False,
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "duplicate_records": 0,
                "duplicate_questions": 0,
                "errors": [{"index": -1, "id": "PARSE_ERROR", "issues": [f"Malformed JSON: {str(e)}"]}],
            }

    if not isinstance(data, list):
        return {
            "is_valid": False,
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "duplicate_records": 0,
            "duplicate_questions": 0,
            "errors": [{"index": -1, "id": "ROOT_ERROR", "issues": ["Root JSON element must be a list"]}],
        }

    report = validate_dataset(data)
    logger.info("Validation complete for %s. Valid: %s | Total: %d | Errors: %d",
                path.name, report["is_valid"], report["total_records"], report["invalid_records"])
    return report


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "raw" / "customer_support_dataset.json")
    result = validate_file(target)
    print(json.dumps(result, indent=2))
