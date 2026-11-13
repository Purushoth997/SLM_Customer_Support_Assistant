"""
Automated Test Suite for Dataset Generation, Validation, Cleaning, and Splitting.
Compatible with both pytest and python's built-in unittest framework.
"""

import json
import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import (
    validate_record,
    validate_dataset,
    validate_file,
    normalize_question,
    ALLOWED_CATEGORIES,
    ALLOWED_DIFFICULTIES,
    REQUIRED_FIELDS,
)
from src.data.clean_dataset import clean_dataset, clean_text, standardize_category
from src.data.split_dataset import split_stratified


class TestDatasetIntegrity(unittest.TestCase):
    """Test suite for synthetic customer support dataset creation and preprocessing."""

    @classmethod
    def setUpClass(cls):
        cls.raw_path = PROJECT_ROOT / "data" / "raw" / "customer_support_dataset.json"
        cls.cleaned_path = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.json"
        cls.train_path = PROJECT_ROOT / "data" / "splits" / "train.json"
        cls.val_path = PROJECT_ROOT / "data" / "splits" / "validation.json"
        cls.test_path = PROJECT_ROOT / "data" / "splits" / "test.json"

    def test_raw_dataset_exists_and_loads(self):
        """Verify the raw dataset exists, parses as valid JSON, and meets minimum size."""
        self.assertTrue(self.raw_path.exists(), f"Raw dataset not found at {self.raw_path}")
        with open(self.raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 300, "Dataset must contain at least 300 records.")
        self.assertEqual(len(data), 350, "Expected exactly 350 generated raw records.")

    def test_cleaned_dataset_exists_and_loads(self):
        """Verify the cleaned dataset exists and has valid count."""
        self.assertTrue(self.cleaned_path.exists(), f"Cleaned dataset not found at {self.cleaned_path}")
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 350, "Expected 350 records in cleaned dataset.")

    def test_required_fields_present_in_all_records(self):
        """Verify that every record in the cleaned dataset has all 7 required schema fields."""
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            missing = REQUIRED_FIELDS - set(record.keys())
            self.assertEqual(len(missing), 0, f"Record {record.get('id')} is missing fields: {missing}")

    def test_field_types_and_value_domains(self):
        """Verify field data types and domain constraints."""
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            self.assertIsInstance(record["id"], str)
            self.assertTrue(record["id"].startswith("CS"), "Record ID must start with 'CS'")
            self.assertIn(record["category"], ALLOWED_CATEGORIES)
            self.assertIn(record["difficulty"], ALLOWED_DIFFICULTIES)
            self.assertIsInstance(record["requires_escalation"], bool)
            self.assertGreaterEqual(len(record["question"]), 5)
            self.assertGreaterEqual(len(record["expected_answer"]), 10)

    def test_all_record_ids_unique(self):
        """Ensure no duplicate record IDs exist in the dataset."""
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = [r["id"] for r in data]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate record IDs found in cleaned dataset.")

    def test_all_ten_categories_covered(self):
        """Verify that all 10 customer support categories are present and balanced."""
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = set(r["category"] for r in data)
        self.assertEqual(categories, ALLOWED_CATEGORIES, "Not all 10 required categories are covered.")

        # Check category balance: each category should have 35 records
        from collections import Counter
        counts = Counter(r["category"] for r in data)
        for cat in ALLOWED_CATEGORIES:
            self.assertEqual(counts[cat], 35, f"Category {cat} should have exactly 35 records.")

    def test_split_files_exist_and_ratios(self):
        """Verify train, validation, and test splits exist and match the configured distribution."""
        self.assertTrue(self.train_path.exists())
        self.assertTrue(self.val_path.exists())
        self.assertTrue(self.test_path.exists())

        with open(self.train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(self.val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(self.test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        self.assertEqual(len(train_data), 250, "Expected 250 records in training split.")
        self.assertEqual(len(val_data), 50, "Expected 50 records in validation split.")
        self.assertEqual(len(test_data), 50, "Expected 50 records in test split.")
        self.assertEqual(len(train_data) + len(val_data) + len(test_data), 350)

    def test_no_cross_split_id_leakage(self):
        """Verify absolute mutual exclusivity of record IDs across splits (zero leakage)."""
        with open(self.train_path, "r", encoding="utf-8") as f:
            train_ids = set(r["id"] for r in json.load(f))
        with open(self.val_path, "r", encoding="utf-8") as f:
            val_ids = set(r["id"] for r in json.load(f))
        with open(self.test_path, "r", encoding="utf-8") as f:
            test_ids = set(r["id"] for r in json.load(f))

        self.assertEqual(len(train_ids & val_ids), 0, "ID leakage detected between train and val!")
        self.assertEqual(len(train_ids & test_ids), 0, "ID leakage detected between train and test!")
        self.assertEqual(len(val_ids & test_ids), 0, "ID leakage detected between val and test!")

    def test_no_cross_split_question_leakage(self):
        """Verify that no question text from the test or val splits appears in the train split."""
        with open(self.train_path, "r", encoding="utf-8") as f:
            train_q = set(normalize_question(r["question"]) for r in json.load(f))
        with open(self.val_path, "r", encoding="utf-8") as f:
            val_q = set(normalize_question(r["question"]) for r in json.load(f))
        with open(self.test_path, "r", encoding="utf-8") as f:
            test_q = set(normalize_question(r["question"]) for r in json.load(f))

        self.assertEqual(len(train_q & val_q), 0, "Question leakage between train and val!")
        self.assertEqual(len(train_q & test_q), 0, "Question leakage between train and test!")
        self.assertEqual(len(val_q & test_q), 0, "Question leakage between val and test!")

    def test_split_reproducibility(self):
        """Verify that splitting with seed 42 produces identical splits every time."""
        with open(self.cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        t1, v1, s1, _ = split_stratified(data, seed=42)
        t2, v2, s2, _ = split_stratified(data, seed=42)

        self.assertEqual([r["id"] for r in t1], [r["id"] for r in t2])
        self.assertEqual([r["id"] for r in v1], [r["id"] for r in v2])
        self.assertEqual([r["id"] for r in s1], [r["id"] for r in s2])

    def test_invalid_category_detection(self):
        """Unit test: validate_record flags invalid category names."""
        bad_record = {
            "id": "CS999",
            "category": "unsupported_category_name",
            "question": "Where is my order?",
            "context": "Context here",
            "expected_answer": "Valid answer text here.",
            "requires_escalation": False,
            "difficulty": "easy",
        }
        is_valid, issues = validate_record(bad_record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid category" in issue for issue in issues))

    def test_empty_question_detection(self):
        """Unit test: validate_record flags empty or whitespace-only questions."""
        bad_record = {
            "id": "CS999",
            "category": "order_status",
            "question": "   ",
            "context": "Context here",
            "expected_answer": "Valid answer text here.",
            "requires_escalation": False,
            "difficulty": "easy",
        }
        is_valid, issues = validate_record(bad_record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("question" in issue.lower() for issue in issues))

    def test_empty_answer_detection(self):
        """Unit test: validate_record flags empty expected answers."""
        bad_record = {
            "id": "CS999",
            "category": "order_status",
            "question": "Where is my order?",
            "context": "Context here",
            "expected_answer": "",
            "requires_escalation": False,
            "difficulty": "easy",
        }
        is_valid, issues = validate_record(bad_record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("expected_answer" in issue.lower() for issue in issues))

    def test_duplicate_record_detection(self):
        """Unit test: validate_dataset detects duplicate IDs and duplicate questions."""
        records = [
            {
                "id": "CS001",
                "category": "order_status",
                "question": "Where is my order?",
                "context": "Order shipped.",
                "expected_answer": "Your order has been shipped.",
                "requires_escalation": False,
                "difficulty": "easy",
            },
            {
                "id": "CS001",  # Duplicate ID
                "category": "order_status",
                "question": "Where is my order?",  # Duplicate question
                "context": "Order shipped.",
                "expected_answer": "Your order has been shipped.",
                "requires_escalation": False,
                "difficulty": "easy",
            },
        ]
        report = validate_dataset(records)
        self.assertFalse(report["is_valid"])
        self.assertGreater(report["duplicate_records"], 0)
        self.assertGreater(report["duplicate_questions"], 0)


if __name__ == "__main__":
    unittest.main()
