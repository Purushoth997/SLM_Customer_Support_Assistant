"""
Unit Tests for Baseline Evaluation Metrics and Aggregate Calculations.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    token_f1_score,
    exact_match_score,
    detect_escalation_intent,
    is_valid_response,
    calculate_aggregate_metrics,
)


class TestBaselineEvaluation(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_token_f1_identical_strings(self):
        """Identical strings must yield F1 of 1.0."""
        text = "Your order has been dispatched via FedEx."
        res = token_f1_score(text, text)
        self.assertEqual(res["f1"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)

    def test_token_f1_disjoint_strings(self):
        """Completely disjoint strings must yield F1 of 0.0."""
        res = token_f1_score("apple orange banana", "cat dog elephant")
        self.assertEqual(res["f1"], 0.0)

    def test_exact_match(self):
        """Exact match should be resilient to casing, punctuation, and articles."""
        gt = "The package was delivered today."
        pred = "package was delivered today!"
        self.assertEqual(exact_match_score(pred, gt), 1.0)

    def test_detect_escalation_positive(self):
        """Keywords indicating escalation must be detected."""
        phrases = [
            "I am escalating your ticket to a supervisor.",
            "Transferring you to a manager now.",
            "A human representative will join shortly.",
            "Our fraud team will investigate this charge.",
        ]
        for p in phrases:
            self.assertTrue(detect_escalation_intent(p), f"Failed to detect escalation in: '{p}'")

    def test_detect_escalation_negative(self):
        """Routine policy answers must not trigger escalation detection."""
        phrases = [
            "You can return items within 30 days of purchase.",
            "Standard shipping takes 3 to 5 business days.",
            "Please check your tracking code online.",
        ]
        for p in phrases:
            self.assertFalse(detect_escalation_intent(p), f"Falsely detected escalation in: '{p}'")

    def test_is_valid_response(self):
        """Check response validity filter."""
        self.assertTrue(is_valid_response("Hello, your order has shipped and will arrive tomorrow."))
        self.assertFalse(is_valid_response(""))
        self.assertFalse(is_valid_response("Too short"))

    def test_calculate_aggregate_metrics(self):
        """Verify summary aggregation logic."""
        dummy_records = [
            {
                "id": "CS001",
                "category": "shipping_delivery",
                "generated_response": "Your package is on its way via FedEx.",
                "expected_answer": "Your package has shipped via FedEx.",
                "latency_seconds": 2.5,
                "completion_tokens": 10,
                "tokens_per_second": 4.0,
                "requires_escalation": False,
                "error_status": None,
            },
            {
                "id": "CS002",
                "category": "escalation_requests",
                "generated_response": "I am escalating this to a supervisor right away.",
                "expected_answer": "Transferring to a supervisor now.",
                "latency_seconds": 1.5,
                "completion_tokens": 10,
                "tokens_per_second": 6.67,
                "requires_escalation": True,
                "error_status": None,
            },
        ]
        agg = calculate_aggregate_metrics(dummy_records)
        self.assertEqual(agg["total_test_records"], 2)
        self.assertEqual(agg["successful_inferences"], 2)
        self.assertEqual(agg["failed_inferences"], 0)
        self.assertEqual(agg["response_validity_rate_pct"], 100.0)
        self.assertEqual(agg["escalation_metrics"]["true_positives"], 1)
        self.assertEqual(agg["escalation_metrics"]["true_negatives"], 1)
        self.assertEqual(agg["latency_stats"]["mean_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
