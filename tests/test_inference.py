"""
Unit Tests for Customer Support Inference Engine and Input Validation.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.inference import generate_response, MAX_QUESTION_LENGTH


class TestInference(unittest.TestCase):
    """Test suite for inference input validation, output structures, and error handling."""

    def test_empty_question_returns_error(self):
        """Verify empty question string returns immediate validation failure without model call."""
        res = generate_response("")
        self.assertFalse(res["success"])
        self.assertEqual(res["latency_seconds"], 0.0)
        self.assertIn("empty", res["error"].lower())

    def test_whitespace_question_returns_error(self):
        """Verify whitespace-only question string is rejected."""
        res = generate_response("    \n\t  ")
        self.assertFalse(res["success"])
        self.assertIn("empty", res["error"].lower())

    def test_excessively_long_question_rejected(self):
        """Verify questions exceeding MAX_QUESTION_LENGTH are rejected."""
        long_q = "Where is my order? " * (MAX_QUESTION_LENGTH // 15 + 10)
        res = generate_response(long_q)
        self.assertFalse(res["success"])
        self.assertIn("exceeds", res["error"].lower())

    def test_response_structure_keys(self):
        """Verify the contract schema returned by generate_response."""
        res = generate_response("")  # Fast check on returned structure
        expected_keys = {
            "success",
            "answer",
            "latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "tokens_per_second",
            "model_name",
            "error",
        }
        self.assertTrue(expected_keys.issubset(set(res.keys())))


if __name__ == "__main__":
    unittest.main()
