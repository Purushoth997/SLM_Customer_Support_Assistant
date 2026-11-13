"""
Unit Tests for Customer Support Prompt Template Construction.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.prompt_template import (
    build_chat_messages,
    format_user_prompt,
    DEFAULT_SYSTEM_PROMPT,
)


class TestPromptTemplate(unittest.TestCase):
    """Unit test suite for prompt formatting and persona constraints."""

    def test_format_user_prompt_with_context(self):
        """Verify proper concatenation of business context and customer inquiry."""
        q = "Where is my order?"
        ctx = "Order APX-101 has departed Dallas hub."
        formatted = format_user_prompt(q, ctx)
        self.assertIn("[Business Context]", formatted)
        self.assertIn(ctx, formatted)
        self.assertIn("[Customer Inquiry]", formatted)
        self.assertIn(q, formatted)

    def test_format_user_prompt_without_context(self):
        """Verify prompt formatting when no context is supplied."""
        q = "What are your store hours?"
        formatted = format_user_prompt(q, None)
        self.assertNotIn("[Business Context]", formatted)
        self.assertIn("[Customer Inquiry]", formatted)
        self.assertIn(q, formatted)

    def test_build_chat_messages_structure(self):
        """Verify standard 2-turn role structure (system + user)."""
        messages = build_chat_messages("How to return an item?", "30-day window.")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Apex Retail Solutions", messages[0]["content"])
        self.assertIn("30-day window", messages[1]["content"])

    def test_system_prompt_contains_core_guardrails(self):
        """Verify essential negative constraints exist in the system prompt."""
        self.assertIn("Never invent", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("missing", DEFAULT_SYSTEM_PROMPT.lower())
        self.assertIn("escalat", DEFAULT_SYSTEM_PROMPT.lower())

    def test_custom_system_prompt_override(self):
        """Verify ability to pass custom system prompt."""
        custom = "You are a test agent."
        messages = build_chat_messages("Hello", system_prompt=custom)
        self.assertEqual(messages[0]["content"], custom)


if __name__ == "__main__":
    unittest.main()
