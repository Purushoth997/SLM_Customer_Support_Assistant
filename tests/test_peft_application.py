"""
Unit and Integration Tests for SLM Customer Support PEFT Application.
Validates schemas, input constraints, model loader error handling, prompt formatting,
mocked inference, API endpoints, and static frontend assets using standard Python unittest.
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_MODEL_NAME_OR_PATH,
    BASE_MODEL_REVISION,
    EXPECTED_ADAPTER_SHA256,
    MAX_QUESTION_LENGTH,
)
from src.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.model.load_peft import (
    compute_file_sha256,
    verify_adapter_integrity,
    check_model_assets_readiness,
    reset_peft_model,
    is_model_loaded,
)
from src.model.peft_inference import (
    build_full_conversation_messages,
    generate_peft_response,
)
from src.model.prompt_template import DEFAULT_SYSTEM_PROMPT, format_user_prompt


class TestSchemasAndValidation(unittest.TestCase):
    """Test suite for Pydantic/dataclass schema contracts and input validation."""

    def test_chat_message_schema(self):
        msg = ChatMessage(role="user", content="Where is my package?")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Where is my package?")

    def test_chat_request_schema(self):
        req = ChatRequest(
            message="Can I return an opened item?",
            context="Standard 30-day return policy",
            temperature=0.3,
            max_new_tokens=128,
        )
        self.assertEqual(req.message, "Can I return an opened item?")
        self.assertEqual(req.context, "Standard 30-day return policy")
        self.assertEqual(req.temperature, 0.3)
        self.assertEqual(req.max_new_tokens, 128)

    def test_chat_response_schema_keys(self):
        resp = ChatResponse(
            success=True,
            answer="Yes, returns are accepted within 30 days.",
            latency_seconds=1.25,
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
            tokens_per_second=12.0,
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_name="customer-support-lora",
            device="cpu",
            error=None,
        )
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        required_keys = {
            "success",
            "answer",
            "latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "tokens_per_second",
            "model_name",
            "adapter_name",
            "device",
            "error",
        }
        self.assertTrue(required_keys.issubset(set(data.keys())))
        self.assertTrue(resp.success)
        self.assertEqual(resp.total_tokens, 45)

    def test_health_response_schema(self):
        health = HealthResponse(
            status="healthy",
            service="Apex Retail Solutions",
            version="1.0.0",
            model_loaded=True,
            adapter_loaded=True,
            device="cpu",
            cpu_usage_percent=15.5,
            ram_used_gb=6.2,
            ram_total_gb=16.0,
        )
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.device, "cpu")
        self.assertTrue(health.model_loaded)

    def test_model_info_response_schema(self):
        info = ModelInfoResponse(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            base_model_revision=BASE_MODEL_REVISION,
            adapter_name="customer-support-lora",
            adapter_dir="exports/customer-support-lora",
            expected_adapter_sha256=EXPECTED_ADAPTER_SHA256,
            adapter_verified=True,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.10,
            target_modules=["q_proj", "k_proj", "v_proj"],
            dynamic_peft=True,
            merged_weights=False,
            gguf_exported=False,
        )
        self.assertTrue(info.dynamic_peft)
        self.assertFalse(info.merged_weights)
        self.assertFalse(info.gguf_exported)
        self.assertEqual(info.lora_rank, 8)


class TestModelLoaderErrorHandling(unittest.TestCase):
    """Test suite for error detection when model assets or packages are absent."""

    def setUp(self):
        reset_peft_model()

    def test_compute_sha256_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            compute_file_sha256(PROJECT_ROOT / "non_existent_file.safetensors")

    def test_verify_adapter_missing_directory(self):
        is_valid, msg, details = verify_adapter_integrity("non_existent_adapter_dir")
        self.assertFalse(is_valid)
        self.assertFalse(details["exists"])
        self.assertIn("does not exist", msg.lower())

    def test_check_model_assets_readiness_runs_safely(self):
        readiness = check_model_assets_readiness()
        self.assertIn("ready", readiness)
        self.assertIn("missing_dependencies", readiness)
        self.assertIn("adapter_verified", readiness)
        self.assertIn("base_model_available_locally", readiness)

    def test_model_loaded_initially_false(self):
        self.assertFalse(is_model_loaded())


class TestPromptTemplate(unittest.TestCase):
    """Test suite for ChatML prompt formatting and context injection."""

    def test_format_user_prompt_with_context(self):
        res = format_user_prompt("Where is my order?", context="Order #12345 shipped via FedEx")
        self.assertIn("[Business Context]", res)
        self.assertIn("Order #12345 shipped via FedEx", res)
        self.assertIn("[Customer Inquiry]", res)
        self.assertIn("Where is my order?", res)

    def test_format_user_prompt_without_context(self):
        res = format_user_prompt("How do I reset my password?")
        self.assertNotIn("[Business Context]", res)
        self.assertIn("[Customer Inquiry]", res)
        self.assertIn("How do I reset my password?", res)

    def test_build_full_conversation_messages_structure(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        ]
        messages = build_full_conversation_messages(
            question="What is the shipping cost?",
            context="Standard shipping is $5.99",
            history=history,
        )
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Apex Retail Solutions", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("Standard shipping is $5.99", messages[3]["content"])


class TestPeftInferenceService(unittest.TestCase):
    """Test suite for input constraints and mocked inference execution."""

    def test_empty_question_returns_immediate_error(self):
        res = generate_peft_response("")
        self.assertFalse(res["success"])
        self.assertEqual(res["latency_seconds"], 0.0)
        self.assertIn("empty", res["error"].lower())

    def test_whitespace_question_returns_immediate_error(self):
        res = generate_peft_response("   \t  \n ")
        self.assertFalse(res["success"])
        self.assertEqual(res["latency_seconds"], 0.0)
        self.assertIn("empty", res["error"].lower())

    def test_oversized_question_rejected(self):
        long_query = "Help " * (MAX_QUESTION_LENGTH // 5 + 10)
        res = generate_peft_response(long_query)
        self.assertFalse(res["success"])
        self.assertIn("exceeds", res["error"].lower())

    @patch("src.model.peft_inference.get_peft_model")
    def test_mocked_model_successful_generation(self, mock_get_model):
        """Verify inference pipeline mechanics when model and tokenizer are mocked."""
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 151645
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

        # Mock tokenizer encoding
        fake_inputs = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        fake_inputs["input_ids"].to.return_value = fake_inputs["input_ids"]
        fake_inputs["input_ids"].shape = (1, 10)  # prompt_tokens = 10
        fake_inputs["attention_mask"].to.return_value = fake_inputs["attention_mask"]
        mock_tokenizer.return_value = fake_inputs

        # Mock tokenizer decoding
        mock_tokenizer.decode.return_value = "Hello! Apex Retail Solutions is here to help."

        # Create mock model
        mock_model = MagicMock()
        # Output has 10 prompt tokens + 8 completion tokens = 18 tokens
        fake_output_ids = MagicMock()
        fake_output_ids.__getitem__.return_value = list(range(18))
        mock_model.generate.return_value = fake_output_ids

        # Inject mock
        mock_get_model.return_value = (mock_model, mock_tokenizer)

        # Call generation under mocked torch
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            res = generate_peft_response("Can I track my shipment?")

        self.assertTrue(res["success"])
        self.assertIn("Apex Retail Solutions", res["answer"])
        self.assertGreater(res["latency_seconds"], 0.0)
        self.assertEqual(res["prompt_tokens"], 10)
        self.assertEqual(res["completion_tokens"], 8)
        self.assertEqual(res["total_tokens"], 18)
        self.assertGreater(res["tokens_per_second"], 0.0)


class TestFrontendAssets(unittest.TestCase):
    """Test suite verifying presence and elements of the web chat interface."""

    def test_index_html_exists(self):
        index_file = PROJECT_ROOT / "src" / "api" / "static" / "index.html"
        self.assertTrue(index_file.is_file(), "Frontend index.html must exist in src/api/static/")

    def test_index_html_contains_critical_ui_elements(self):
        index_file = PROJECT_ROOT / "src" / "api" / "static" / "index.html"
        content = index_file.read_text(encoding="utf-8")

        # Verify chat message input
        self.assertIn('id="messageInput"', content)
        # Verify send button
        self.assertIn('id="btnSend"', content)
        # Verify chat history container
        self.assertIn('id="chatStream"', content)
        # Verify typing loading indicator
        self.assertIn('id="typingIndicator"', content)
        # Verify status pill / badge
        self.assertIn('id="statusBadge"', content)
        # Verify clear button
        self.assertIn('id="btnClear"', content)
        # Verify error alert banner
        self.assertIn('id="alertBanner"', content)


if __name__ == "__main__":
    unittest.main()
