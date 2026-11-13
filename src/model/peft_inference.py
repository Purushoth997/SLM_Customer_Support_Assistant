"""
PEFT Inference Service for SLM Customer Support Assistant.
Executes CPU chat generation using dynamically loaded Qwen2.5-1.5B-Instruct + LoRA adapter
via Hugging Face Transformers and PEFT, with ChatML prompt formatting, latency profiling,
and token accounting.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_MODEL_NAME_OR_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    MAX_QUESTION_LENGTH,
    ALLOW_MOCK_FALLBACK,
)
from src.model.load_peft import get_peft_model, check_model_assets_readiness
from src.model.prompt_template import DEFAULT_SYSTEM_PROMPT, format_user_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_full_conversation_messages(
    question: str,
    context: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Construct multi-turn ChatML messages list, preserving past turns and appending
    the current customer question with grounded business context.
    """
    messages: List[Dict[str, str]] = []

    # 1. System persona prompt
    sys_content = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    messages.append({"role": "system", "content": sys_content})

    # 2. Historical conversation turns
    if history:
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # 3. Current inquiry with optional context
    latest_user_content = format_user_prompt(question, context)
    messages.append({"role": "user", "content": latest_user_content})

    return messages


def generate_peft_response(
    question: str,
    context: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute customer support response generation using the PEFT-adapted model on CPU.
    """
    # 1. Input validation
    if not question or not str(question).strip():
        return {
            "success": False,
            "answer": "Please provide a valid question so I can assist you.",
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": BASE_MODEL_NAME_OR_PATH,
            "adapter_name": "customer-support-lora",
            "device": "cpu",
            "error": "Empty or whitespace-only question received.",
        }

    clean_question = str(question).strip()
    if len(clean_question) > MAX_QUESTION_LENGTH:
        return {
            "success": False,
            "answer": (
                f"Your inquiry exceeds the maximum allowed length of {MAX_QUESTION_LENGTH} characters. "
                "Please summarize your question."
            ),
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": BASE_MODEL_NAME_OR_PATH,
            "adapter_name": "customer-support-lora",
            "device": "cpu",
            "error": "Question exceeds maximum character limit.",
        }

    # 2. Acquire model and tokenizer
    try:
        model, tokenizer = get_peft_model()
    except Exception as e:
        logger.warning("PEFT model acquisition unavailable: %s", e)

        # Optional mock fallback for testing without full downloaded weights
        if ALLOW_MOCK_FALLBACK:
            logger.info("Serving synthetic response via ALLOW_MOCK_FALLBACK")
            return {
                "success": True,
                "answer": (
                    "[Mock Mode] Thank you for contacting Apex Retail Solutions. "
                    f"Received your inquiry: '{clean_question}'. "
                    "Model weights are currently in mock mode."
                ),
                "latency_seconds": 0.05,
                "prompt_tokens": 24,
                "completion_tokens": 28,
                "total_tokens": 52,
                "tokens_per_second": 560.0,
                "model_name": BASE_MODEL_NAME_OR_PATH,
                "adapter_name": "customer-support-lora (mock)",
                "device": "cpu",
                "error": None,
            }

        return {
            "success": False,
            "answer": (
                "The fine-tuned customer support model is not currently loaded. "
                "Please ensure the base model snapshot and customer-support-lora adapter "
                "are downloaded and verified in exports/customer-support-lora."
            ),
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": BASE_MODEL_NAME_OR_PATH,
            "adapter_name": "customer-support-lora",
            "device": "cpu",
            "error": f"Model loading failure: {str(e)}",
        }

    # 3. Generation hyperparameter resolution
    t = temperature if temperature is not None else DEFAULT_TEMPERATURE
    m = max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    p = top_p if top_p is not None else DEFAULT_TOP_P
    k = top_k if top_k is not None else DEFAULT_TOP_K
    rp = repetition_penalty if repetition_penalty is not None else DEFAULT_REPETITION_PENALTY

    # 4. Construct messages & apply ChatML template
    messages = build_full_conversation_messages(
        clean_question, context=context, history=history, system_prompt=system_prompt
    )

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Manual ChatML fallback if apply_chat_template lacks jinja config
        formatted_prompt = ""
        for msg in messages:
            formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

    # 5. Tokenize input prompt on CPU
    import torch

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cpu")
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")

    prompt_tokens = input_ids.shape[1]

    # 6. Execute CPU generation with precise latency profiling
    do_sample = (t > 0.0)
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": m,
        "do_sample": do_sample,
        "repetition_penalty": rp,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = t
        gen_kwargs["top_p"] = p
        gen_kwargs["top_k"] = k

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
        latency = time.perf_counter() - t0
    except Exception as e:
        latency = time.perf_counter() - t0
        logger.error("Generation exception occurred: %s", e)
        return {
            "success": False,
            "answer": "An error occurred during response generation. Please try again shortly.",
            "latency_seconds": round(latency, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "tokens_per_second": 0.0,
            "model_name": BASE_MODEL_NAME_OR_PATH,
            "adapter_name": "customer-support-lora",
            "device": "cpu",
            "error": f"Generation error: {str(e)}",
        }

    # 7. Decode generated completion (slice off prompt tokens)
    completion_ids = output_ids[0][prompt_tokens:]
    completion_tokens = len(completion_ids)
    total_tokens = prompt_tokens + completion_tokens

    answer_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    tok_per_sec = round(completion_tokens / latency, 2) if latency > 0 and completion_tokens > 0 else 0.0

    return {
        "success": True,
        "answer": answer_text,
        "latency_seconds": round(latency, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tok_per_sec,
        "model_name": BASE_MODEL_NAME_OR_PATH,
        "adapter_name": "customer-support-lora",
        "device": "cpu",
        "error": None,
    }
