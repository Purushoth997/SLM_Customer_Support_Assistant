"""
Inference Module for SLM Customer Support Assistant.
Executes local CPU chat inference using Qwen2.5-1.5B-Instruct in GGUF format
via llama-cpp-python, with input validation, latency profiling, and token accounting.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.load_model import get_model, load_model_config
from src.model.prompt_template import build_chat_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 2000


def generate_response(
    question: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a customer support response for a given question and context.
    Executes actual model inference through the local GGUF model on CPU.
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
            "model_name": "Qwen2.5-1.5B-Instruct",
            "error": "Empty or whitespace-only question received.",
        }

    clean_question = str(question).strip()
    if len(clean_question) > MAX_QUESTION_LENGTH:
        return {
            "success": False,
            "answer": f"Your inquiry exceeds the maximum allowed length of {MAX_QUESTION_LENGTH} characters. Please summarize your question.",
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": "Qwen2.5-1.5B-Instruct",
            "error": "Question exceeds maximum character limit.",
        }

    # 2. Load model and configuration
    try:
        config = load_model_config()
        model = get_model()
    except Exception as e:
        logger.error("Failed to acquire model instance: %s", e)
        return {
            "success": False,
            "answer": "The customer support AI engine is currently unavailable. Please contact support via email or phone.",
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": "Qwen2.5-1.5B-Instruct",
            "error": f"Model loading failure: {str(e)}",
        }

    # 3. Generation parameters
    gen_cfg = config.get("generation", {})
    t = temperature if temperature is not None else float(gen_cfg.get("temperature", 0.3))
    m = max_tokens if max_tokens is not None else int(gen_cfg.get("max_tokens", 384))
    p = top_p if top_p is not None else float(gen_cfg.get("top_p", 0.85))
    k = top_k if top_k is not None else int(gen_cfg.get("top_k", 40))
    rp = repeat_penalty if repeat_penalty is not None else float(gen_cfg.get("repeat_penalty", 1.15))
    stop_tokens = gen_cfg.get("stop", ["<|im_end|>", "<|endoftext|>", "<|im_start|>"])

    # 4. Construct messages
    messages = build_chat_messages(clean_question, context, system_prompt)

    # 5. Execute CPU inference with precise latency measurement
    t0 = time.perf_counter()
    try:
        response = model.create_chat_completion(
            messages=messages,
            temperature=t,
            max_tokens=m,
            top_p=p,
            top_k=k,
            repeat_penalty=rp,
            stop=stop_tokens,
        )
        latency = time.perf_counter() - t0
    except Exception as e:
        latency = time.perf_counter() - t0
        logger.error("Inference execution failed: %s", e)
        return {
            "success": False,
            "answer": "An unexpected error occurred while generating a response. Please try again shortly.",
            "latency_seconds": round(latency, 4),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0.0,
            "model_name": config.get("model", {}).get("name", "Qwen2.5-1.5B-Instruct"),
            "error": f"Generation exception: {str(e)}",
        }

    # 6. Extract results and token statistics
    choice = response["choices"][0]
    answer_text = choice["message"]["content"].strip()
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    tok_per_sec = round(completion_tokens / latency, 2) if latency > 0 and completion_tokens > 0 else 0.0

    return {
        "success": True,
        "answer": answer_text,
        "latency_seconds": round(latency, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tok_per_sec,
        "model_name": config.get("model", {}).get("name", "Qwen2.5-1.5B-Instruct"),
        "generation_params": {
            "temperature": t,
            "max_tokens": m,
            "top_p": p,
            "top_k": k,
            "repeat_penalty": rp,
        },
        "error": None,
    }
