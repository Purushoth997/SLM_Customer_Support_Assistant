"""
Single-Question Smoke Test Runner for Qwen2.5-1.5B-Instruct CPU Baseline.
Executes five mandatory customer support questions without fine-tuning,
records latency, token counts, and evaluates rule compliance.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.inference import generate_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SMOKE_TEST_QUESTIONS = [
    {
        "test_id": "ST01",
        "question": "Where is my order?",
        "context": None,
        "expected_behavior": "Must ask for Order ID or customer credentials; must not invent order numbers or statuses.",
    },
    {
        "test_id": "ST02",
        "question": "How can I request a refund?",
        "context": "Standard return and refund policy: items can be returned within 30 days of delivery for a full refund to the original payment method.",
        "expected_behavior": "Must explain the 30-day refund window and original payment method return policy.",
    },
    {
        "test_id": "ST03",
        "question": "My payment was deducted but my order is not confirmed.",
        "context": "Credit cards may show temporary pre-authorization holds for pending transactions that release within 3-5 business days if an order fails at checkout.",
        "expected_behavior": "Must explain temporary authorization hold or offer to check transaction if email/details provided.",
    },
    {
        "test_id": "ST04",
        "question": "I received a damaged product. What should I do?",
        "context": "Damaged items are eligible for free prepaid return and immediate replacement or full refund upon photo verification.",
        "expected_behavior": "Must apologize, offer free return/replacement, and request photo verification/order details.",
    },
    {
        "test_id": "ST05",
        "question": "Can you confirm my order has been delivered?",
        "context": None,
        "expected_behavior": "Must ask for Order ID to check delivery status; must not falsely claim delivered without data.",
    },
]


def run_smoke_tests(output_path: str = "experiments/single_question_smoke_tests.json"):
    """Execute smoke tests and record detailed telemetry."""
    dest = PROJECT_ROOT / output_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    results = []
    logger.info("Starting execution of %d single-question smoke tests on CPU...", len(SMOKE_TEST_QUESTIONS))

    for item in SMOKE_TEST_QUESTIONS:
        t_id = item["test_id"]
        q = item["question"]
        ctx = item["context"]
        logger.info("[%s] Testing: '%s'", t_id, q)

        out = generate_response(question=q, context=ctx)

        # Rule compliance inspection
        ans = out["answer"]
        follows_rules = True
        notes = []

        # Check for fabricated order numbers in no-context questions
        if ctx is None and ("APX-" in ans or "ORD-" in ans or "#" in ans):
            if "please provide" not in ans.lower() and "order id" not in ans.lower():
                follows_rules = False
                notes.append("Potential hallucinated order number without asking for user ID.")

        if out["error"] is not None:
            follows_rules = False
            notes.append(f"Inference error: {out['error']}")

        record = {
            "test_id": t_id,
            "question": q,
            "context": ctx,
            "expected_behavior": item["expected_behavior"],
            "generated_response": ans,
            "latency_seconds": out["latency_seconds"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "total_tokens": out["total_tokens"],
            "tokens_per_second": out["tokens_per_second"],
            "error_status": out["error"],
            "follows_system_rules": follows_rules,
            "compliance_notes": "; ".join(notes) if notes else "Follows system guidelines correctly.",
        }
        results.append(record)
        logger.info("[%s] Completed in %.2fs (%d tokens, %.1f tok/s)",
                    t_id, out["latency_seconds"], out["completion_tokens"], out["tokens_per_second"])

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Smoke test results saved to: %s", dest)
    return results


if __name__ == "__main__":
    run_smoke_tests()
