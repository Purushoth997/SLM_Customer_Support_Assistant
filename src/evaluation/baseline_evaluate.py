"""
Baseline Evaluation Runner for SLM Customer Support Assistant.
Evaluates Qwen2.5-1.5B-Instruct (GGUF Q4_K_M) across the 50 held-out test records
on pure CPU, records latency and token telemetry, and calculates baseline metrics.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.inference import generate_response
from src.evaluation.metrics import (
    token_f1_score,
    exact_match_score,
    detect_escalation_intent,
    is_valid_response,
    calculate_aggregate_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_evaluation(
    test_split_path: str = "data/splits/test.json",
    output_path: str = "experiments/baseline_results.json",
) -> Dict[str, Any]:
    """Execute evaluation on all records in the held-out test split."""
    test_file = PROJECT_ROOT / test_split_path
    dest = PROJECT_ROOT / output_path

    if not test_file.exists():
        raise FileNotFoundError(f"Held-out test dataset not found: {test_file}")

    with open(test_file, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    logger.info("Loaded %d held-out test records from %s", len(test_records), test_file.name)
    eval_results: List[Dict[str, Any]] = []

    t_start_total = time.time()
    for idx, item in enumerate(test_records, start=1):
        rec_id = item["id"]
        cat = item["category"]
        q = item["question"]
        ctx = item.get("context", "")
        gt = item["expected_answer"]
        req_esc = item.get("requires_escalation", False)
        diff = item.get("difficulty", "medium")

        logger.info("[%d/%d] Evaluating %s (%s): '%s' ...", idx, len(test_records), rec_id, cat, q[:45])

        out = generate_response(question=q, context=ctx)
        pred = out["answer"]

        # Compute record-level metrics
        f1_dict = token_f1_score(pred, gt)
        em = exact_match_score(pred, gt)
        pred_esc = detect_escalation_intent(pred)
        valid = is_valid_response(pred)

        eval_record = {
            "id": rec_id,
            "category": cat,
            "question": q,
            "context": ctx,
            "expected_answer": gt,
            "generated_response": pred,
            "requires_escalation": req_esc,
            "difficulty": diff,
            "latency_seconds": out["latency_seconds"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "total_tokens": out["total_tokens"],
            "tokens_per_second": out["tokens_per_second"],
            "error_status": out["error"],
            "metrics": {
                "token_f1": f1_dict["f1"],
                "token_precision": f1_dict["precision"],
                "token_recall": f1_dict["recall"],
                "exact_match": em,
                "detected_escalation": pred_esc,
                "is_valid": valid,
            },
        }
        eval_results.append(eval_record)

    total_duration = time.time() - t_start_total
    aggregate = calculate_aggregate_metrics(eval_results)
    aggregate["total_evaluation_wallclock_seconds"] = round(total_duration, 2)

    output_payload = {
        "metadata": {
            "model_name": "Qwen2.5-1.5B-Instruct",
            "model_format": "GGUF",
            "quantization": "Q4_K_M",
            "runtime": "llama-cpp-python==0.3.2",
            "device": "CPU (Intel Core i5-12450H)",
            "threads": 8,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_dataset_path": str(test_split_path),
            "test_record_count": len(test_records),
        },
        "aggregate_metrics": aggregate,
        "evaluated_records": eval_results,
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation complete! 50 records evaluated in %.1fs. Results saved to: %s", total_duration, dest)
    return output_payload


if __name__ == "__main__":
    run_baseline_evaluation()
