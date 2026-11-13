"""
Evaluation Metrics Module for SLM Customer Support Assistant.
Calculates token F1, exact match, response validity, escalation behavior classification,
and statistical latency/throughput metrics for baseline and fine-tuned models.
"""

import math
import re
import statistics
from typing import Dict, List, Any, Set


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact match comparison after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level Precision, Recall, and F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        score = 1.0 if pred_tokens == gt_tokens else 0.0
        return {"precision": score, "recall": score, "f1": score}

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def detect_escalation_intent(text: str) -> bool:
    """Detect whether a response contains language indicating escalation to human support."""
    escalation_keywords = [
        r"\bescalat",
        r"\bsupervisor\b",
        r"\bmanager\b",
        r"\bsenior support\b",
        r"\bhuman agent\b",
        r"\blive agent\b",
        r"\blive support\b",
        r"\blive representative\b",
        r"\bhuman representative\b",
        r"\btransferred to\b",
        r"\btransfer you\b",
        r"\brouting you\b",
        r"\bspecialized team\b",
        r"\bfraud team\b",
        r"\bsecurity team\b",
        r"\blegal team\b",
        r"\bclaims adjuster\b",
        r"\badvocacy\b",
    ]
    pattern = re.compile("|".join(escalation_keywords), re.IGNORECASE)
    return bool(pattern.search(text))


def is_valid_response(text: str) -> bool:
    """Check if generated response is a non-empty, substantive, well-formed reply."""
    if not text or not isinstance(text, str):
        return False
    words = text.strip().split()
    if len(words) < 5:
        return False
    # Check if repetitive garbage loop
    if len(words) > 30 and len(set(words)) < 5:
        return False
    return True


def calculate_aggregate_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistical, token, and behavioral metrics across all evaluated records."""
    total = len(eval_records)
    if total == 0:
        return {"total_records": 0}

    successful = [r for r in eval_records if r.get("error_status") is None and r.get("generated_response")]
    failed = [r for r in eval_records if r.get("error_status") is not None]

    latencies = [r["latency_seconds"] for r in successful if "latency_seconds" in r]
    tokens_per_sec = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second", 0) > 0]
    completion_tokens = [r["completion_tokens"] for r in successful if "completion_tokens" in r]

    f1_scores = []
    em_scores = []
    valid_count = 0

    # Escalation accuracy tracking
    tp_esc = 0  # Expected True, Predicted True
    fp_esc = 0  # Expected False, Predicted True
    fn_esc = 0  # Expected True, Predicted False
    tn_esc = 0  # Expected False, Predicted False

    category_stats: Dict[str, Dict[str, Any]] = {}

    for r in successful:
        pred = r["generated_response"]
        gt = r["expected_answer"]
        req_esc = r.get("requires_escalation", False)
        cat = r.get("category", "unknown")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "f1_sum": 0.0, "valid": 0, "latencies": []}

        category_stats[cat]["total"] += 1
        category_stats[cat]["latencies"].append(r["latency_seconds"])

        # Validity
        if is_valid_response(pred):
            valid_count += 1
            category_stats[cat]["valid"] += 1

        # Token metrics
        f1_res = token_f1_score(pred, gt)
        f1_scores.append(f1_res["f1"])
        category_stats[cat]["f1_sum"] += f1_res["f1"]

        em = exact_match_score(pred, gt)
        em_scores.append(em)

        # Escalation detection
        pred_esc = detect_escalation_intent(pred)
        if req_esc and pred_esc:
            tp_esc += 1
        elif not req_esc and pred_esc:
            fp_esc += 1
        elif req_esc and not pred_esc:
            fn_esc += 1
        else:
            tn_esc += 1

    esc_acc = round((tp_esc + tn_esc) / total, 4) if total > 0 else 0.0
    esc_precision = round(tp_esc / (tp_esc + fp_esc), 4) if (tp_esc + fp_esc) > 0 else 0.0
    esc_recall = round(tp_esc / (tp_esc + fn_esc), 4) if (tp_esc + fn_esc) > 0 else 0.0
    esc_f1 = round((2 * esc_precision * esc_recall) / (esc_precision + esc_recall), 4) if (esc_precision + esc_recall) > 0 else 0.0

    # Aggregate category metrics
    category_summary = {}
    for cat, s in category_stats.items():
        n = s["total"]
        category_summary[cat] = {
            "record_count": n,
            "mean_f1": round(s["f1_sum"] / n, 4) if n > 0 else 0.0,
            "validity_rate": round(s["valid"] / n * 100, 2) if n > 0 else 0.0,
            "avg_latency_s": round(statistics.mean(s["latencies"]), 3) if s["latencies"] else 0.0,
        }

    return {
        "total_test_records": total,
        "successful_inferences": len(successful),
        "failed_inferences": len(failed),
        "response_validity_rate_pct": round(valid_count / total * 100, 2),
        "mean_token_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
        "mean_exact_match": round(statistics.mean(em_scores), 4) if em_scores else 0.0,
        "latency_stats": {
            "mean_seconds": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "median_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
            "min_seconds": round(min(latencies), 3) if latencies else 0.0,
            "max_seconds": round(max(latencies), 3) if latencies else 0.0,
            "stdev_seconds": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
        },
        "throughput_stats": {
            "mean_tokens_per_second": round(statistics.mean(tokens_per_sec), 2) if tokens_per_sec else 0.0,
            "median_tokens_per_second": round(statistics.median(tokens_per_sec), 2) if tokens_per_sec else 0.0,
            "min_tokens_per_second": round(min(tokens_per_sec), 2) if tokens_per_sec else 0.0,
            "max_tokens_per_second": round(max(tokens_per_sec), 2) if tokens_per_sec else 0.0,
        },
        "token_usage_stats": {
            "mean_completion_tokens": round(statistics.mean(completion_tokens), 1) if completion_tokens else 0.0,
            "median_completion_tokens": round(statistics.median(completion_tokens), 1) if completion_tokens else 0.0,
            "total_completion_tokens": sum(completion_tokens),
        },
        "escalation_metrics": {
            "accuracy": esc_acc,
            "precision": esc_precision,
            "recall": esc_recall,
            "f1": esc_f1,
            "true_positives": tp_esc,
            "false_positives": fp_esc,
            "true_negatives": tn_esc,
            "false_negatives": fn_esc,
        },
        "category_breakdown": category_summary,
    }
