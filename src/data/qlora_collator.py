"""
Completion-Only ChatML Data Formatting & Collation Module for Qwen2.5 QLoRA Fine-Tuning.

Responsibilities:
1. Load ONLY approved `train.json` and `validation.json` splits while enforcing strict
   quarantine blocking any access to held-out `test` data.
2. Format examples using the exact production ChatML prompt structure (`src/model/prompt_template.py`).
3. Perform deterministic token-boundary-based label masking (`IGNORE_INDEX = -100`) on all
   system/user prompt tokens up to and including `<|im_start|>assistant\n`, keeping ONLY
   assistant completion tokens (including `<|im_end|>`) trainable.
4. Pad batches on the right with `<|endoftext|>` (`pad_token_id != eos_token_id`) and mask
   all padding positions with `-100`, while verifying truncation never removes assistant responses.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import validate_record
from src.model.prompt_template import build_chat_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100
QWEN_PAD_TOKEN: str = "<|endoftext|>"
QWEN_IM_START_TOKEN: str = "<|im_start|>"
QWEN_IM_END_TOKEN: str = "<|im_end|>"
QWEN_ASSISTANT_HEADER: str = "<|im_start|>assistant\n"
FORBIDDEN_RAW_SPECIAL_TOKENS: Tuple[str, ...] = (
    QWEN_PAD_TOKEN,
    QWEN_IM_START_TOKEN,
    QWEN_IM_END_TOKEN,
)
ALLOWED_TRAINING_SPLITS = {"train", "validation"}


def compute_file_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hex digest of a file without modifying it."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def assert_not_test_split(file_path: Path) -> None:
    """
    Enforce strict quarantine against loading or referencing any held-out test split.
    Raises PermissionError if the filename or parent path references a test split.
    """
    lower_name = file_path.name.lower()
    lower_stem = file_path.stem.lower()
    if "test" in lower_stem or lower_name == "test.json":
        raise PermissionError(
            f"Dataset Quarantine Violation: Access to held-out evaluation file '{file_path}' "
            "is strictly prohibited in training and validation pipelines."
        )


def load_approved_split(
    split_path: str | Path,
    expected_split: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load and validate ONLY `train.json` or `validation.json` and return `(records, sha256)`.
    Fails safely on quarantine violations, missing files, or malformed records.
    """
    if expected_split not in ALLOWED_TRAINING_SPLITS:
        raise ValueError(
            f"Invalid split role '{expected_split}'. Allowed roles: {sorted(ALLOWED_TRAINING_SPLITS)}"
        )

    resolved = Path(split_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    assert_not_test_split(resolved)

    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {resolved}")

    file_sha256 = compute_file_sha256(resolved)

    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"Dataset split '{resolved}' must be a non-empty JSON list of records.")

    seen_ids = set()
    for idx, rec in enumerate(payload):
        is_valid, issues = validate_record(rec, idx)
        if not is_valid:
            raise ValueError(f"Malformed record at index {idx} in '{resolved.name}': {issues}")

        rec_id = rec["id"]
        if rec_id in seen_ids:
            raise ValueError(f"Duplicate record ID '{rec_id}' detected in '{resolved.name}'.")
        seen_ids.add(rec_id)

        for field_name in ("question", "context", "expected_answer"):
            val = str(rec.get(field_name, ""))
            for sp in FORBIDDEN_RAW_SPECIAL_TOKENS:
                if sp in val:
                    raise ValueError(
                        f"Special ChatML token '{sp}' found inside '{field_name}' of record '{rec_id}'."
                    )

    logger.info(
        "Loaded verified '%s' split (%d records, sha256=%s...) from %s",
        expected_split,
        len(payload),
        file_sha256[:12],
        resolved.name,
    )
    return payload, file_sha256


def configure_qwen_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """
    Validate and configure a Qwen2.5 ChatML tokenizer for completion-only QLoRA training.
    Guarantees `<|endoftext|>` is used for padding and `<|im_end|>` remains a distinct EOS token.
    """
    if getattr(tokenizer, "pad_token", None) != QWEN_PAD_TOKEN:
        tokenizer.pad_token = QWEN_PAD_TOKEN

    tokenizer.padding_side = "right"

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None or not isinstance(pad_token_id, int):
        raise ValueError("Tokenizer pad_token_id could not be resolved to an integer.")
    if eos_token_id is None or not isinstance(eos_token_id, int):
        raise ValueError("Tokenizer eos_token_id could not be resolved to an integer.")
    if pad_token_id == eos_token_id:
        raise ValueError(
            f"Unsafe tokenizer configuration: pad_token_id ({pad_token_id}) equals "
            f"eos_token_id ({eos_token_id}). This would mask end-of-turn tokens during training."
        )

    assistant_header_ids = encode_without_special(tokenizer, QWEN_ASSISTANT_HEADER)
    if not assistant_header_ids:
        raise ValueError("Failed to encode Qwen ChatML assistant header '<|im_start|>assistant\\n'.")

    im_end_ids = encode_without_special(tokenizer, QWEN_IM_END_TOKEN)
    if len(im_end_ids) != 1 or im_end_ids[0] != eos_token_id:
        raise ValueError(
            f"Expected '{QWEN_IM_END_TOKEN}' to tokenize to single eos_token_id={eos_token_id}, "
            f"got {im_end_ids}."
        )

    return {
        "pad_token": QWEN_PAD_TOKEN,
        "pad_token_id": pad_token_id,
        "eos_token": QWEN_IM_END_TOKEN,
        "eos_token_id": eos_token_id,
        "assistant_header_ids": list(assistant_header_ids),
    }


def encode_without_special(tokenizer: Any, text: str) -> List[int]:
    """Encode text into token IDs without prepending/appending automatic BOS/EOS tokens."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(t) for t in encoded]


def render_qwen_prompt_prefix(
    question: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Render the exact Qwen2.5 ChatML prompt prefix up to `<|im_start|>assistant\n`.
    Matches `src/model/prompt_template.py` and Qwen2.5's `apply_chat_template(..., add_generation_prompt=True)`.
    """
    messages = build_chat_messages(question=question, context=context, system_prompt=system_prompt)
    return (
        f"{QWEN_IM_START_TOKEN}system\n{messages[0]['content']}{QWEN_IM_END_TOKEN}\n"
        f"{QWEN_IM_START_TOKEN}user\n{messages[1]['content']}{QWEN_IM_END_TOKEN}\n"
        f"{QWEN_ASSISTANT_HEADER}"
    )


def find_assistant_completion_start(
    input_ids: Sequence[int],
    assistant_header_ids: Sequence[int],
) -> int:
    """
    Locate the first token index of the assistant's completion immediately following the
    LAST occurrence of `assistant_header_ids` (`<|im_start|>assistant\n`).

    Raises ValueError if the header boundary is absent or no completion tokens follow it.
    """
    seq_len = len(input_ids)
    hdr_len = len(assistant_header_ids)
    if hdr_len == 0 or seq_len <= hdr_len:
        raise ValueError(
            f"Sequence length ({seq_len}) is too short to contain assistant header ({hdr_len} tokens) "
            "and a valid completion."
        )

    target = list(assistant_header_ids)
    seq = list(input_ids)

    last_match_idx = -1
    for idx in range(seq_len - hdr_len + 1):
        if seq[idx : idx + hdr_len] == target:
            last_match_idx = idx

    if last_match_idx == -1:
        raise ValueError(
            "Qwen ChatML assistant turn boundary '<|im_start|>assistant\\n' not found in token sequence."
        )

    completion_start = last_match_idx + hdr_len
    if completion_start >= seq_len:
        raise ValueError(
            "Assistant response was completely removed or empty after '<|im_start|>assistant\\n'."
        )

    return completion_start


def tokenize_support_record(
    record: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 512,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate and tokenize a single customer support example into Qwen2.5 ChatML `input_ids`,
    respecting `max_seq_length` while guaranteeing the assistant completion and `<|im_end|>`
    end-of-turn token are preserved.
    """
    if max_seq_length < 16:
        raise ValueError(f"max_seq_length must be >= 16, got {max_seq_length}.")
    if not isinstance(record, dict):
        raise ValueError("Training example must be a dictionary.")

    question = record.get("question")
    expected_answer = record.get("expected_answer")
    context = record.get("context", "")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Record '{record.get('id', '<unknown>')}' has an empty or invalid 'question'.")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        raise ValueError(
            f"Record '{record.get('id', '<unknown>')}' has an empty or invalid 'expected_answer'."
        )

    clean_answer = expected_answer.strip()
    for field_name, field_val in (
        ("question", question),
        ("context", str(context or "")),
        ("expected_answer", clean_answer),
    ):
        for sp in FORBIDDEN_RAW_SPECIAL_TOKENS:
            if sp in field_val:
                raise ValueError(
                    f"Forbidden special token '{sp}' found in '{field_name}' of record '{record.get('id')}'."
                )

    tok_meta = configure_qwen_tokenizer(tokenizer)
    eos_token_id = tok_meta["eos_token_id"]
    assistant_header_ids = tok_meta["assistant_header_ids"]

    prompt_prefix = render_qwen_prompt_prefix(
        question=question,
        context=context,
        system_prompt=system_prompt,
    )
    prompt_ids = encode_without_special(tokenizer, prompt_prefix)
    answer_body_ids = encode_without_special(tokenizer, clean_answer)

    if not answer_body_ids:
        raise ValueError(
            f"Record '{record.get('id', '<unknown>')}' produced zero assistant body tokens."
        )

    if prompt_ids[-len(assistant_header_ids) :] != assistant_header_ids:
        raise ValueError(
            "Tokenized prompt prefix does not end with exact '<|im_start|>assistant\\n' token IDs."
        )

    # Require room for the full prompt plus at least 1 assistant content token and `<|im_end|>`
    min_required_length = len(prompt_ids) + 2
    if min_required_length > max_seq_length:
        raise ValueError(
            f"Truncation to max_seq_length={max_seq_length} would remove the assistant response "
            f"for record '{record.get('id', '<unknown>')}' (prompt requires {len(prompt_ids)} tokens)."
        )

    completion_ids = list(answer_body_ids)
    if completion_ids[-1] != eos_token_id:
        completion_ids.append(eos_token_id)

    was_truncated = False
    if len(prompt_ids) + len(completion_ids) > max_seq_length:
        # Preserve prompt + maximum fitting assistant tokens + mandatory closing `<|im_end|>`
        allowed_body_len = max_seq_length - len(prompt_ids) - 1
        completion_ids = list(answer_body_ids[:allowed_body_len]) + [eos_token_id]
        was_truncated = True

    input_ids = prompt_ids + completion_ids
    attention_mask = [1] * len(input_ids)

    # Final boundary sanity check on the assembled sequence
    completion_start = find_assistant_completion_start(input_ids, assistant_header_ids)
    if completion_start != len(prompt_ids):
        raise ValueError(
            f"Unexpected assistant completion boundary shift ({completion_start} != {len(prompt_ids)})."
        )

    return {
        "id": record.get("id", ""),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_token_count": len(prompt_ids),
        "completion_token_count": len(completion_ids),
        "was_truncated": was_truncated,
    }


def prepare_tokenized_dataset(
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 512,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tokenize and validate a full split of customer support records."""
    if not records:
        raise ValueError("Cannot prepare an empty dataset split.")

    tokenized: List[Dict[str, Any]] = []
    for rec in records:
        tokenized.append(
            tokenize_support_record(
                record=rec,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                system_prompt=system_prompt,
            )
        )
    return tokenized


class QLoRACompletionCollator:
    """
    Data collator for Qwen2.5 ChatML instruction tuning with completion-only loss masking.

    For each example in a batch:
    - Locates `<|im_start|>assistant\n` by exact token-ID subsequence matching.
    - Sets `labels = -100` (`IGNORE_INDEX`) for all system/user prompt tokens (including the header).
    - Keeps all assistant completion tokens (including `<|im_end|>`) trainable in `labels`.
    - Right-pads `input_ids` with `<|endoftext|>` (`pad_token_id`), `attention_mask` with `0`,
      and `labels` with `-100`.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 512,
        ignore_index: int = IGNORE_INDEX,
        return_tensors: str = "pt",
    ) -> None:
        if max_seq_length < 16:
            raise ValueError(f"max_seq_length must be >= 16, got {max_seq_length}.")
        if return_tensors not in {"pt", "np", "py"}:
            raise ValueError(f"Unsupported return_tensors='{return_tensors}'. Use 'pt', 'np', or 'py'.")

        tok_meta = configure_qwen_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        self.return_tensors = return_tensors

        self.pad_token_id: int = tok_meta["pad_token_id"]
        self.eos_token_id: int = tok_meta["eos_token_id"]
        self.assistant_header_ids: List[int] = tok_meta["assistant_header_ids"]

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("QLoRACompletionCollator received an empty batch.")

        processed_inputs: List[List[int]] = []
        processed_masks: List[List[int]] = []
        processed_labels: List[List[int]] = []

        for idx, feat in enumerate(features):
            if not isinstance(feat, dict) or "input_ids" not in feat:
                raise ValueError(f"Batch feature at index {idx} is missing 'input_ids'.")

            raw_ids = [int(t) for t in feat["input_ids"]]
            if not raw_ids:
                raise ValueError(f"Batch feature at index {idx} has empty 'input_ids'.")

            # Enforce max_seq_length safely without ever deleting the assistant completion
            if len(raw_ids) > self.max_seq_length:
                comp_start_check = find_assistant_completion_start(raw_ids, self.assistant_header_ids)
                if comp_start_check + 2 > self.max_seq_length:
                    raise ValueError(
                        f"Batch feature at index {idx} exceeds max_seq_length={self.max_seq_length} "
                        "and truncation would remove the assistant response."
                    )
                raw_ids = raw_ids[: self.max_seq_length - 1] + [self.eos_token_id]

            completion_start = find_assistant_completion_start(raw_ids, self.assistant_header_ids)
            seq_len = len(raw_ids)

            labels = [self.ignore_index] * completion_start + raw_ids[completion_start:seq_len]
            trainable_tokens = [tok for tok in labels if tok != self.ignore_index]

            # Must contain at least 1 content token + `<|im_end|>` EOS token
            if len(trainable_tokens) < 2:
                raise ValueError(
                    f"Batch feature at index {idx} has insufficient trainable assistant tokens "
                    f"({len(trainable_tokens)})."
                )
            if labels[-1] != self.eos_token_id:
                raise ValueError(
                    f"Batch feature at index {idx} is missing trainable end-of-turn token "
                    f"eos_token_id={self.eos_token_id} at end of completion."
                )

            att_mask = [1] * seq_len
            processed_inputs.append(raw_ids)
            processed_masks.append(att_mask)
            processed_labels.append(labels)

        batch_max_len = max(len(seq) for seq in processed_inputs)

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for ids, mask, lbls in zip(processed_inputs, processed_masks, processed_labels):
            pad_len = batch_max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(lbls + [self.ignore_index] * pad_len)

        if self.return_tensors == "py":
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
            }

        if self.return_tensors == "np":
            import numpy as np

            return {
                "input_ids": np.asarray(batch_input_ids, dtype=np.int64),
                "attention_mask": np.asarray(batch_attention_mask, dtype=np.int64),
                "labels": np.asarray(batch_labels, dtype=np.int64),
            }

        import torch

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
