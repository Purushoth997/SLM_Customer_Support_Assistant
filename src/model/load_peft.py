"""
PEFT Model Loader Module for SLM Customer Support Assistant.
Dynamically loads the pinned Qwen2.5-1.5B-Instruct base model snapshot and attaches
the fine-tuned customer-support-lora adapter on CPU without merging or exporting weights.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_MODEL_NAME_OR_PATH,
    BASE_MODEL_REVISION,
    ADAPTER_DIR,
    EXPECTED_ADAPTER_SHA256,
    DEVICE,
    TORCH_DTYPE,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cached singleton instances
_PEFT_MODEL = None
_PEFT_TOKENIZER = None


def compute_file_sha256(filepath: str | Path) -> str:
    """Calculate the cryptographic SHA-256 hash of a file using buffered streaming."""
    target = Path(filepath)
    if not target.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {target}")

    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1 MB chunks
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_adapter_integrity(adapter_path: Optional[str | Path] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate the existence and cryptographic checksum of the PEFT adapter artifact.
    Returns (is_valid, message, details_dict).
    """
    target_dir = Path(adapter_path or ADAPTER_DIR)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir

    details: Dict[str, Any] = {
        "adapter_dir": str(target_dir),
        "exists": target_dir.is_dir(),
        "config_exists": False,
        "safetensors_exists": False,
        "computed_sha256": None,
        "expected_sha256": EXPECTED_ADAPTER_SHA256,
        "hash_matches": False,
    }

    if not target_dir.is_dir():
        return False, f"Adapter directory does not exist: {target_dir}", details

    config_file = target_dir / "adapter_config.json"
    weights_file = target_dir / "adapter_model.safetensors"

    details["config_exists"] = config_file.is_file()
    details["safetensors_exists"] = weights_file.is_file()

    if not details["config_exists"]:
        return False, f"Missing adapter_config.json in: {target_dir}", details

    if not details["safetensors_exists"]:
        return False, f"Missing adapter_model.safetensors in: {target_dir}", details

    # Compute SHA-256 of adapter weights
    try:
        actual_hash = compute_file_sha256(weights_file)
        details["computed_sha256"] = actual_hash
        details["hash_matches"] = (actual_hash == EXPECTED_ADAPTER_SHA256.lower())

        if not details["hash_matches"]:
            msg = (
                f"Adapter SHA-256 checksum mismatch!\n"
                f"Expected: {EXPECTED_ADAPTER_SHA256}\n"
                f"Actual:   {actual_hash}"
            )
            logger.warning(msg)
            return False, msg, details

        return True, "Adapter verified successfully.", details
    except Exception as e:
        return False, f"Failed calculating adapter checksum: {e}", details


def check_model_assets_readiness() -> Dict[str, Any]:
    """
    Perform a pre-flight inspection of runtime dependencies, base model, and adapter assets.
    Provides diagnostic information for API telemetry and status endpoints.
    """
    # 1. Dependency checks
    missing_deps = []
    for pkg in ["torch", "transformers", "peft", "accelerate", "safetensors"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_deps.append(pkg)

    # 2. Adapter check
    adapter_ok, adapter_msg, adapter_details = verify_adapter_integrity()

    # 3. Base model check (local path or HuggingFace cache)
    base_model_path = Path(BASE_MODEL_NAME_OR_PATH)
    if not base_model_path.is_absolute():
        potential_local = PROJECT_ROOT / base_model_path
    else:
        potential_local = base_model_path

    base_model_local = potential_local.is_dir() and (potential_local / "config.json").is_file()

    # Overall readiness
    ready = (len(missing_deps) == 0) and adapter_ok

    return {
        "ready": ready,
        "missing_dependencies": missing_deps,
        "base_model": BASE_MODEL_NAME_OR_PATH,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_available_locally": base_model_local,
        "adapter_verified": adapter_ok,
        "adapter_message": adapter_msg,
        "adapter_details": adapter_details,
    }


def get_peft_model(
    base_model_name_or_path: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """
    Acquire the singleton (PeftModel, AutoTokenizer) pair on CPU.
    Loads the unquantized base model and dynamically attaches the LoRA adapter.
    Weights remain unmerged.
    """
    global _PEFT_MODEL, _PEFT_TOKENIZER

    if _PEFT_MODEL is not None and _PEFT_TOKENIZER is not None and not force_reload:
        return _PEFT_MODEL, _PEFT_TOKENIZER

    # 1. Verify runtime dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            f"Missing required PEFT runtime dependency: {e.name}. "
            "Please ensure you are running in venv-peft with requirements-peft.txt installed."
        ) from e

    # 2. Verify adapter integrity
    target_adapter = adapter_dir or ADAPTER_DIR
    adapter_ok, msg, details = verify_adapter_integrity(target_adapter)
    if not adapter_ok:
        raise FileNotFoundError(
            f"Cannot load PEFT model: {msg}\n"
            f"Please download the verified customer-support-lora adapter from Google Drive "
            f"into: {details['adapter_dir']}"
        )

    # 3. Resolve base model reference
    target_base = base_model_name_or_path or BASE_MODEL_NAME_OR_PATH
    logger.info("Loading tokenizer from: %s", target_base)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            target_base,
            revision=BASE_MODEL_REVISION,
            trust_remote_code=False,
            padding_side="right",
        )
    except Exception as e:
        logger.error("Failed loading tokenizer for %s: %s", target_base, e)
        raise RuntimeError(
            f"Tokenizer acquisition failed for {target_base}. "
            f"If working offline, ensure base model snapshot is pre-downloaded: {e}"
        ) from e

    # Ensure required pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Determine PyTorch data type for CPU execution
    dtype = torch.float32
    if TORCH_DTYPE.lower() in ("bfloat16", "bf16") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    logger.info("Loading base causal LM: %s (revision: %s, dtype: %s, device: cpu)",
                target_base, BASE_MODEL_REVISION, dtype)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            target_base,
            revision=BASE_MODEL_REVISION,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    except Exception as e:
        logger.error("Failed loading base model %s: %s", target_base, e)
        raise RuntimeError(
            f"Failed loading base model weights for {target_base} (revision: {BASE_MODEL_REVISION}): {e}\n"
            "Please ensure the base model is downloaded locally or Hugging Face Hub is reachable."
        ) from e

    # 5. Dynamically attach LoRA adapter WITHOUT merging
    logger.info("Attaching PEFT LoRA adapter dynamically from: %s", target_adapter)
    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(Path(target_adapter).resolve()),
            is_trainable=False,
        )
        peft_model.eval()  # Set evaluation mode
    except Exception as e:
        logger.error("Failed attaching LoRA adapter: %s", e)
        raise RuntimeError(f"Error attaching PEFT adapter from {target_adapter}: {e}") from e

    logger.info("Successfully initialized PEFT LoRA model. Weights dynamically bound (unmerged).")

    _PEFT_MODEL = peft_model
    _PEFT_TOKENIZER = tokenizer
    return _PEFT_MODEL, _PEFT_TOKENIZER


def reset_peft_model() -> None:
    """Reset cached model and tokenizer instances (for testing and lifecycle reloads)."""
    global _PEFT_MODEL, _PEFT_TOKENIZER
    _PEFT_MODEL = None
    _PEFT_TOKENIZER = None


def is_model_loaded() -> bool:
    """Check if the model and tokenizer are currently instantiated in memory."""
    global _PEFT_MODEL, _PEFT_TOKENIZER
    return (_PEFT_MODEL is not None) and (_PEFT_TOKENIZER is not None)
