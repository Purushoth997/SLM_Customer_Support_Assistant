"""
Cloud-GPU QLoRA Fine-Tuning Pipeline for Qwen2.5-1.5B-Instruct.

Target Platform: Google Colab Free (NVIDIA T4, FP16) or Ampere+ GPU (BF16).
Safety Guarantees:
- Never runs on CPU (`local_cpu_training_allowed: false`).
- Dynamically selects FP16 (Turing T4) vs BF16 (Ampere+ with native BF16) and never mixes
  BF16 compute with FP16 GradScaler.
- Loads ONLY `train.json` and `validation.json` (strictly blocks `test` splits).
- Computes loss ONLY on assistant completion tokens via `QLoRACompletionCollator`.
- Requires explicit `--confirm-train` flag to execute training; `--dry-run` validates locally.
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packaging.version import Version
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlora_collator import (
    QLoRACompletionCollator,
    compute_file_sha256,
    configure_qwen_tokenizer,
    load_approved_split,
    prepare_tokenized_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML SafeLoader that raises ValueError on duplicate keys within any mapping."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line_num = key_node.start_mark.line + 1 if key_node.start_mark else "unknown"
            raise ValueError(
                f"Duplicate YAML configuration key '{key}' detected at line {line_num}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def assert_globally_unique_config_keys(
    cfg: Dict[str, Any],
    source_name: str = "training_config.yaml",
) -> None:
    """Ensure every configuration key appears at most once across the entire YAML hierarchy."""
    seen_paths: Dict[str, str] = {}

    def _walk(mapping: Dict[str, Any], prefix: str) -> None:
        for key, val in mapping.items():
            key_str = str(key)
            cur_path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in seen_paths:
                raise ValueError(
                    f"Duplicate configuration key '{key_str}' encountered in '{source_name}' "
                    f"at both '{seen_paths[key_str]}' and '{cur_path}'."
                )
            seen_paths[key_str] = cur_path
            if isinstance(val, dict):
                _walk(val, cur_path)

    if isinstance(cfg, dict):
        _walk(cfg, "")


def parse_unique_yaml(yaml_content: str, source_name: str = "training_config.yaml") -> Dict[str, Any]:
    """Parse YAML text while rejecting both intra-section and cross-section duplicate keys."""
    cfg = yaml.load(yaml_content, Loader=_UniqueKeySafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration '{source_name}' must parse to a YAML dictionary.")
    assert_globally_unique_config_keys(cfg, source_name=source_name)
    return cfg


def load_training_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load, duplicate-check, and validate YAML training configuration."""
    target = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "training_config.yaml"
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        raise FileNotFoundError(f"Training configuration file not found: {target}")

    raw_yaml = target.read_text(encoding="utf-8")
    cfg = parse_unique_yaml(raw_yaml, source_name=target.name)
    validate_training_config(cfg)
    return cfg


def validate_training_config(cfg: Dict[str, Any]) -> None:
    """
    Validate all configuration blocks so invalid, unsafe, or conflicting settings fail fast
    without ever silently altering hyperparameters.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Training configuration must parse to a YAML dictionary.")

    for section in ("base_model", "lora", "datasets", "training_hyperparameters", "execution_environment"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing required configuration section: '{section}'.")

    env_cfg = cfg["execution_environment"]
    if env_cfg.get("local_cpu_training_allowed", False) is not False:
        raise ValueError("Safety violation: 'local_cpu_training_allowed' must be false.")

    base_cfg = cfg["base_model"]
    if str(base_cfg.get("repo_id", "")).endswith(".gguf"):
        raise ValueError("Cannot run PEFT LoRA training directly on a GGUF inference file.")

    # Guard against simultaneous static fp16 + bf16 conflict
    hp = cfg["training_hyperparameters"]
    if hp.get("fp16") is True and hp.get("bf16") is True:
        raise ValueError("Conflicting configuration: both 'fp16' and 'bf16' cannot be true.")
    if str(base_cfg.get("torch_dtype", "")).lower() == "bfloat16" and hp.get("fp16") is True:
        raise ValueError(
            "Incompatible precision configuration: base_model.torch_dtype='bfloat16' cannot be "
            "combined with fp16=true GradScaler."
        )

    lora_cfg = cfg["lora"]
    if int(lora_cfg.get("r", 0)) <= 0 or int(lora_cfg.get("lora_alpha", 0)) <= 0:
        raise ValueError("LoRA rank 'r' and 'lora_alpha' must be positive integers.")
    if list(lora_cfg.get("target_modules", [])) != REQUIRED_TARGET_MODULES:
        raise ValueError(
            f"LoRA target_modules must match approved Qwen2 modules {REQUIRED_TARGET_MODULES}, "
            f"got {lora_cfg.get('target_modules')}."
        )

    ds_cfg = cfg["datasets"]
    for key in ("train_path", "validation_path"):
        p_str = str(ds_cfg.get(key, ""))
        if not p_str:
            raise ValueError(f"Dataset path '{key}' is missing in training_config.yaml.")
        if "test" in Path(p_str).stem.lower():
            raise PermissionError(f"Quarantine violation in config '{key}': '{p_str}'.")
    if "test_path" in ds_cfg:
        raise PermissionError("Quarantine violation: 'test_path' must never appear in training_config.yaml.")


def select_precision_policy(
    cuda_available: bool,
    device_name: str,
    vram_gb: float,
    compute_capability: Tuple[int, int],
    bf16_supported: bool,
    min_gpu_vram_gb: float = 8.0,
    min_compute_capability: float = 7.0,
    precision_mode: str = "auto",
) -> Dict[str, Any]:
    """
    Pure, deterministic hardware validation and precision selector:
    - Fails safely if no CUDA GPU is present or minimum VRAM / compute capability is unmet.
    - Selects FP16 (`fp16=True, bf16=False`) on NVIDIA T4 (`sm_75`).
    - Selects BF16 (`bf16=True, fp16=False`) on Ampere+ (`sm_80+`) when BF16 is natively supported.
    - Never combines BF16 compute with FP16 GradScaler.
    """
    if not cuda_available:
        raise RuntimeError(
            "No CUDA-capable GPU detected. Local CPU QLoRA training is disallowed; "
            "run this pipeline on Google Colab (NVIDIA T4) or another supported Cloud GPU."
        )

    cc_float = float(f"{compute_capability[0]}.{compute_capability[1]}")
    if cc_float < min_compute_capability:
        raise RuntimeError(
            f"Unsupported GPU '{device_name}' with compute capability {cc_float} "
            f"(minimum required: {min_compute_capability})."
        )

    if vram_gb < min_gpu_vram_gb:
        raise RuntimeError(
            f"Insufficient GPU VRAM on '{device_name}': {vram_gb:.2f} GB available "
            f"(minimum required: {min_gpu_vram_gb:.2f} GB)."
        )

    is_t4_gpu = ("T4" in device_name.upper()) or (compute_capability == (7, 5))

    mode = precision_mode.strip().lower()
    if mode == "auto":
        use_bf16 = bool((not is_t4_gpu) and compute_capability[0] >= 8 and bf16_supported)
        use_fp16 = not use_bf16
    elif mode == "fp16":
        use_fp16 = True
        use_bf16 = False
    elif mode == "bf16":
        if is_t4_gpu or compute_capability[0] < 8 or not bf16_supported:
            raise RuntimeError(
                f"Requested precision_mode='bf16', but GPU '{device_name}' (cc={cc_float}) "
                "requires FP16 and does not natively support hardware bfloat16."
            )
        use_bf16 = True
        use_fp16 = False
    else:
        raise ValueError(f"Unsupported precision_mode='{precision_mode}'. Expected 'auto', 'fp16', or 'bf16'.")

    if use_fp16 and use_bf16:
        raise RuntimeError("Internal safety violation: both fp16 and bf16 evaluated to True.")
    if is_t4_gpu and (not use_fp16 or use_bf16):
        raise RuntimeError("Safety violation: NVIDIA T4 requires FP16 (fp16=True, bf16=False).")

    dtype_name = "bfloat16" if use_bf16 else "float16"
    return {
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "compute_dtype_name": dtype_name,
    }


def verify_pretraining_compatibility_gate(
    python_version_info: Tuple[int, int],
    cuda_available: bool,
    cuda_version: Optional[str],
    device_name: str,
    compute_capability: Tuple[int, int],
    available_vram_gb: float,
    torch_version: str,
    transformers_version: str,
    peft_version: str,
    bitsandbytes_version: str,
    use_fp16: bool,
    use_bf16: bool,
    bnb_nf4_functional: bool,
    min_gpu_vram_gb: float = 8.0,
    min_compute_capability: float = 7.0,
) -> Dict[str, Any]:
    """
    Mandatory pre-training runtime compatibility gate (W-07).
    Stops execution BEFORE any Hugging Face model download if any runtime, package,
    precision, or bitsandbytes 4-bit NF4 requirement is unmet.
    """
    if not ((3, 10) <= python_version_info[:2] <= (3, 12)):
        raise RuntimeError(
            f"Unsupported Python version {python_version_info[0]}.{python_version_info[1]}; "
            "expected Python 3.10–3.12 for pinned cloud wheels."
        )

    if not cuda_available or not cuda_version:
        raise RuntimeError(
            "CUDA runtime is unavailable. Ensure a CUDA-enabled GPU runtime (e.g., Colab T4) is active."
        )

    cc_float = float(f"{compute_capability[0]}.{compute_capability[1]}")
    if cc_float < min_compute_capability:
        raise RuntimeError(
            f"GPU '{device_name}' compute capability {cc_float} is below minimum {min_compute_capability}."
        )

    if available_vram_gb < min_gpu_vram_gb:
        raise RuntimeError(
            f"Insufficient available GPU VRAM on '{device_name}': {available_vram_gb:.2f} GB free "
            f"(minimum required: {min_gpu_vram_gb:.2f} GB)."
        )

    def _base_ver(v_str: str) -> Version:
        return Version(v_str.split("+")[0])

    torch_v = _base_ver(torch_version)
    if not (Version("2.4.0") <= torch_v < Version("2.8.0")):
        raise RuntimeError(
            f"Incompatible PyTorch version '{torch_version}'; expected >=2.4.0,<2.8.0 with CUDA support."
        )
    if _base_ver(transformers_version) != Version("4.48.3"):
        raise RuntimeError(
            f"Expected transformers==4.48.3, detected '{transformers_version}'. Re-run dependency install."
        )
    if _base_ver(peft_version) != Version("0.14.0"):
        raise RuntimeError(
            f"Expected peft==0.14.0, detected '{peft_version}'. Re-run dependency install."
        )
    if _base_ver(bitsandbytes_version) != Version("0.45.3"):
        raise RuntimeError(
            f"Expected bitsandbytes==0.45.3, detected '{bitsandbytes_version}'. Re-run dependency install."
        )

    if use_fp16 == use_bf16:
        raise RuntimeError(
            f"Invalid precision flags: fp16={use_fp16}, bf16={use_bf16}. Exactly one must be True."
        )

    is_t4_gpu = ("T4" in device_name.upper()) or (compute_capability == (7, 5))
    if is_t4_gpu:
        if use_bf16 or not use_fp16:
            raise RuntimeError(
                f"NVIDIA T4 ('{device_name}', sm_75) requires FP16 (fp16=True, bf16=False); "
                "BF16 training configuration is rejected."
            )

    if not bnb_nf4_functional:
        raise RuntimeError(
            "bitsandbytes failed CUDA initialization or 4-bit NF4 verification. "
            "Execution stopped safely before downloading model weights."
        )

    return {
        "gate_passed": True,
        "python_version": f"{python_version_info[0]}.{python_version_info[1]}",
        "cuda_version": cuda_version,
        "gpu_model": device_name,
        "compute_capability": cc_float,
        "available_vram_gb": round(available_vram_gb, 2),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "bitsandbytes_version": bitsandbytes_version,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "bnb_nf4_verified": True,
    }


def probe_bitsandbytes_nf4_on_cuda(compute_dtype: Any) -> bool:
    """
    Execute an in-memory 4-bit NF4 double-quantization forward probe on GPU 0
    BEFORE downloading any Hugging Face checkpoint. Fails safely if bitsandbytes cannot initialize.
    """
    import torch
    import bitsandbytes as bnb

    probe_layer = None
    sample_in = None
    try:
        probe_layer = bnb.nn.Linear4bit(
            64,
            64,
            bias=False,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda:0")
        q_state = getattr(probe_layer.weight, "quant_state", None)
        if q_state is None or getattr(q_state, "quant_type", None) != "nf4":
            return False
        sample_in = torch.ones((2, 64), device="cuda:0", dtype=compute_dtype)
        with torch.no_grad():
            probe_out = probe_layer(sample_in)
        return bool(torch.isfinite(probe_out).all().item())
    except Exception as exc:
        logger.error("bitsandbytes 4-bit NF4 CUDA initialization failed: %s", exc)
        return False
    finally:
        del probe_layer, sample_in
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dry_run_validation(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Validate configuration, dataset quarantine, split disjointness, and simulated T4/Ampere
    precision policies locally without importing GPU libraries or starting training.
    """
    cfg = load_training_config(config_path)
    ds_cfg = cfg["datasets"]

    train_records, train_sha = load_approved_split(ds_cfg["train_path"], expected_split="train")
    val_records, val_sha = load_approved_split(ds_cfg["validation_path"], expected_split="validation")

    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Cross-split ID leakage detected between train and validation: {sorted(overlap)}")

    env_cfg = cfg["execution_environment"]
    min_vram = float(env_cfg.get("min_gpu_vram_gb", 8.0))
    min_cc = float(env_cfg.get("min_compute_capability", 7.0))
    mode = str(cfg["base_model"].get("precision_mode", "auto"))

    t4_policy = select_precision_policy(
        cuda_available=True,
        device_name="Tesla T4 (Simulated Check)",
        vram_gb=15.0,
        compute_capability=(7, 5),
        bf16_supported=False,
        min_gpu_vram_gb=min_vram,
        min_compute_capability=min_cc,
        precision_mode=mode,
    )
    ampere_policy = select_precision_policy(
        cuda_available=True,
        device_name="NVIDIA L4 (Simulated Check)",
        vram_gb=22.5,
        compute_capability=(8, 9),
        bf16_supported=True,
        min_gpu_vram_gb=min_vram,
        min_compute_capability=min_cc,
        precision_mode=mode,
    )

    hp = cfg["training_hyperparameters"]
    eff_batch = int(hp["per_device_train_batch_size"]) * int(hp["gradient_accumulation_steps"])

    summary = {
        "status": "DRY_RUN_VALIDATED_NO_TRAINING_EXECUTED",
        "base_model": cfg["base_model"]["repo_id"],
        "train_records": len(train_records),
        "train_sha256": train_sha,
        "validation_records": len(val_records),
        "validation_sha256": val_sha,
        "effective_batch_size": eff_batch,
        "lora": cfg["lora"],
        "simulated_t4_precision": t4_policy,
        "simulated_l4_precision": ampere_policy,
    }
    logger.info("Dry-run validation succeeded (zero training executed).")
    return summary


def run_qlora_training(
    config_path: Optional[str | Path] = None,
    confirm_train: bool = False,
) -> Dict[str, Any]:
    """
    Execute QLoRA fine-tuning on a verified CUDA GPU.
    Refuses to start unless `confirm_train=True` is explicitly provided.
    """
    if not confirm_train:
        raise PermissionError(
            "Training execution blocked: 'confirm_train=True' (or CLI flag '--confirm-train') "
            "is required before launching GPU training."
        )

    # Pin to GPU 0 on multi-GPU hosts (e.g., dual-T4 notebooks) before CUDA initialization
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cfg = load_training_config(config_path)

    # Lazy import of cloud GPU stack so local CPU syntax/dry-run validation never requires PyTorch
    import numpy as np
    import peft
    import torch
    import transformers
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    import bitsandbytes as bnb

    env_cfg = cfg["execution_environment"]
    base_cfg = cfg["base_model"]
    lora_cfg = cfg["lora"]
    ds_cfg = cfg["datasets"]
    hp = cfg["training_hyperparameters"]

    # 1. Dynamic GPU validation & precision selection
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    props = torch.cuda.get_device_properties(0) if cuda_ok else None
    vram_gb = (props.total_memory / (1024**3)) if props else 0.0
    cc = (props.major, props.minor) if props else (0, 0)
    bf16_hw = bool(cuda_ok and torch.cuda.is_bf16_supported())

    precision = select_precision_policy(
        cuda_available=cuda_ok,
        device_name=dev_name,
        vram_gb=vram_gb,
        compute_capability=cc,
        bf16_supported=bf16_hw,
        min_gpu_vram_gb=float(env_cfg.get("min_gpu_vram_gb", 8.0)),
        min_compute_capability=float(env_cfg.get("min_compute_capability", 7.0)),
        precision_mode=str(base_cfg.get("precision_mode", "auto")),
    )
    compute_dtype = torch.bfloat16 if precision["bf16"] else torch.float16
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)
    bnb_nf4_ok = probe_bitsandbytes_nf4_on_cuda(compute_dtype)

    verify_pretraining_compatibility_gate(
        python_version_info=(sys.version_info.major, sys.version_info.minor),
        cuda_available=cuda_ok,
        cuda_version=torch.version.cuda,
        device_name=dev_name,
        compute_capability=cc,
        available_vram_gb=free_vram_gb,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        peft_version=peft.__version__,
        bitsandbytes_version=bnb.__version__,
        use_fp16=bool(precision["fp16"]),
        use_bf16=bool(precision["bf16"]),
        bnb_nf4_functional=bnb_nf4_ok,
        min_gpu_vram_gb=float(env_cfg.get("min_gpu_vram_gb", 8.0)),
        min_compute_capability=float(env_cfg.get("min_compute_capability", 7.0)),
    )
    logger.info(
        "Pre-training compatibility gate PASSED on %s (%.2f GB free VRAM, cc=%s, NF4 verified) -> %s",
        dev_name,
        free_vram_gb,
        precision["compute_capability"],
        precision["compute_dtype_name"],
    )

    # 2. Reproducible random seeds
    seed = int(hp["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # 3. Load approved datasets ONLY (test split strictly quarantined)
    train_records, train_sha = load_approved_split(ds_cfg["train_path"], expected_split="train")
    val_records, val_sha = load_approved_split(ds_cfg["validation_path"], expected_split="validation")
    if not {r["id"] for r in train_records}.isdisjoint({r["id"] for r in val_records}):
        raise ValueError("Train and validation datasets have overlapping record IDs.")

    # 4. Load tokenizer & verify ChatML special tokens (`<|endoftext|>` pad != `<|im_end|>` eos)
    repo_id = str(base_cfg["repo_id"])
    hf_token = os.environ.get("HF_TOKEN") or None

    model_revision = "unknown"
    try:
        info = HfApi(token=hf_token).model_info(repo_id)
        model_revision = info.sha or "unknown"
    except Exception as exc:
        logger.warning("Could not query remote commit SHA via HfApi (%s); will check loaded config.", exc)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
    tok_meta = configure_qwen_tokenizer(tokenizer)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or model_revision

    max_seq_len = int(hp["max_seq_length"])
    train_features = prepare_tokenized_dataset(train_records, tokenizer, max_seq_length=max_seq_len)
    val_features = prepare_tokenized_dataset(val_records, tokenizer, max_seq_length=max_seq_len)

    class _TokenizedListDataset(Dataset):
        def __init__(self, items: List[Dict[str, Any]]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = self._items[idx]
            return {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }

    train_dataset = _TokenizedListDataset(train_features)
    val_dataset = _TokenizedListDataset(val_features)
    collator = QLoRACompletionCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        return_tensors="pt",
    )

    # 5. Configure 4-bit NF4 double quantization & PEFT LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(base_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(base_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(base_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        token=hf_token,
    )
    if model_revision == "unknown":
        model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

    use_grad_ckpt = bool(hp.get("gradient_checkpointing", True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)

    # Verify trainable LoRA parameters remain FP32 when FP16 GradScaler is active
    if precision["fp16"]:
        bad_dtypes = {
            p.dtype for _, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32
        }
        if bad_dtypes:
            raise RuntimeError(
                f"Detected non-FP32 trainable parameters {bad_dtypes} while fp16 GradScaler is enabled."
            )

    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable LoRA parameters: %s / %s (%.4f%%)",
        f"{trainable_params:,}",
        f"{all_params:,}",
        (trainable_params / all_params) * 100.0,
    )

    # 6. Configure Hugging Face Trainer with exact approved hyperparameters
    output_dir = PROJECT_ROOT / str(hp["output_dir"])
    adapter_dir = PROJECT_ROOT / str(hp["adapter_output_dir"])
    log_path = PROJECT_ROOT / str(hp["training_log_path"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(hp["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(hp["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
        learning_rate=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
        num_train_epochs=float(hp["num_train_epochs"]),
        lr_scheduler_type=str(hp["lr_scheduler_type"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        logging_steps=int(hp["logging_steps"]),
        save_strategy=str(hp["save_strategy"]),
        save_steps=int(hp["save_steps"]),
        save_total_limit=int(hp.get("save_total_limit", 3)),
        eval_strategy=str(hp["eval_strategy"]),
        eval_steps=int(hp["eval_steps"]),
        load_best_model_at_end=bool(hp.get("load_best_model_at_end", True)),
        metric_for_best_model=str(hp.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(hp.get("greater_is_better", False)),
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=str(hp["optim"]),
        fp16=bool(precision["fp16"]),
        bf16=bool(precision["bf16"]),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        report_to="none",
    )

    patience = int(hp.get("early_stopping_patience", 2))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # 7. Execute training with explicit CUDA OOM handling
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError as oom_err:
        alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
        res_mb = torch.cuda.memory_reserved(0) / (1024**2)
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA Out-Of-Memory during QLoRA training on '{dev_name}' "
            f"(allocated={alloc_mb:.1f} MB, reserved={res_mb:.1f} MB). "
            "Restart the Colab runtime to clear fragmented VRAM before retrying. "
            "Hyperparameters were NOT silently modified."
        ) from oom_err

    wallclock_seconds = round(time.time() - t0, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)

    # 8. Save best PEFT adapter & compute SHA-256 provenance
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    adapter_weights_file = adapter_dir / "adapter_model.safetensors"
    adapter_sha256 = (
        compute_file_sha256(adapter_weights_file) if adapter_weights_file.exists() else "missing"
    )

    # Extract structured loss curves
    train_loss_curve = [
        {"step": e["step"], "epoch": e.get("epoch"), "loss": e["loss"], "learning_rate": e.get("learning_rate")}
        for e in trainer.state.log_history
        if "loss" in e
    ]
    val_loss_curve = [
        {"step": e["step"], "epoch": e.get("epoch"), "eval_loss": e["eval_loss"]}
        for e in trainer.state.log_history
        if "eval_loss" in e
    ]

    training_log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": {
            "repo_id": repo_id,
            "revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "special_tokens": {
                "pad_token": tok_meta["pad_token"],
                "pad_token_id": tok_meta["pad_token_id"],
                "eos_token": tok_meta["eos_token"],
                "eos_token_id": tok_meta["eos_token_id"],
            },
        },
        "environment": {
            "python_version": sys.version,
            "cuda_version": torch.version.cuda,
            "gpu_name": dev_name,
            "gpu_vram_gb": precision["vram_gb"],
            "compute_capability": precision["compute_capability"],
            "precision_selected": precision["compute_dtype_name"],
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "bitsandbytes_version": bnb.__version__,
        },
        "datasets": {
            "train_path": str(ds_cfg["train_path"]),
            "train_record_count": len(train_records),
            "train_sha256": train_sha,
            "validation_path": str(ds_cfg["validation_path"]),
            "validation_record_count": len(val_records),
            "validation_sha256": val_sha,
        },
        "training_configuration": cfg,
        "random_seed": seed,
        "parameter_counts": {
            "trainable_parameters": trainable_params,
            "all_parameters": all_params,
            "trainable_percent": round((trainable_params / all_params) * 100.0, 4),
        },
        "results": {
            "global_steps_completed": trainer.state.global_step,
            "final_train_loss": train_result.training_loss,
            "best_eval_loss": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "wallclock_seconds": wallclock_seconds,
            "peak_gpu_vram_mb": peak_vram_mb,
            "train_loss_history": train_loss_curve,
            "validation_loss_history": val_loss_curve,
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir),
            "adapter_model_safetensors_sha256": adapter_sha256,
            "merged_model_sha256": None,
            "final_gguf_sha256": None,
        },
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    logger.info("Saved PEFT adapter to %s (sha256=%s...)", adapter_dir, adapter_sha256[:12])
    logger.info("Saved training provenance log to %s", log_path)
    return training_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 QLoRA Fine-Tuning Pipeline for Qwen2.5-1.5B-Instruct."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Path to training_config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset quarantine, and precision rules without starting training.",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Explicitly confirm and start Cloud GPU QLoRA training.",
    )
    args = parser.parse_args()

    if args.dry_run or not args.confirm_train:
        summary = dry_run_validation(args.config)
        print(json.dumps(summary, indent=2))
        if not args.confirm_train and not args.dry_run:
            logger.info(
                "Note: Executed safe dry-run only. Pass '--confirm-train' on a Cloud GPU to start training."
            )
        return

    run_qlora_training(config_path=args.config, confirm_train=True)


if __name__ == "__main__":
    main()
