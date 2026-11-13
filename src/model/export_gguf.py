"""
Post-Training LoRA Merge, GGUF Conversion & Q4_K_M Quantization Pipeline.

Workflow:
1. Load unquantized Hugging Face base checkpoint (`Qwen/Qwen2.5-1.5B-Instruct`) in clean FP16/BF16.
2. Verify zero 4-bit/8-bit `bitsandbytes` layers are present (never merge onto 4-bit weights).
3. Attach trained PEFT LoRA adapter and run `merge_and_unload()`.
4. Save merged Hugging Face `safetensors` checkpoint separately and compute SHA-256.
5. Convert merged checkpoint to intermediate `f16` GGUF via `llama.cpp/convert_hf_to_gguf.py`.
6. Quantize `f16` GGUF to `Q4_K_M` (`qwen2.5-1.5b-customer-support-q4_k_m.gguf`) via `llama-quantize`.
7. Verify GGUF binary header/version, calculate SHA-256, and record full artifact provenance.

Supports safe `--dry-run` validation locally without downloading checkpoints or `llama.cpp`.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlora_collator import compute_file_sha256
from src.model.train import parse_unique_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTENDED_FINAL_GGUF_FILENAME = "qwen2.5-1.5b-customer-support-q4_k_m.gguf"
PROTECTED_BASELINE_GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
PINNED_LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
# Exact upstream llama.cpp commit (tag b4094) bundled in local llama-cpp-python==0.3.2
PINNED_LLAMA_CPP_COMMIT = "74d73dc85cc2057446bf63cc37ff649ae7cebd80"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_export_settings(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and duplicate-check training and export settings from YAML."""
    target = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "training_config.yaml"
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found: {target}")

    return parse_unique_yaml(target.read_text(encoding="utf-8"), source_name=target.name)


def assert_baseline_model_protected(
    gguf_output_path: Path,
    baseline_gguf_path: Path,
) -> None:
    """
    Guarantee the export pipeline can never overwrite the baseline GGUF model.
    """
    out_resolved = gguf_output_path.resolve()
    base_resolved = baseline_gguf_path.resolve()

    if out_resolved == base_resolved or out_resolved.name.lower() == PROTECTED_BASELINE_GGUF_FILENAME:
        raise PermissionError(
            f"Safety Violation: Export target '{out_resolved}' collides with protected "
            f"baseline model '{base_resolved}'."
        )


def assert_export_target_not_overwritten(
    gguf_output_path: Path,
    baseline_gguf_path: Path,
    allow_overwrite_finetuned: bool = False,
) -> None:
    """
    Enforce single-run export safety (W-05):
    1. Unconditionally block overwriting the baseline GGUF model (even if overwrite is True).
    2. Never silently overwrite an existing fine-tuned GGUF model unless explicitly confirmed.
    """
    assert_baseline_model_protected(gguf_output_path, baseline_gguf_path)

    if gguf_output_path.exists() and not allow_overwrite_finetuned:
        raise FileExistsError(
            f"Existing fine-tuned GGUF detected at '{gguf_output_path}'. "
            "Silent overwrite is disabled; pass '--overwrite-existing-gguf' "
            "(or set ALLOW_OVERWRITE_FINETUNED_GGUF = True) for an intentional replacement."
        )


def assert_valid_immutable_commit_ref(commit_sha: str) -> str:
    """Verify that a configured llama.cpp Git reference is an immutable 40-char SHA-1 commit hash."""
    cleaned = commit_sha.strip().lower()
    if not _COMMIT_SHA_RE.match(cleaned):
        raise ValueError(
            f"Unpinned or invalid llama.cpp commit reference '{commit_sha}'. "
            "Expected a verified 40-character hexadecimal Git commit SHA."
        )
    return cleaned


def verify_llama_cpp_checkout_commit(
    llama_root: Path,
    expected_commit: str = PINNED_LLAMA_CPP_COMMIT,
) -> str:
    """Verify a local llama.cpp checkout matches the pinned immutable commit hash (W-06)."""
    expected = assert_valid_immutable_commit_ref(expected_commit)
    head_sha = subprocess.check_output(
        ["git", "-C", str(llama_root), "rev-parse", "HEAD"],
        text=True,
    ).strip().lower()
    if head_sha != expected:
        raise RuntimeError(
            f"llama.cpp commit mismatch in '{llama_root}': expected pinned commit '{expected}', "
            f"found '{head_sha}'."
        )
    return head_sha


def assert_unquantized_base_model(model: Any) -> None:
    """
    Verify that a Hugging Face model loaded for LoRA merging is NOT quantized in 4-bit or 8-bit.
    Merging LoRA weights onto 4-bit NF4 `bitsandbytes` tensors corrupts weights and breaks GGUF export.
    """
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError(
            "Safety Violation: Cannot merge LoRA adapter onto a 4-bit or 8-bit bitsandbytes model. "
            "Reload the base checkpoint in clean FP16 or BF16 before merging."
        )

    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
        raise RuntimeError(
            "Safety Violation: Base model config contains an active quantization_config. "
            "Load clean unquantized weights for adapter merging."
        )

    for mod_name, mod in model.named_modules():
        cls_name = mod.__class__.__name__
        if "Linear4bit" in cls_name or "Linear8bit" in cls_name or "Params4bit" in cls_name:
            raise RuntimeError(
                f"Safety Violation: Detected quantized module '{mod_name}' ({cls_name}) during merge."
            )


def compute_safetensors_manifest_sha256(model_dir: Path) -> str:
    """
    Compute a deterministic aggregate SHA-256 across all `.safetensors` files in a directory.
    """
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in '{model_dir}'.")

    if len(shards) == 1:
        return compute_file_sha256(shards[0])

    agg = hashlib.sha256()
    for shard in shards:
        shard_hash = compute_file_sha256(shard)
        agg.update(f"{shard.name}:{shard_hash}\n".encode("utf-8"))
    return agg.hexdigest()


def verify_gguf_binary(gguf_path: Path, min_size_bytes: int = 1024) -> Dict[str, Any]:
    """
    Inspect the binary header of a GGUF file (`GGUF` magic, version 2/3, tensor count, KV count).
    """
    if not gguf_path.exists() or not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    file_size = gguf_path.stat().st_size
    if file_size < min_size_bytes:
        raise ValueError(
            f"GGUF file '{gguf_path.name}' is unexpectedly small ({file_size} bytes < {min_size_bytes})."
        )

    with open(gguf_path, "rb") as f:
        header = f.read(24)

    if len(header) < 24:
        raise ValueError(f"File '{gguf_path.name}' is shorter than the 24-byte GGUF v2/v3 header.")

    magic, version, tensor_count, metadata_kv_count = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF":
        raise ValueError(f"Invalid GGUF magic header in '{gguf_path.name}': {magic!r}")
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF format version {version} in '{gguf_path.name}'.")
    if tensor_count == 0 or metadata_kv_count == 0:
        raise ValueError(
            f"Corrupt GGUF header in '{gguf_path.name}': tensors={tensor_count}, kv={metadata_kv_count}."
        )

    return {
        "filename": gguf_path.name,
        "file_size_bytes": file_size,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
    }


def validate_export_plan(
    config_path: Optional[str | Path] = None,
    llama_cpp_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Dry-run validation of all export paths, baseline protection rules, target filenames,
    and disk availability without performing conversion or downloading checkpoints.
    """
    cfg = load_export_settings(config_path)
    base_cfg = cfg.get("base_model", {})
    hp_cfg = cfg.get("training_hyperparameters", {})
    exp_cfg = cfg.get("export", {})

    adapter_dir = PROJECT_ROOT / str(hp_cfg.get("adapter_output_dir", "./models/customer-support-lora"))
    merged_dir = PROJECT_ROOT / str(
        exp_cfg.get("merged_model_dir", "./models/qwen2.5-1.5b-customer-support-merged-hf")
    )
    gguf_out = PROJECT_ROOT / str(
        exp_cfg.get("gguf_output_path", f"./models/{INTENDED_FINAL_GGUF_FILENAME}")
    )
    baseline_gguf = PROJECT_ROOT / str(
        exp_cfg.get("baseline_gguf_path", f"./models/{PROTECTED_BASELINE_GGUF_FILENAME}")
    )
    quant_type = str(exp_cfg.get("quantization_type", "Q4_K_M"))
    allow_overwrite = bool(exp_cfg.get("allow_overwrite_existing_gguf", False))
    pinned_commit = assert_valid_immutable_commit_ref(
        str(exp_cfg.get("llama_cpp_commit", PINNED_LLAMA_CPP_COMMIT))
    )

    assert_baseline_model_protected(gguf_out, baseline_gguf)

    if gguf_out.name != INTENDED_FINAL_GGUF_FILENAME:
        raise ValueError(
            f"Intended final GGUF filename must be '{INTENDED_FINAL_GGUF_FILENAME}', got '{gguf_out.name}'."
        )
    if quant_type != "Q4_K_M":
        raise ValueError(f"Expected quantization_type='Q4_K_M', got '{quant_type}'.")

    disk = shutil.disk_usage(PROJECT_ROOT.drive or PROJECT_ROOT)
    free_gb = round(disk.free / (1024**3), 2)

    llama_dir_resolved = Path(llama_cpp_dir).resolve() if llama_cpp_dir else None
    convert_script_exists = (
        (llama_dir_resolved / "convert_hf_to_gguf.py").exists() if llama_dir_resolved else False
    )

    plan = {
        "status": "DRY_RUN_EXPORT_PLAN_VALIDATED",
        "base_checkpoint": base_cfg.get("repo_id"),
        "adapter_input_dir": str(adapter_dir),
        "adapter_currently_present": (adapter_dir / "adapter_model.safetensors").exists(),
        "merged_hf_output_dir": str(merged_dir),
        "final_gguf_output_path": str(gguf_out),
        "final_gguf_filename": gguf_out.name,
        "final_gguf_already_exists": gguf_out.exists(),
        "allow_overwrite_existing_gguf": allow_overwrite,
        "protected_baseline_gguf_path": str(baseline_gguf),
        "baseline_gguf_intact": baseline_gguf.exists(),
        "quantization_type": quant_type,
        "pinned_llama_cpp_repo": str(exp_cfg.get("llama_cpp_repo_url", PINNED_LLAMA_CPP_REPO_URL)),
        "pinned_llama_cpp_commit": pinned_commit,
        "merge_guardrail": "Clean FP16/BF16 base reload enforced (4-bit bitsandbytes merge prohibited)",
        "available_disk_gb": free_gb,
        "llama_cpp_dir": str(llama_dir_resolved) if llama_dir_resolved else "Not provided (dry-run)",
        "convert_hf_to_gguf_detected": convert_script_exists,
    }
    logger.info("Export dry-run validation succeeded (no conversion executed).")
    return plan


def run_export_pipeline(
    config_path: Optional[str | Path] = None,
    llama_cpp_dir: Optional[str | Path] = None,
    quantize_bin: Optional[str | Path] = None,
    merge_dtype_str: str = "float16",
    cleanup_f16_gguf: bool = True,
    allow_overwrite_finetuned: bool = False,
    confirm_export: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete post-training merge -> GGUF -> Q4_K_M quantization pipeline.
    Requires `confirm_export=True` so dry-run calls never trigger downloads or builds.
    """
    if not confirm_export:
        raise PermissionError(
            "Export execution blocked: pass 'confirm_export=True' (or CLI '--confirm-export') "
            "after training has produced a verified PEFT adapter."
        )

    plan = validate_export_plan(config_path=config_path, llama_cpp_dir=llama_cpp_dir)
    cfg = load_export_settings(config_path)

    merged_dir = Path(plan["merged_hf_output_dir"])
    final_gguf = Path(plan["final_gguf_output_path"])
    baseline_gguf = Path(plan["protected_baseline_gguf_path"])
    effective_overwrite = bool(allow_overwrite_finetuned or plan["allow_overwrite_existing_gguf"])
    assert_export_target_not_overwritten(
        final_gguf,
        baseline_gguf,
        allow_overwrite_finetuned=effective_overwrite,
    )

    if not llama_cpp_dir:
        raise ValueError("`--llama-cpp-dir` pointing to a `llama.cpp` checkout is required for export.")

    llama_root = Path(llama_cpp_dir).resolve()
    verified_llama_commit = verify_llama_cpp_checkout_commit(
        llama_root,
        expected_commit=str(plan["pinned_llama_cpp_commit"]),
    )
    convert_script = llama_root / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"llama.cpp conversion script not found at: {convert_script}")

    if quantize_bin:
        q_bin = Path(quantize_bin).resolve()
    else:
        candidates = [
            llama_root / "build" / "bin" / "llama-quantize",
            llama_root / "build" / "bin" / "Release" / "llama-quantize.exe",
            llama_root / "llama-quantize",
        ]
        q_bin = next((c for c in candidates if c.exists()), candidates[0])

    if not q_bin.exists():
        raise FileNotFoundError(f"llama-quantize binary not found at: {q_bin}")

    adapter_dir = Path(plan["adapter_input_dir"])
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"Trained LoRA adapter not found at '{adapter_weights}'. Complete QLoRA training first."
        )

    # Lazy import of cloud PyTorch / HF / PEFT stack
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if merge_dtype_str == "bfloat16":
        merge_dtype = torch.bfloat16
    elif merge_dtype_str == "float16":
        merge_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported merge_dtype_str='{merge_dtype_str}'. Use 'float16' or 'bfloat16'.")

    repo_id = str(cfg["base_model"]["repo_id"])
    hf_token = os.environ.get("HF_TOKEN") or None

    # 1. Load clean unquantized FP16/BF16 base checkpoint (NEVER 4-bit bitsandbytes!)
    logger.info("Loading clean unquantized base checkpoint '%s' in %s...", repo_id, merge_dtype_str)
    base_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=merge_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    assert_unquantized_base_model(base_model)
    base_revision = getattr(base_model.config, "_commit_hash", None) or "unknown"

    # 2. Attach PEFT adapter and merge into clean base weights
    adapter_sha256 = compute_file_sha256(adapter_weights)
    logger.info("Attaching PEFT adapter from %s (sha256=%s...)", adapter_dir, adapter_sha256[:12])
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    assert_unquantized_base_model(merged_model)

    # 3. Save standalone merged Hugging Face checkpoint
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
    tokenizer.save_pretrained(str(merged_dir))
    merged_sha256 = compute_safetensors_manifest_sha256(merged_dir)
    logger.info("Saved merged HF model to %s (safetensors sha256=%s...)", merged_dir, merged_sha256[:12])

    # Free base/merged tensors before running external C++/Python GGUF tools
    del merged_model
    del peft_model
    del base_model

    # 4. Convert merged HF model -> intermediate FP16 GGUF
    final_gguf.parent.mkdir(parents=True, exist_ok=True)
    f16_gguf = final_gguf.with_name("qwen2.5-1.5b-customer-support-f16.gguf")
    assert_baseline_model_protected(f16_gguf, baseline_gguf)

    convert_cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(f16_gguf),
        "--outtype",
        "f16",
    ]
    logger.info("Running GGUF conversion: %s", " ".join(convert_cmd))
    subprocess.run(convert_cmd, check=True)
    verify_gguf_binary(f16_gguf, min_size_bytes=500 * 1024 * 1024)

    # 5. Quantize FP16 GGUF -> Q4_K_M using atomic staging so adapter/merged/prior GGUF survive errors
    quant_type = str(cfg["export"].get("quantization_type", "Q4_K_M"))
    staged_gguf = final_gguf.with_name(f"{final_gguf.stem}.part.gguf")
    assert_baseline_model_protected(staged_gguf, baseline_gguf)

    quant_cmd = [
        str(q_bin),
        str(f16_gguf),
        str(staged_gguf),
        quant_type,
    ]
    logger.info("Running GGUF quantization (%s): %s", quant_type, " ".join(quant_cmd))
    subprocess.run(quant_cmd, check=True)

    # 6. Verify resulting Q4_K_M GGUF header BEFORE promoting staged file & compute SHA-256
    gguf_header_info = verify_gguf_binary(staged_gguf, min_size_bytes=500 * 1024 * 1024)
    os.replace(staged_gguf, final_gguf)
    gguf_header_info["filename"] = final_gguf.name
    final_gguf_sha256 = compute_file_sha256(final_gguf)

    if cleanup_f16_gguf and f16_gguf.exists():
        f16_gguf.unlink()
        logger.info("Removed temporary unquantized intermediate file: %s", f16_gguf.name)

    # 7. Record complete export provenance metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_repo": repo_id,
        "base_model_revision": base_revision,
        "merge_precision": merge_dtype_str,
        "quantization": quant_type,
        "llama_cpp_commit": verified_llama_commit,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": adapter_sha256,
        "merged_model_dir": str(merged_dir),
        "merged_model_safetensors_sha256": merged_sha256,
        "final_gguf_path": str(final_gguf),
        "final_gguf_filename": final_gguf.name,
        "final_gguf_sha256": final_gguf_sha256,
        "gguf_header_verification": gguf_header_info,
    }

    meta_path = final_gguf.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Update `experiments/training_log.json` artifacts block if present
    log_path = PROJECT_ROOT / str(
        cfg["training_hyperparameters"].get("training_log_path", "./experiments/training_log.json")
    )
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            train_log = json.load(f)
        train_log.setdefault("artifacts", {})
        train_log["artifacts"]["merged_model_sha256"] = merged_sha256
        train_log["artifacts"]["final_gguf_sha256"] = final_gguf_sha256
        train_log["artifacts"]["final_gguf_filename"] = final_gguf.name
        train_log["artifacts"]["llama_cpp_commit"] = verified_llama_commit
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)

    logger.info(
        "Export complete: %s (%d bytes, sha256=%s..., llama.cpp=%s)",
        final_gguf.name,
        gguf_header_info["file_size_bytes"],
        final_gguf_sha256[:12],
        verified_llama_commit[:12],
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge QLoRA adapter cleanly and export Q4_K_M GGUF model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Path to training_config.yaml.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=None,
        help="Path to llama.cpp repository containing convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--quantize-bin",
        type=str,
        default=None,
        help="Optional explicit path to compiled llama-quantize binary.",
    )
    parser.add_argument(
        "--merge-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Clean unquantized floating-point precision for adapter merging.",
    )
    parser.add_argument(
        "--overwrite-existing-gguf",
        action="store_true",
        help="Explicitly allow replacing an existing fine-tuned GGUF (never overwrites baseline).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, baseline protection, and disk space without converting.",
    )
    parser.add_argument(
        "--confirm-export",
        action="store_true",
        help="Explicitly authorize running adapter merge and GGUF quantization.",
    )
    args = parser.parse_args()

    if args.dry_run or not args.confirm_export:
        plan = validate_export_plan(config_path=args.config, llama_cpp_dir=args.llama_cpp_dir)
        print(json.dumps(plan, indent=2))
        if not args.confirm_export and not args.dry_run:
            logger.info(
                "Note: Executed safe dry-run only. Pass '--confirm-export' after training to export GGUF."
            )
        return

    run_export_pipeline(
        config_path=args.config,
        llama_cpp_dir=args.llama_cpp_dir,
        quantize_bin=args.quantize_bin,
        merge_dtype_str=args.merge_dtype,
        allow_overwrite_finetuned=args.overwrite_existing_gguf,
        confirm_export=True,
    )


if __name__ == "__main__":
    main()
