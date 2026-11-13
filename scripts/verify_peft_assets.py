"""
Asset Verification Utility for SLM Customer Support PEFT Inference.
Verifies the cryptographic integrity of the local customer-support-lora adapter
and checks for the availability of the Qwen2.5-1.5B-Instruct base model snapshot.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_MODEL_NAME_OR_PATH,
    BASE_MODEL_REVISION,
    ADAPTER_DIR,
    EXPECTED_ADAPTER_SHA256,
)
from src.model.load_peft import compute_file_sha256, verify_adapter_integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 78)
    print("   SLM CUSTOMER SUPPORT - PEFT & BASE MODEL ASSET VERIFICATION AUDIT")
    print("=" * 78)
    print(f"Project Directory:   {PROJECT_ROOT}")
    print(f"Base Model Name:     {BASE_MODEL_NAME_OR_PATH}")
    print(f"Pinned Revision:     {BASE_MODEL_REVISION}")
    print(f"Target Adapter Dir:  {ADAPTER_DIR}")
    print(f"Expected SHA-256:    {EXPECTED_ADAPTER_SHA256}")
    print("-" * 78)

    # 1. Check Adapter
    print("\n[1/2] Verifying LoRA Adapter Integrity...")
    is_valid, msg, details = verify_adapter_integrity()

    if details["exists"]:
        print(f"  [FOUND] Adapter Directory: {details['adapter_dir']}")
        print(f"  - adapter_config.json:     {'PRESENT' if details['config_exists'] else 'MISSING'}")
        print(f"  - adapter_model.safetensors: {'PRESENT' if details['safetensors_exists'] else 'MISSING'}")
        if details["computed_sha256"]:
            print(f"  - Computed SHA-256:        {details['computed_sha256']}")
            if details["hash_matches"]:
                print("  >> RESULT: SHA-256 MATCHES EXPECTED VALUE (PASS)")
            else:
                print("  >> RESULT: SHA-256 MISMATCH! (FAIL)")
        else:
            print("  >> RESULT: INCOMPLETE ADAPTER FILES (FAIL)")
    else:
        print(f"  [MISSING] Adapter directory not found at: {details['adapter_dir']}")
        print("  >> ACTION REQUIRED: Copy 'customer-support-lora/' from Google Drive")
        print("     (/content/drive/MyDrive/slm-customer-support-phase6/exports/customer-support-lora)")
        print(f"     into: {details['adapter_dir']}")

    # 2. Check Base Model
    print("\n[2/2] Checking Base Model Snapshot Availability...")
    base_path = Path(BASE_MODEL_NAME_OR_PATH)
    if not base_path.is_absolute():
        potential_path = PROJECT_ROOT / base_path
    else:
        potential_path = base_path

    if potential_path.is_dir() and (potential_path / "config.json").is_file():
        print(f"  [FOUND] Local Base Model directory: {potential_path}")
        print("  >> RESULT: LOCAL SNAPSHOT AVAILABLE (PASS)")
    else:
        # Check Hugging Face hub cache
        hf_cache_hub = Path.home() / ".cache" / "huggingface" / "hub"
        found_in_cache = False
        if hf_cache_hub.is_dir():
            for d in hf_cache_hub.iterdir():
                if "qwen2.5-1.5b-instruct" in d.name.lower():
                    found_in_cache = True
                    print(f"  [FOUND] Cached Hugging Face snapshot: {d.name}")
                    break

        if found_in_cache:
            print("  >> RESULT: BASE MODEL FOUND IN HUGGING FACE CACHE (PASS)")
        else:
            print(f"  [NOT FOUND] Base model '{BASE_MODEL_NAME_OR_PATH}' is not cached locally.")
            print("  >> ACTION REQUIRED: Run the download script when authorized:")
            print("     python scripts/download_base_model.py")

    print("\n" + "=" * 78)
    overall_ready = is_valid and (potential_path.is_dir() or found_in_cache)
    if overall_ready:
        print("OVERALL ASSET STATUS: ALL ASSETS VERIFIED (READY)")
    else:
        print("OVERALL ASSET STATUS: ASSETS PENDING ACQUISITION (BLOCKED)")
    print("=" * 78)

    return 0 if overall_ready else 1


if __name__ == "__main__":
    sys.exit(main())
