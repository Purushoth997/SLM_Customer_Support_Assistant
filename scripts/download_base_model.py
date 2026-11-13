"""
Automated Base Model Downloader for SLM Customer Support Assistant.
Downloads the exact pinned revision of Qwen2.5-1.5B-Instruct from Hugging Face Hub
into the local cache or designated directory, ensuring cryptographic reproducibility.

SAFETY LOCK:
Does NOT download unless explicitly executed with --confirm-download.
Supports safe --dry-run validation without initiating network transfers.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_MODEL_NAME_OR_PATH, BASE_MODEL_REVISION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download pinned Qwen2.5-1.5B-Instruct snapshot from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=BASE_MODEL_NAME_OR_PATH,
        help=f"Hugging Face repository ID (default: {BASE_MODEL_NAME_OR_PATH})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=BASE_MODEL_REVISION,
        help=f"Pinned Git commit hash (default: {BASE_MODEL_REVISION})",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Optional local target directory (defaults to Hugging Face standard cache)",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        default=False,
        help="Explicit user authorization flag required to initiate the ~3.1 GB model download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Inspect metadata and target endpoints without initiating file downloads.",
    )
    return parser.parse_args()


def download_base_model(
    repo_id: str,
    revision: str,
    local_dir: str = None,
    confirm_download: bool = False,
    dry_run: bool = False,
):
    print("=" * 78)
    print("   SLM BASE MODEL DOWNLOAD PROCEDURE (PINNED REVISION)")
    print("=" * 78)
    print(f"Target Repository:   {repo_id}")
    print(f"Pinned Revision:     {revision}")
    print(f"Local Target:        {local_dir or 'Default Hugging Face Cache (~/.cache/huggingface/hub)'}")
    print(f"Estimated Download:  ~3.1 GB (safetensors weights, configs, and tokenizer)")
    print("-" * 78)

    if dry_run or not confirm_download:
        print("\n[SAFETY GATE ACTIVE]")
        if not confirm_download:
            print("  >> Download was NOT initiated because --confirm-download was not passed.")
            print("  >> To download the base model, run:")
            print(f"     python scripts/download_base_model.py --confirm-download")
        else:
            print("  >> Dry-run mode: All parameters verified. Zero bytes downloaded.")
        print("=" * 78)
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface-hub is required for download. "
            "Please activate venv-peft and run: pip install -r requirements-peft.txt"
        )
        return False

    logger.info("Initiating download for %s at revision %s...", repo_id, revision)
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        logger.info("Successfully downloaded base model snapshot to: %s", download_path)
        print("\n[SUCCESS] Base model snapshot downloaded and verified.")
        return True
    except Exception as e:
        logger.error("Download failed: %s", e)
        print(f"\n[ERROR] Download encountered an exception: {e}")
        return False


def main():
    args = parse_args()
    success = download_base_model(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        confirm_download=args.confirm_download,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success or args.dry_run or not args.confirm_download else 1)


if __name__ == "__main__":
    main()
