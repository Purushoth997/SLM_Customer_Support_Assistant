"""
Application Configuration Module for SLM Customer Support Assistant.
Provides centralized, environment-driven configuration for Hugging Face
Transformers and PEFT LoRA inference, API server parameters, and safety bounds.
"""

import os
from pathlib import Path
from typing import List, Optional

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model Provenance & Checkpoints
BASE_MODEL_NAME_OR_PATH = os.environ.get(
    "SLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
BASE_MODEL_REVISION = os.environ.get(
    "SLM_BASE_REVISION", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

# Adapter Configuration
ADAPTER_DIR = os.environ.get(
    "SLM_ADAPTER_DIR", str(PROJECT_ROOT / "exports" / "customer-support-lora")
)
EXPECTED_ADAPTER_SHA256 = os.environ.get(
    "SLM_EXPECTED_ADAPTER_SHA256",
    "5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6",
)
LORA_RANK = int(os.environ.get("SLM_LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("SLM_LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("SLM_LORA_DROPOUT", "0.10"))
TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Hardware & Device Runtime
DEVICE = os.environ.get("SLM_DEVICE", "cpu")
TORCH_DTYPE = os.environ.get("SLM_TORCH_DTYPE", "float32")

# Generation Defaults
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("SLM_MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.environ.get("SLM_TEMPERATURE", "0.3"))
DEFAULT_TOP_P = float(os.environ.get("SLM_TOP_P", "0.85"))
DEFAULT_TOP_K = int(os.environ.get("SLM_TOP_K", "40"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("SLM_REPETITION_PENALTY", "1.15"))
MAX_QUESTION_LENGTH = int(os.environ.get("SLM_MAX_QUESTION_LENGTH", "2000"))

# Server Network Configuration
SERVER_HOST = os.environ.get("SLM_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SLM_PORT", "8000"))
ALLOW_MOCK_FALLBACK = os.environ.get("SLM_ALLOW_MOCK_FALLBACK", "false").lower() in ("true", "1", "yes")

# UI Configuration
APP_TITLE = "Apex Retail Solutions — SLM Customer Support"
APP_VERSION = "1.0.0"
