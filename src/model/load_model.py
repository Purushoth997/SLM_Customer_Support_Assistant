"""
Model Loader Module for SLM Customer Support Assistant.
Loads, validates, and initializes the local quantized GGUF model via llama-cpp-python
using pure CPU configuration, thread pinning, and cached singleton instantiation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global singleton instance cache
_MODEL_INSTANCE = None
_MODEL_CONFIG = None


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load model and runtime configuration from YAML."""
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None and config_path is None:
        return _MODEL_CONFIG

    target_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "model_config.yaml"
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _MODEL_CONFIG = config
    logger.info("Loaded model configuration from: %s", target_path)
    return config


def get_model(config_path: Optional[str] = None, force_reload: bool = False):
    """
    Load and return the Llama model instance as a singleton.
    Guarantees the model is loaded only once in memory across requests.
    """
    global _MODEL_INSTANCE

    if _MODEL_INSTANCE is not None and not force_reload:
        return _MODEL_INSTANCE

    config = load_model_config(config_path)
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})

    local_path_str = model_cfg.get("local_path", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    model_path = Path(local_path_str)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model file not found at: {model_path}. "
            "Please ensure the model is downloaded to the models/ directory before initialization."
        )

    # Runtime parameters
    n_gpu_layers = int(runtime_cfg.get("n_gpu_layers", 0))
    if n_gpu_layers != 0:
        logger.warning("n_gpu_layers was set to %d. Overriding to 0 for pure CPU execution.", n_gpu_layers)
        n_gpu_layers = 0

    n_threads = int(runtime_cfg.get("n_threads", 8))
    n_threads_batch = int(runtime_cfg.get("n_threads_batch", n_threads))
    n_ctx = int(runtime_cfg.get("n_ctx", 2048))
    n_batch = int(runtime_cfg.get("n_batch", 512))
    verbose = bool(runtime_cfg.get("verbose", False))
    use_mmap = bool(runtime_cfg.get("use_mmap", True))
    use_mlock = bool(runtime_cfg.get("use_mlock", False))

    logger.info("Initializing llama-cpp-python CPU runtime for: %s", model_path.name)
    logger.info("Parameters: threads=%d, ctx=%d, batch=%d, gpu_layers=%d, mmap=%s",
                n_threads, n_ctx, n_batch, n_gpu_layers, use_mmap)

    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is not installed in the active environment. "
            "Please install it using the project instructions."
        ) from e

    try:
        instance = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
        )
        _MODEL_INSTANCE = instance
        logger.info("Successfully loaded GGUF model: %s (Context: %d tokens)", model_cfg.get("name"), n_ctx)
        return _MODEL_INSTANCE
    except Exception as e:
        logger.error("Failed to initialize Llama model from %s: %s", model_path, e)
        raise RuntimeError(f"Error loading model weights into memory: {e}") from e


def reset_model_instance():
    """Reset the cached model instance (useful for testing and reloading)."""
    global _MODEL_INSTANCE
    _MODEL_INSTANCE = None
