"""
Unit Tests for Model Loader and Configuration Validation.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.load_model import load_model_config, get_model, reset_model_instance


class TestModelLoading(unittest.TestCase):
    """Test suite for configuration loading, singleton management, and error handling."""

    def setUp(self):
        reset_model_instance()

    def test_load_valid_config(self):
        """Verify configuration loads with all expected keys."""
        config = load_model_config()
        self.assertIn("model", config)
        self.assertIn("runtime", config)
        self.assertIn("generation", config)
        self.assertEqual(config["model"]["name"], "Qwen2.5-1.5B-Instruct")
        self.assertEqual(config["runtime"]["device"], "cpu")
        self.assertEqual(config["runtime"]["n_gpu_layers"], 0)

    def test_missing_config_raises_file_not_found(self):
        """Verify FileNotFoundError on non-existent config path."""
        with self.assertRaises(FileNotFoundError):
            load_model_config("configs/non_existent_config.yaml")

    def test_missing_model_file_raises_error(self):
        """Verify FileNotFoundError when model path does not exist on disk."""
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            dummy_cfg = {
                "model": {"local_path": "models/non_existent_model_file.gguf", "name": "FakeModel"},
                "runtime": {"n_gpu_layers": 0, "n_threads": 4},
            }
            yaml.dump(dummy_cfg, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(FileNotFoundError):
                get_model(config_path=tmp_path, force_reload=True)
        finally:
            import os
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
