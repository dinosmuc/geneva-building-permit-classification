"""config.yaml and the project directories it declares."""

from pathlib import Path

import yaml

# src/geneva_permits/config.py -> repository root, valid for the editable install uv sync makes.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    """Parse config.yaml."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


_PATHS = load_config()["paths"]
RAW_DIR = PROJECT_ROOT / _PATHS["raw"]
FIGURES_DIR = PROJECT_ROOT / _PATHS["figures"]
ANNOTATIONS_DIR = PROJECT_ROOT / _PATHS["annotations"]
