"""The single reader of config.yaml."""

from pathlib import Path

import yaml

# src/geneva_permits/config.py -> repository root, valid for the editable install uv sync makes.
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Parse config.yaml."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
