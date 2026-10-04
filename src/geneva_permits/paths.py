"""Project root and data directories, as declared under `paths` in config.yaml."""

from geneva_permits.config import CONFIG_PATH, load_config

PROJECT_ROOT = CONFIG_PATH.parent
_PATHS = load_config()["paths"]

RAW_DIR = PROJECT_ROOT / _PATHS["raw"]
PREPARED_DIR = PROJECT_ROOT / _PATHS["prepared"]
SPLITS_DIR = PROJECT_ROOT / _PATHS["splits"]
RESULTS_DIR = PROJECT_ROOT / _PATHS["results"]
FIGURES_DIR = PROJECT_ROOT / _PATHS["figures"]
ANNOTATIONS_DIR = PROJECT_ROOT / _PATHS["annotations"]
