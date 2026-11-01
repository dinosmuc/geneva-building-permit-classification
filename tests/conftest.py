"""Fixtures on the pinned snapshot, shared by the real-data tests."""

import os

import pytest

from geneva_permits import config
from geneva_permits.prepare import prepare_records
from geneva_permits.source import load_source, read_lock


@pytest.fixture(scope="session")
def raw():
    """The pinned snapshot; skipped locally when not fetched, a failure in CI."""
    if not (config.RAW_DIR / read_lock()["file"]).is_file():
        if os.environ.get("CI"):
            pytest.fail("pinned snapshot missing; CI must run `uv run permits data fetch` first")
        pytest.skip("pinned snapshot not fetched; run `uv run permits data fetch`")
    return load_source()


@pytest.fixture(scope="session")
def records(raw):
    return prepare_records(raw)


@pytest.fixture(autouse=True)
def mlruns(tmp_path, monkeypatch):
    """A throwaway MLflow store for every test, so none reads or writes the project's mlruns/."""
    monkeypatch.setattr(config, "MLRUNS_DIR", tmp_path / "mlruns")
    return tmp_path / "mlruns"


@pytest.fixture
def classical_settings():
    """A two-point C grid with the project's solver settings, for toy-data fits."""
    return {
        "token_pattern": r"(?u)\b\w+\b",
        "ngram_ranges": [[1, 1]],
        "C_log2": [0, 2],
        "dual": False,
        "max_iter": 100,
        "random_state": 13,
    }
