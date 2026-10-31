"""The majority baseline predicts the most frequent training label for every validation row."""

import mlflow
import pandas as pd
import pytest

from geneva_permits.baseline import majority_label, run_majority

TEXTS = pd.DataFrame(
    {
        "partition": ["train_original"] * 3 + ["validation"] * 4,
        "label": ["REN", "REN", "VIL", "REN", "VIL", "VIL", "AFF"],
    }
)


def test_majority_label():
    assert majority_label(TEXTS["label"]) == "REN"


def test_run_logs_the_validation_scores(mlruns):
    summary = run_majority(TEXTS)
    assert summary["accuracy"] == 0.25
    assert summary["macro_f1"] == pytest.approx(0.4 / 8)  # REN: precision 1/4, recall 1, F1 0.4
    run = mlflow.search_runs(experiment_names=["selection"]).iloc[0]
    assert run["params.model"] == "majority"
    assert run["params.label"] == "REN"
    assert run["metrics.f1_VIL"] == 0
    assert any(mlruns.rglob("confusion.json"))
