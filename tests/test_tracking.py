"""The MLflow store stays under the project root; runs carry their provenance; reads give tables."""

import re

import mlflow
import pandas as pd
import pytest

from geneva_permits.evaluate import CLASSES
from geneva_permits.source import read_lock
from geneva_permits.tracking import (
    LEARNING_CURVES,
    SELECTION,
    histories,
    learning_curves,
    mean_table,
    overview,
    runs,
    set_experiment,
    start_run,
)

F1 = {f"f1_{c}": 0.0 for c in CLASSES}


def log(experiment, params, macro_f1):
    """One finished run with the given params and a macro-F1; its run id."""
    set_experiment(experiment)
    with start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({"macro_f1": macro_f1, "accuracy": macro_f1} | F1)
        return mlflow.active_run().info.run_id


def test_store_and_artifacts_land_under_the_root(tmp_path, monkeypatch):
    root, cwd = tmp_path / "mlruns", tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    set_experiment("probe", root=root)
    set_experiment("probe", root=root)  # reselecting an existing experiment
    with mlflow.start_run():
        mlflow.log_metric("macro_f1", 0.5)
        mlflow.log_text("x", "note.txt")

    assert (root / "mlflow.db").is_file()
    assert any((root / "probe").rglob("note.txt"))
    assert list(cwd.iterdir()) == []


def test_runs_are_tagged_with_the_commit_and_the_snapshot(mlruns):
    set_experiment("probe")
    with start_run(run_name="probe"):
        pass
    run = mlflow.search_runs(experiment_names=["probe"]).iloc[0]
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", run["tags.git_commit"])
    assert run["tags.snapshot_sha256"] == read_lock()["sha256"]
    assert run["tags.mlflow.runName"] == "probe"


def test_overview_takes_the_best_grid_point_and_the_mean_over_seeds(mlruns):
    log(SELECTION, {"model": "majority", "label": "REN"}, 0.07)
    svc = {"model": "linearsvc", "variant": "A", "weighting": "none", "ngram_range": "1-2"}
    log(SELECTION, svc | {"C_log2": -3}, 0.6)
    log(SELECTION, svc | {"C_log2": -1}, 0.5)
    bert = {"model": "camembert", "weighting": "none", "stage": "main"}
    log(SELECTION, bert | {"seed": 42}, 0.6)
    log(SELECTION, bert | {"seed": 13}, 0.7)
    log(SELECTION, bert | {"seed": 13, "stage": "pilot"}, 0.9)  # not a main run
    table = overview(runs(SELECTION))
    assert table["model"].tolist() == ["majority", "linearsvc", "camembert"]
    assert table["macro_f1"].tolist() == pytest.approx([0.07, 0.6, 0.65])
    assert table.loc[1, "C_log2"] == "-3"
    assert table.loc[2, "seed"] == "13, 42"
    assert "f1_AFF" in table.columns


def test_histories_and_mean_tables_line_runs_up(mlruns):
    set_experiment(SELECTION)
    ids = []
    for seed, curve in [(13, [0.1, 0.3]), (42, [0.2, 0.4])]:
        with start_run():
            mlflow.log_params({"weighting": "none", "seed": seed})
            for step, value in enumerate(curve, 1):
                mlflow.log_metric("val_macro_f1", value, step=step)
            per_class = pd.DataFrame({"class": ["AFF", "REN"], "f1": [seed / 100, 0.5]})
            mlflow.log_table(per_class, "per_class.json")
            ids.append(mlflow.active_run().info.run_id)
    wide = histories(runs(SELECTION), "val_macro_f1", ["weighting", "seed"])
    assert wide.index.tolist() == [1, 2]
    assert wide[("none", "13")].tolist() == [0.1, 0.3]
    mean = mean_table(ids, "per_class.json")
    assert mean.index.tolist() == ["AFF", "REN"]
    assert mean.loc["AFF", "f1"] == pytest.approx(0.275)


def test_learning_curves_join_both_models_with_the_full_camembert_set(mlruns):
    draw = {"fraction": 0.1, "seed": 13, "train_rows": 3}
    log(LEARNING_CURVES, {"model": "linearsvc", "C_log2": -3} | draw, 0.5)
    log(LEARNING_CURVES, {"model": "linearsvc", "C_log2": -1} | draw, 0.4)
    log(LEARNING_CURVES, {"model": "camembert", "weighting": "none"} | draw, 0.6)
    main = {"model": "camembert", "stage": "main", "seed": 13, "train_rows": 30}
    log(SELECTION, main | {"weighting": "none"}, 0.7)
    log(SELECTION, main | {"weighting": "full"}, 0.65)  # not the curves' weighting
    table = learning_curves()
    assert table.values.tolist() == [
        ["camembert", 0.1, 13, 3, 0.6],
        ["camembert", 1.0, 13, 30, 0.7],
        ["linearsvc", 0.1, 13, 3, 0.5],
    ]
