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
    TEST,
    bootstrap_table,
    camembert_seeds,
    histories,
    holdout_overview,
    holdout_predictions,
    holdout_runs,
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


def test_reading_runs_leaves_the_active_experiment_alone(mlruns):
    log(SELECTION, {"model": "majority"}, 0.1)
    probe = set_experiment("probe")
    assert runs(SELECTION)["params.model"].tolist() == ["majority"]
    assert runs("missing").empty
    with start_run():
        assert mlflow.active_run().info.experiment_id == probe.experiment_id


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
    set_experiment("probe")  # reading must not depend on the active experiment
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


def test_the_test_experiment_reads_back_per_run_and_summarized(mlruns):
    log(TEST, {"model": "majority", "label": "REN"}, 0.06)
    log(TEST, {"model": "linearsvc", "variant": "B", "weighting": "medium", "C_log2": -3}, 0.62)
    log(TEST, {"model": "camembert", "weighting": "none", "seed": 42}, 0.66)
    log(TEST, {"model": "camembert", "weighting": "none", "seed": 13}, 0.68)
    set_experiment(TEST)
    with start_run(run_name="bootstrap"):
        table = pd.DataFrame(
            {
                "model": ["camembert", "camembert - linearsvc B medium"],
                "metric": ["macro_f1", "macro_f1"],
                "comparison": ["", "primary"],
                "estimate": [0.67, 0.05],
                "low": [0.63, 0.03],
                "high": [0.71, 0.08],
            }
        )
        mlflow.log_table(table, "bootstrap.json")
        predictions = pd.DataFrame({"record_key": ["k1", "k2"], "label": ["REN", "PI"]})
        mlflow.log_table(predictions.assign(majority="REN"), "predictions.json")
    set_experiment("probe")  # reading must not depend on the active experiment
    per_run = holdout_runs()
    assert per_run["model"].tolist() == ["majority", "linearsvc", "camembert", "camembert"]
    assert per_run["seed"].tolist()[2:] == ["13", "42"]
    summary = holdout_overview()
    assert summary["model"].tolist() == ["majority", "linearsvc", "camembert"]
    assert summary.loc[2, "macro_f1"] == pytest.approx(0.67)
    assert summary.loc[2, "seed"] == "13, 42"
    intervals = bootstrap_table()
    assert intervals.loc[("camembert - linearsvc B medium", "macro_f1"), "comparison"] == "primary"
    assert intervals.loc[("camembert", "macro_f1"), "high"] == pytest.approx(0.71)
    assert holdout_predictions().loc["k2", "label"] == "PI"


def test_camembert_seeds_join_the_test_runs_with_their_stored_main_runs(mlruns):
    stored = {}
    for seed, epoch, macro_f1 in [(42, 3, 0.70), (13, 5, 0.71)]:
        set_experiment(SELECTION)
        with start_run():
            mlflow.log_params({"model": "camembert", "stage": "main", "seed": seed})
            mlflow.log_metrics({"kept_epoch": epoch, "macro_f1": macro_f1})
            stored[seed] = mlflow.active_run().info.run_id
    for seed, macro_f1 in [(42, 0.68), (13, 0.66)]:
        params = {"model": "camembert", "seed": seed, "stored_run": stored[seed]}
        log(TEST, params, macro_f1)
    table = camembert_seeds()
    assert table["seed"].tolist() == [13, 42]
    assert table["kept_epoch"].tolist() == [5, 3]
    assert table["validation_macro_f1"].tolist() == pytest.approx([0.71, 0.70])
    assert table["test_macro_f1"].tolist() == pytest.approx([0.66, 0.68])
