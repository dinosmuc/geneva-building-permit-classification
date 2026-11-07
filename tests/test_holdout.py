"""The one-shot test: refit LinearSVC points, stored CamemBERT models, one run each, bootstrap."""

import mlflow
import pandas as pd
import pytest
from test_camembert import SETTINGS as BERT_SETTINGS
from test_camembert import TRAIN_LABELS, WORDS, toy_model, toy_tokenizer

from geneva_permits import camembert, classical, holdout, tracking
from geneva_permits.config import load_config

TEXTS = pd.DataFrame(  # the training rows of test_camembert cover every class, REN twice
    {
        "record_key": [f"k{i}" for i in range(17)],
        "partition": ["train_original"] * 9 + ["validation"] * 4 + ["test_historical"] * 4,
        "label": TRAIN_LABELS + ["VIL", "REN", "NOU", "AFF"] + ["VIL", "REN", "PI", "REN"],
        "text": [f"{w} {w}" for w in WORDS]
        + ["villa", "bureaux", "immeuble", "usine"]
        + ["villa", "bureaux", "jardin", "bureaux"],
    }
)
TEXTS["lemmas"] = TEXTS["text"]
GROUPS = ["d1", "d1", "d2", "d3"]
PLAN = {
    "partition": "test_historical",
    "groups": "base_dossier",
    "models": {
        "majority": {},
        "linearsvc": [
            {"variant": "A", "weighting": "none"},
            {"variant": "B", "weighting": "medium"},
        ],
        "camembert": {"weighting": "none", "seeds": [13]},
    },
    "primary_comparison": ["camembert", "linearsvc B medium"],
    "secondary_comparisons": [["linearsvc B medium", "linearsvc A none"]],
}


@pytest.fixture
def store(mlruns, monkeypatch, classical_settings):
    """A store holding both LinearSVC grids and the toy main CamemBERT runs, from a clean tree."""
    monkeypatch.setattr(camembert, "load_tokenizer", lambda _settings: toy_tokenizer())
    monkeypatch.setattr(camembert, "load_model", toy_model)
    monkeypatch.setattr(tracking, "git_commit", lambda: "0" * 40)
    monkeypatch.setattr(holdout, "git_commit", lambda: "0" * 40)
    classical.run_grid("A", TEXTS, classical_settings)
    classical.run_grid("B", TEXTS, classical_settings)
    camembert.run_main(TEXTS, BERT_SETTINGS)
    evaluation = {"bootstrap": {"samples": 50, "level": 0.95, "seed": 13}, "test": PLAN}
    return load_config() | {
        "classical": classical_settings,
        "training": BERT_SETTINGS,
        "evaluation": evaluation,
    }


def test_every_listed_model_is_scored_once_and_bootstrapped(store, mlruns):
    table = holdout.run_test(TEXTS, GROUPS, store)
    assert table.index.get_level_values("model").unique().tolist() == [
        "majority",
        "linearsvc A none",
        "linearsvc B medium",
        "camembert",
        "camembert - linearsvc B medium",
        "linearsvc B medium - linearsvc A none",
    ]
    assert table["estimate"].between(-1, 1).all()
    assert table.loc[("camembert - linearsvc B medium", "macro_f1"), "comparison"] == "primary"
    assert (
        table.loc[("linearsvc B medium - linearsvc A none", "macro_f1"), "comparison"]
        == "secondary"
    )
    assert table.loc[("majority", "macro_f1"), "comparison"] == ""
    assert "f1_AFF" in table.index.get_level_values("metric")
    test_runs = tracking.runs(tracking.TEST).set_index("tags.mlflow.runName")
    assert sorted(test_runs.index) == [
        "bootstrap",
        "camembert-none-13",
        "linearsvc-A-none",
        "linearsvc-B-medium",
        "majority",
    ]
    assert test_runs.loc["majority", "metrics.accuracy"] == 0.5  # REN: two of the four test rows
    selected = tracking.runs(tracking.SELECTION)
    svc = selected[selected["params.variant"].eq("B") & selected["params.weighting"].eq("medium")]
    point = tracking.best(svc, ["variant", "weighting"]).iloc[0]  # ties keep the earlier run
    assert test_runs.loc["linearsvc-B-medium", "params.selected_run"] == point["run_id"]
    assert test_runs.loc["linearsvc-B-medium", "params.C_log2"] == point["params.C_log2"]
    stored = selected[selected["params.model"].eq("camembert") & selected["params.seed"].eq("13")]
    assert test_runs.loc["camembert-none-13", "params.stored_run"] in set(stored["run_id"])
    assert test_runs.loc["bootstrap", "params.distinct_groups"] == "3"
    predictions = mlflow.load_table(
        "predictions.json", run_ids=[test_runs.loc["bootstrap", "run_id"]]
    )
    assert predictions.columns.tolist() == [
        "record_key",
        "label",
        "majority",
        "linearsvc A none",
        "linearsvc B medium",
        "camembert 13",
    ]
    assert predictions["label"].tolist() == ["VIL", "REN", "PI", "REN"]
    stored_table = mlflow.load_table(
        "bootstrap.json", run_ids=[test_runs.loc["bootstrap", "run_id"]]
    )
    assert len(stored_table) == len(table)


def test_the_test_is_scored_once_and_only_from_a_clean_tree(store, monkeypatch):
    holdout.run_test(TEXTS, GROUPS, store)
    with pytest.raises(RuntimeError, match="already scored"):
        holdout.run_test(TEXTS, GROUPS, store)
    monkeypatch.setattr(holdout, "git_commit", lambda: "0" * 40 + "-dirty")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        holdout.run_test(TEXTS, GROUPS, store)


def test_an_interrupted_attempt_is_deleted_and_the_rerun_tagged(store):
    tracking.set_experiment(tracking.TEST)
    with tracking.start_run(run_name="majority"):
        mlflow.log_param("model", "majority")  # finished, but no bootstrap run followed
    mlflow.start_run(run_name="linearsvc-A-none")
    mlflow.end_run("FAILED")
    holdout.run_test(TEXTS, GROUPS, store)
    test_runs = tracking.runs(tracking.TEST)
    assert len(test_runs) == 5
    assert test_runs["tags.rerun"].eq("after an interrupted attempt; 2 partial runs deleted").all()


def test_dossiers_come_from_the_split_records(monkeypatch):
    split = pd.DataFrame({"record_key": ["a", "b"], "base_dossier": ["DD 1", "DD 2"]})
    monkeypatch.setattr(holdout, "load_splits", lambda: split)
    texts = pd.DataFrame({"record_key": ["b", "a", "b"]})
    assert holdout.dossiers(texts).tolist() == ["DD 2", "DD 1", "DD 2"]
