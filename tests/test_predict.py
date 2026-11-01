"""New descriptions get a class and eight scores from the stored model of the configured seed."""

import pandas as pd
import pytest
from test_camembert import SETTINGS, TEXTS, toy_model, toy_tokenizer

from geneva_permits import camembert, predict, tracking
from geneva_permits.config import load_config
from geneva_permits.evaluate import CLASSES


@pytest.fixture
def stored(mlruns, monkeypatch):
    """A store with the toy main runs of seed 13, and a config pointing at the unweighted one."""
    monkeypatch.setattr(camembert, "load_tokenizer", lambda _settings: toy_tokenizer())
    monkeypatch.setattr(camembert, "load_model", toy_model)
    camembert.run_main(TEXTS, SETTINGS)
    return load_config() | {"training": SETTINGS, "prediction": {"weighting": "none", "seed": 13}}


def test_encode_leaves_out_labels_when_the_rows_have_none():
    features = camembert.encode(toy_tokenizer(), pd.DataFrame({"text": ["villa"]}), 4)
    assert features == [{"input_ids": [1, 4, 2], "attention_mask": [1, 1, 1]}]


def test_classify_gives_a_class_and_a_score_per_class(stored):
    table = predict.classify(["villa villa", "usine"], stored)
    assert table.index.tolist() == ["villa villa", "usine"]
    assert table.columns.tolist() == ["predicted", *CLASSES]
    assert table[CLASSES].sum(axis=1).tolist() == pytest.approx([1, 1])
    assert table[CLASSES].ge(0).all().all()
    assert table["predicted"].tolist() == table[CLASSES].idxmax(axis=1).tolist()


def test_the_stored_run_is_the_main_run_of_the_configured_seed(stored):
    run_id = camembert.stored_run("none", 13)
    run = tracking.runs(tracking.SELECTION).set_index("run_id").loc[run_id]
    assert (run["params.weighting"], run["params.seed"], run["params.stage"]) == (
        "none",
        "13",
        "main",
    )
    with pytest.raises(ValueError):  # no stored model for a seed that never ran
        camembert.stored_run("none", 99)
