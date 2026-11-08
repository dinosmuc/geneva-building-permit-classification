"""The run identity: a fingerprint of the settings, the model and the rows; readers keep current
runs only."""

import pandas as pd
import pytest
from test_camembert import SETTINGS, TEXTS

from geneva_permits.config import load_config
from geneva_permits.identity import current_runs, fingerprint
from geneva_permits.preprocess import train_validation

MODEL = load_config()["models"]["primary"]
TRAIN, VALIDATION = train_validation(TEXTS)
IDENTITY = {"fingerprint": "f" * 64, "learning_rate": 3e-5}


def test_the_fingerprint_follows_the_settings_the_model_and_the_rows():
    base = fingerprint(SETTINGS, MODEL, TRAIN, VALIDATION)
    unchanged = [
        fingerprint(SETTINGS | {"seeds": [1], "learning_rate": 1.0}, MODEL, TRAIN, VALIDATION),
        fingerprint(SETTINGS, MODEL, TRAIN.assign(lemmas="x"), VALIDATION),  # not trained on
    ]
    assert unchanged == [base, base]
    relabelled = VALIDATION.assign(label=["PI", *VALIDATION["label"][1:]])
    changed = [
        fingerprint(SETTINGS | {"max_grad_norm": 0.5}, MODEL, TRAIN, VALIDATION),
        fingerprint(SETTINGS, MODEL | {"revision": "0" * 40}, TRAIN, VALIDATION),
        fingerprint(SETTINGS, MODEL, TRAIN.assign(text="villa"), VALIDATION),
        fingerprint(SETTINGS, MODEL, TRAIN.iloc[::-1], VALIDATION),  # the order trained in
        fingerprint(SETTINGS, MODEL, TRAIN, relabelled),
    ]
    assert len({base, *changed}) == 1 + len(changed)


def test_current_runs_are_the_finished_camembert_runs_of_the_identity():
    run = {
        "status": "FINISHED",
        "params.model": "camembert",
        "params.fingerprint": IDENTITY["fingerprint"],
        "params.learning_rate": "3e-05",
        "params.stage": "main",
        "params.weighting": "none",
        "params.seed": "13",
    }
    frame = pd.DataFrame(
        [
            run | {"run_id": "current"},
            run | {"run_id": "other seed", "params.seed": "42"},
            run | {"run_id": "curve", "params.stage": "curves", "params.fraction": "0.1"},
            run | {"run_id": "failed", "status": "FAILED"},
            run | {"run_id": "other fingerprint", "params.fingerprint": "0" * 64},
            run | {"run_id": "other learning rate", "params.learning_rate": "5e-05"},
            run | {"run_id": "linearsvc", "params.model": "linearsvc"},
        ]
    )
    assert current_runs(frame, IDENTITY)["run_id"].tolist() == ["current", "other seed", "curve"]
    assert current_runs(frame.drop(columns="params.fingerprint"), IDENTITY).empty
    copy = pd.DataFrame([run | {"run_id": "copy"}])
    with pytest.raises(ValueError, match="share a point: copy, current"):
        current_runs(pd.concat([frame, copy]), IDENTITY)
