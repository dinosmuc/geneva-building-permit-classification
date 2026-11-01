"""Scores over all eight classes, including ones absent from the truth or never predicted."""

import warnings

import numpy as np
import pytest
from sklearn.metrics import f1_score

from geneva_permits.evaluate import (
    BOOTSTRAP,
    CLASSES,
    bootstrap,
    confusion,
    resample_weights,
    scores,
    weighted_scores,
)

TRUE = ["AFF", "AGR", "AGV", "EQU", "NOU", "PI", "REN"]  # VIL absent from the truth
PRED = ["AFF", "AFF", "AGV", "EQU", "NOU", "PI", "PI"]  # AGR and REN never predicted
GROUPS = ["d1", "d1", "d2", "d3", "d3", "d3", "d4"]  # four dossiers over the seven rows


def test_scores_cover_all_eight_classes_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary, per_class = scores(TRUE, PRED)
    assert list(per_class.index) == CLASSES
    assert per_class.loc["VIL"].tolist() == [0, 0, 0, 0]
    assert per_class.loc["AGR"].tolist() == [0, 0, 0, 1]
    assert summary["macro_f1"] == f1_score(
        TRUE, PRED, labels=CLASSES, average="macro", zero_division=0
    )
    assert summary["accuracy"] == 5 / 7


def test_confusion_rows_are_the_true_classes():
    matrix = confusion(TRUE, PRED)
    assert matrix.shape == (8, 8)
    assert matrix.sum(axis=1).tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    assert matrix.loc["REN", "PI"] == 1


def test_resamples_draw_whole_dossiers():
    weights = resample_weights(GROUPS, samples=50, seed=13)
    assert weights.shape == (50, 7)
    assert (weights[:, 0] == weights[:, 1]).all()  # the rows of a dossier move together
    assert (weights[:, 3:6] == weights[:, 3:4]).all()
    assert (weights[:, [0, 2, 3, 6]].sum(axis=1) == 4).all()  # four dossiers drawn each time
    assert (resample_weights(GROUPS, 50, 13) == weights).all()
    assert not (resample_weights(GROUPS, 50, 42) == weights).all()


def test_weighted_scores_match_scores_on_the_repeated_rows():
    weights = np.array([[1, 1, 1, 1, 1, 1, 1], [2, 2, 0, 1, 1, 1, 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = weighted_scores(TRUE, PRED, weights)
    assert table.iloc[0].to_dict() == pytest.approx(scores(TRUE, PRED)[0])
    repeated = np.repeat(np.arange(7), weights[1])
    expected = scores(np.array(TRUE)[repeated], np.array(PRED)[repeated])[0]
    assert table.iloc[1].to_dict() == pytest.approx(expected)


def test_bootstrap_pairs_the_differences_on_the_same_resamples():
    table = bootstrap(TRUE, {"perfect": TRUE, "worse": PRED}, GROUPS, [("perfect", "worse")])
    assert table.index.tolist()[:2] == [("perfect", "macro_f1"), ("perfect", "accuracy")]
    assert table.columns.tolist() == ["estimate", "low", "high"]
    assert table.loc[("perfect", "accuracy")].tolist() == [1, 1, 1]
    worse = table.loc[("worse", "accuracy")]
    assert worse["estimate"] == pytest.approx(5 / 7)
    assert worse["low"] <= worse["estimate"] <= worse["high"]
    assert table.loc[("worse", "macro_f1"), "estimate"] == scores(TRUE, PRED)[0]["macro_f1"]
    # perfect scores 1 on every resample, so the paired difference is 1 - worse on each of them
    difference = table.loc[("perfect - worse", "accuracy")]
    assert difference["estimate"] == pytest.approx(1 - 5 / 7)
    assert difference["low"] == pytest.approx(1 - worse["high"])
    assert difference["high"] == pytest.approx(1 - worse["low"])
    assert BOOTSTRAP["samples"] == 2000 and BOOTSTRAP["level"] == 0.95
