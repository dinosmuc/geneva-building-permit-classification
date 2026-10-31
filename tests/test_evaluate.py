"""Scores over all eight classes, including ones absent from the truth or never predicted."""

import warnings

from sklearn.metrics import f1_score

from geneva_permits.evaluate import CLASSES, confusion, scores

TRUE = ["AFF", "AGR", "AGV", "EQU", "NOU", "PI", "REN"]  # VIL absent from the truth
PRED = ["AFF", "AFF", "AGV", "EQU", "NOU", "PI", "PI"]  # AGR and REN never predicted


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
