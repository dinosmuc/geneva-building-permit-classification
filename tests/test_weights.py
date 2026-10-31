"""Per-row weights: the three options on the training class counts."""

import numpy as np
import pytest

from geneva_permits.weights import OPTIONS, class_weights, row_weights

# class counts of the training partition in the frozen split (30,782 rows)
COUNTS = {
    "AFF": 437,
    "AGR": 1310,
    "AGV": 1258,
    "EQU": 2808,
    "NOU": 1709,
    "PI": 9917,
    "REN": 11712,
    "VIL": 1631,
}
LABELS = np.repeat(list(COUNTS), list(COUNTS.values()))


@pytest.mark.parametrize(
    ("option", "aff", "ren"), [("none", 1, 1), ("medium", 3.38, 0.65), ("full", 8.80, 0.33)]
)
def test_class_weights_on_the_training_counts(option, aff, ren):
    weights = class_weights(LABELS, option)
    assert weights["AFF"] == pytest.approx(aff, abs=0.005)
    assert weights["REN"] == pytest.approx(ren, abs=0.005)


@pytest.mark.parametrize("option", OPTIONS)
def test_every_option_averages_one_per_row(option):
    weights = row_weights(LABELS, option)
    assert len(weights) == len(LABELS)
    assert weights.mean() == pytest.approx(1)


def test_unknown_option_is_refused():
    with pytest.raises(ValueError, match="balanced"):
        class_weights(LABELS, "balanced")
