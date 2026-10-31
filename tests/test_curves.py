"""Learning curves: nested draws, one per seed, retuned with the weighting fixed."""

import mlflow
import numpy as np
import pandas as pd

from geneva_permits.curves import draws, nested_subsets, run_curves

TEXTS = pd.DataFrame(
    {
        "partition": ["train_original"] * 12 + ["validation"] * 4,
        "label": ["VIL", "REN"] * 6 + ["VIL", "REN", "REN", "VIL"],
        "text": [f"{'villa' if i % 2 == 0 else 'rénovation'} mot{i}" for i in range(16)],
    }
)


def test_subsets_are_nested_and_sized_by_fraction():
    subsets = nested_subsets(30_782, [0.1, 0.3, 1.0], seed=13)
    assert [len(subsets[f]) for f in [0.1, 0.3, 1.0]] == [3078, 9235, 30_782]
    assert set(subsets[0.1]) < set(subsets[0.3]) < set(subsets[1.0])
    assert not np.array_equal(subsets[0.1], nested_subsets(30_782, [0.1], seed=42)[0.1])
    assert np.array_equal(subsets[0.1], nested_subsets(30_782, [0.1], seed=13)[0.1])


def test_the_full_set_is_drawn_once():
    assert draws([0.1, 1.0], [13, 42]) == [(0.1, 13), (0.1, 42), (1.0, 13)]


def test_run_curves_fixes_the_weighting_and_logs_each_draw(mlruns, classical_settings):
    best = run_curves("A", "medium", TEXTS, classical_settings, fractions=[0.5, 1.0], seeds=[1, 2])
    assert best[["fraction", "seed"]].values.tolist() == [[0.5, 1], [0.5, 2], [1.0, 1]]
    assert set(best["weighting"]) == {"medium"}
    assert best.loc[best["fraction"].eq(0.5), "train_rows"].eq(6).all()
    runs = mlflow.search_runs(experiment_names=["learning_curves"])
    assert set(runs["params.weighting"]) == {"medium"}
    assert set(runs["params.fraction"]) == {"0.5", "1.0"}
    assert len(runs) >= 3 * 2  # three draws, two C values each, plus any edge extension
