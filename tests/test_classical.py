"""The LinearSVC grid: per-row weights, logged convergence, edge extension and selection."""

import mlflow
import pandas as pd
import pytest

from geneva_permits.classical import edge_extension, grid, run_grid, select, selected

TEXTS = pd.DataFrame(
    {
        "partition": ["train_original"] * 8 + ["validation"] * 4,
        "label": ["VIL", "REN", "VIL", "REN", "PI", "VIL", "REN", "PI"]
        + ["VIL", "REN", "PI", "VIL"],
        "text": [
            "construction villa",
            "rénovation toiture",
            "villa avec garage",
            "rénovation façade",
            "pose capteurs",
            "villa jumelle",
            "rénovation cuisine",
            "pose panneaux",
            "villa neuve",
            "rénovation totale",
            "pose vélux",
            "villa et piscine",
        ],
    }
)


def test_grid_order_breaks_ties_towards_none_small_c_and_short_ngrams():
    assert grid([-1, 1], [[1, 1], [1, 2]])[:3] == [
        ("none", -1, [1, 1]),
        ("none", -1, [1, 2]),
        ("none", 1, [1, 1]),
    ]


@pytest.mark.parametrize(("best", "extra"), [(-5, [-7, -9]), (15, [17, 19]), (3, [])])
def test_edge_extension_adds_two_steps_beyond_the_edge(best, extra):
    assert edge_extension(best, [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]) == extra


def test_run_grid_logs_every_point_and_extends_past_an_edge(mlruns, classical_settings):
    results = run_grid("A", TEXTS, classical_settings)
    base = results[~results["extension"]]
    assert len(base) == 3 * 2  # weightings x C values, one n-gram range
    assert len(results) in {6, 12}  # plus the two-step extension when the best C is on an edge
    assert results["converged"].all() if "converged" in results else True
    best = select(results)
    assert best["macro_f1"] == results["macro_f1"].max()
    runs = mlflow.search_runs(experiment_names=["selection"])
    assert len(runs) == len(results)
    assert set(runs["params.weighting"]) == {"none", "medium", "full"}
    assert (runs["metrics.converged"] == 1).all()


def test_a_convergence_warning_is_logged_not_silenced(mlruns, classical_settings):
    results = run_grid("A", TEXTS, classical_settings | {"max_iter": 1})
    assert (results["n_iter"] == 1).all()
    runs = mlflow.search_runs(experiment_names=["selection"])
    assert (runs["metrics.converged"] == 0).all()
    assert runs["tags.convergence_warning"].str.contains("failed to converge").all()


def test_selected_reads_the_best_from_the_store(mlruns, classical_settings):
    results = run_grid("A", TEXTS, classical_settings)
    mlflow.set_tracking_uri("")  # a fresh process has no store selected either
    assert selected() == ("A", select(results)["weighting"])
