"""TF-IDF + LinearSVC over the pre-registered grid (config.yaml `classical`), one run per fit."""

import itertools
import warnings

import mlflow
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from geneva_permits.config import load_config
from geneva_permits.preprocess import load_texts, train_validation
from geneva_permits.tracking import SELECTION, log_evaluation, runs, set_experiment
from geneva_permits.weights import OPTIONS, row_weights

VARIANTS = {"A": "text", "B": "lemmas"}  # the column each variant reads
EDGE_STEPS = 2  # grid points added beyond an edge that holds the best C


def make_pipeline(c_log2, ngram_range, settings):
    """TF-IDF fitted on the training rows only, then the primal LinearSVC."""
    tfidf = TfidfVectorizer(token_pattern=settings["token_pattern"], ngram_range=tuple(ngram_range))
    svm = LinearSVC(
        C=2.0**c_log2,
        dual=settings["dual"],
        max_iter=settings["max_iter"],
        random_state=settings["random_state"],
    )
    return Pipeline([("tfidf", tfidf), ("svm", svm)])


def fit(pipeline, texts, labels, weighting):
    """Fit with per-row weights; returns the iteration count and any convergence warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(texts, labels, svm__sample_weight=row_weights(labels, weighting))
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return pipeline["svm"].n_iter_, "; ".join(messages)


def grid(c_log2s, ngram_ranges, weightings=OPTIONS):
    """Every (weighting, log2 C, n-gram range) point, in the order ties are broken."""
    return list(itertools.product(weightings, c_log2s, ngram_ranges))


def edge_extension(best_c_log2, c_log2s):
    """Two further grid steps beyond the edge the best C sits on, or none."""
    if len(c_log2s) < 2:
        return []
    step = c_log2s[1] - c_log2s[0]
    if best_c_log2 == min(c_log2s):
        return [best_c_log2 - step * k for k in range(1, EDGE_STEPS + 1)]
    if best_c_log2 == max(c_log2s):
        return [best_c_log2 + step * k for k in range(1, EDGE_STEPS + 1)]
    return []


def select(results):
    """The row with the highest validation macro-F1; an exact tie keeps the earlier grid point."""
    return results.loc[results["macro_f1"].idxmax()]


def run_points(variant, points, train, validation, settings, extra=None):
    """Fit and score each grid point on the validation rows, one MLflow run each."""
    column, rows = VARIANTS[variant], []
    for weighting, c_log2, ngram_range in points:
        with mlflow.start_run(run_name=f"linearsvc-{variant}"):
            pipeline = make_pipeline(c_log2, ngram_range, settings)
            n_iter, warning = fit(pipeline, train[column], train["label"], weighting)
            params = {
                "model": "linearsvc",
                "variant": variant,
                "weighting": weighting,
                "C_log2": c_log2,
                "ngram_range": "-".join(map(str, ngram_range)),
                "extension": False,
                "train_rows": len(train),
            } | (extra or {})
            mlflow.log_params(params)
            mlflow.log_metrics({"n_iter": n_iter, "converged": n_iter < settings["max_iter"]})
            if warning:
                mlflow.set_tag("convergence_warning", warning)
            summary = log_evaluation(validation["label"], pipeline.predict(validation[column]))
            rows.append(params | summary | {"n_iter": n_iter})
    return pd.DataFrame(rows)


def tune(variant, train, validation, settings, weightings=OPTIONS, extra=None):
    """The grid over the weightings, extended by two C steps when the best C sits on an edge."""
    c_log2s, ngram_ranges = settings["C_log2"], settings["ngram_ranges"]
    points = grid(c_log2s, ngram_ranges, weightings)
    results = run_points(variant, points, train, validation, settings, extra)
    extra_c = edge_extension(select(results)["C_log2"], c_log2s)
    if extra_c:
        points = grid(extra_c, ngram_ranges, weightings)
        extension = (extra or {}) | {"extension": True}
        more = run_points(variant, points, train, validation, settings, extension)
        results = pd.concat([results, more], ignore_index=True)
    return results


def run_grid(variant, texts=None, settings=None):
    """The whole grid of a variant on the full training set, in the selection experiment."""
    settings = settings or load_config()["classical"]
    texts = load_texts() if texts is None else texts
    train, validation = train_validation(texts)
    set_experiment(SELECTION)
    return tune(variant, train, validation, settings)


def selected():
    """Variant and weighting of the best LinearSVC run on validation; ties keep the earlier run."""
    grid_runs = runs(SELECTION, "params.model = 'linearsvc'").sort_values("start_time")
    best = grid_runs.loc[grid_runs["metrics.macro_f1"].idxmax()]
    return best["params.variant"], best["params.weighting"]
