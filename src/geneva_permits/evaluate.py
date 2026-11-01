"""Scores over all eight classes, whether or not a class occurs in the truth or the predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from geneva_permits.config import load_config

CLASSES = load_config()["data"]["target_classes"]
BOOTSTRAP = load_config()["evaluation"]["bootstrap"]  # samples, level and seed; test rows only
ZERO_DIVISION = 0  # a class never predicted gets precision 0 and F1 0 instead of a warning


def scores(y_true, y_pred):
    """Macro-F1 and accuracy, plus precision, recall, F1 and support per class."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=ZERO_DIVISION
    )
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=CLASSES)
    return {"macro_f1": f.mean(), "accuracy": accuracy_score(y_true, y_pred)}, per_class


def confusion(y_true, y_pred):
    """Confusion matrix with the true classes as rows and the predicted ones as columns."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)


def resample_weights(groups, samples, seed):
    """How often each row's group is drawn in each clustered resample, as a (samples, rows) array.

    A resample draws as many groups as there are, with replacement, and keeps every row of each
    drawn group, so the rows of one dossier stay together.
    """
    codes, uniques = pd.factorize(np.asarray(groups))
    uniform = np.full(len(uniques), 1 / len(uniques))
    counts = np.random.default_rng(seed).multinomial(len(uniques), uniform, size=samples)
    return counts[:, codes]


def weighted_scores(y_true, y_pred, weights):
    """Macro-F1, accuracy and per-class F1 of one prediction under each row of weights.

    The same scores as `scores`, read off the weighted confusion matrix of every weighting at once.
    """
    k = len(CLASSES)
    true, pred = (pd.Index(CLASSES).get_indexer(np.asarray(y)) for y in [y_true, y_pred])
    if (true < 0).any() or (pred < 0).any():  # -1: not one of the classes
        raise ValueError(f"labels outside the eight classes {CLASSES}")
    cells = np.eye(k * k)[true * k + pred]  # the confusion cell of each row, one-hot
    matrices = (np.asarray(weights) @ cells).reshape(-1, k, k)
    tp = np.diagonal(matrices, axis1=1, axis2=2)
    doubled = matrices.sum(axis=1) + matrices.sum(axis=2)  # 2 tp + fp + fn per class
    f1 = np.divide(2 * tp, doubled, out=np.zeros_like(tp), where=doubled > 0)  # ZERO_DIVISION
    accuracy = tp.sum(axis=1) / matrices.sum(axis=(1, 2))
    per_class = {f"f1_{c}": f1[:, i] for i, c in enumerate(CLASSES)}
    return pd.DataFrame({"macro_f1": f1.mean(axis=1), "accuracy": accuracy} | per_class)


def mean_scores(y_true, prediction_sets, weights):
    """weighted_scores averaged over the prediction sets of one model, resample by resample.

    One prediction set is a single sequence; CamemBERT's seeds are a list of three of them.
    """
    sets = np.atleast_2d(np.asarray(prediction_sets))
    return sum(weighted_scores(y_true, s, weights) for s in sets) / len(sets)


def bootstrap(y_true, predictions, groups, pairs=(), settings=None):
    """Percentile intervals of every score per model, and of the paired differences
    between the models in pairs, from the same clustered resamples (config evaluation.bootstrap).

    predictions maps model names to their predictions on the rows, or to several prediction
    sets (one per seed) scored as their mean within each resample; pairs are (model, reference)
    names. Rows per (model, metric), columns estimate (every row once), low and high.
    """
    settings = settings or BOOTSTRAP
    weights = resample_weights(groups, settings["samples"], settings["seed"])
    weights = np.vstack([np.ones(len(groups)), weights])  # row 0: the unweighted estimate
    scored = {model: mean_scores(y_true, sets, weights) for model, sets in predictions.items()}
    scored |= {
        f"{model} - {reference}": scored[model] - scored[reference] for model, reference in pairs
    }
    tail = (1 - settings["level"]) / 2
    summary = {
        name: pd.DataFrame(
            {
                "estimate": s.iloc[0],
                "low": s.iloc[1:].quantile(tail),
                "high": s.iloc[1:].quantile(1 - tail),
            }
        )
        for name, s in scored.items()
    }
    return pd.concat(summary, names=["model", "metric"])
