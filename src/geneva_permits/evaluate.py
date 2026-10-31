"""Scores over all eight classes, whether or not a class occurs in the truth or the predictions."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from geneva_permits.config import load_config

CLASSES = load_config()["data"]["target_classes"]
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
