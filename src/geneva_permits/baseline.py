"""Majority-class baseline: every validation row gets the most frequent training label."""

import mlflow

from geneva_permits.preprocess import load_texts, train_validation
from geneva_permits.tracking import SELECTION, log_evaluation, set_experiment, start_run


def majority_label(labels):
    """The most frequent label."""
    return labels.value_counts().idxmax()


def run_majority(texts=None):
    """Score the training majority label on the validation rows and log the run."""
    texts = load_texts() if texts is None else texts
    train, validation = train_validation(texts)
    label = majority_label(train["label"])
    set_experiment(SELECTION)
    with start_run(run_name="majority"):
        mlflow.log_params({"model": "majority", "label": label, "train_rows": len(train)})
        return log_evaluation(validation["label"], [label] * len(validation))
