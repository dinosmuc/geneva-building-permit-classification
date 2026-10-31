"""One MLflow store for the whole project, under mlruns/ at the repository root.

Browse it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
"""

import mlflow

from geneva_permits import config, evaluate

SELECTION = "selection"  # experiment holding every run scored on the validation rows
LEARNING_CURVES = "learning_curves"  # the selected classical model retuned on training subsets


def set_experiment(name, root=None):
    """Select the experiment in the project store, creating it with its artifacts under root.

    MLflow otherwise writes the store and the artifacts into the working directory of
    whatever runs it, so a notebook and the CLI would each get their own.
    """
    root = root or config.MLRUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(root / name))
    return mlflow.set_experiment(name)


def log_evaluation(y_true, y_pred):
    """Log summary and per-class F1 as metrics, the per-class and confusion tables as artifacts."""
    summary, per_class = evaluate.scores(y_true, y_pred)
    mlflow.log_metrics(summary | {f"f1_{label}": f1 for label, f1 in per_class["f1"].items()})
    mlflow.log_table(per_class.reset_index(names="class"), "per_class.json")
    mlflow.log_table(evaluate.confusion(y_true, y_pred).reset_index(names="true"), "confusion.json")
    return summary


def runs(experiment, filter_string=""):
    """The runs of an experiment in the project store, one row each with params.* and metrics.*."""
    set_experiment(experiment)
    return mlflow.search_runs(experiment_names=[experiment], filter_string=filter_string)
