"""One MLflow store for the whole project, under mlruns/ at the repository root.

Browse it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
"""

import mlflow

from geneva_permits.config import MLRUNS_DIR


def set_experiment(name, root=MLRUNS_DIR):
    """Select the experiment in the project store, creating it with its artifacts under root.

    MLflow otherwise writes the store and the artifacts into the working directory of
    whatever runs it, so a notebook and the CLI would each get their own.
    """
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(root / name))
    return mlflow.set_experiment(name)
