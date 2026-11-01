"""One MLflow store for the whole project, under mlruns/ at the repository root.

Browse it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`. The
read helpers below return pandas tables for the results notebooks.
"""

import subprocess

import mlflow
import pandas as pd

from geneva_permits import config, evaluate, source
from geneva_permits.weights import OPTIONS

SELECTION = "selection"  # experiment holding every run scored on the validation rows
LEARNING_CURVES = "learning_curves"  # the selected settings of both models on training subsets
TEST = "test"  # the one scoring of the 2013 partition: a run per model, plus the bootstrap run
MODELS = ["majority", "linearsvc", "camembert"]  # params.model values, in report order


def set_store(root=None):
    """Point MLflow at the project store under root.

    MLflow otherwise writes the store and the artifacts into the working directory of
    whatever runs it, so a notebook and the CLI would each get their own.
    """
    root = root or config.MLRUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    return root


def set_experiment(name, root=None):
    """Select the experiment in the project store, creating it with its artifacts under root."""
    root = set_store(root)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(root / name))
    return mlflow.set_experiment(name)


def git(*args):
    """Output of a git command in the repository."""
    command = ["git", "-C", str(config.PROJECT_ROOT), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def git_commit():
    """HEAD's hash, with -dirty appended when tracked files hold uncommitted changes."""
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")


def start_run(**kwargs):
    """mlflow.start_run tagged with the run's provenance: git commit and data snapshot SHA-256."""
    tags = {"git_commit": git_commit(), "snapshot_sha256": source.read_lock()["sha256"]}
    return mlflow.start_run(tags=tags, **kwargs)


def log_evaluation(y_true, y_pred):
    """Log summary and per-class F1 as metrics, the per-class and confusion tables as artifacts."""
    summary, per_class = evaluate.scores(y_true, y_pred)
    mlflow.log_metrics(summary | {f"f1_{label}": f1 for label, f1 in per_class["f1"].items()})
    mlflow.log_table(per_class.reset_index(names="class"), "per_class.json")
    mlflow.log_table(evaluate.confusion(y_true, y_pred).reset_index(names="true"), "confusion.json")
    return summary


def runs(experiment, filter_string=""):
    """The runs of an experiment in the project store, one row each with params.* and metrics.*.

    Reading leaves the active experiment alone, so a run started afterwards lands where
    set_experiment put it.
    """
    set_store()
    return mlflow.search_runs(experiment_names=[experiment], filter_string=filter_string)


def best(selected_runs, by):
    """The run with the highest macro-F1 per group of params; an exact tie keeps the earlier run."""
    ordered = selected_runs.sort_values("start_time").sort_values(
        "metrics.macro_f1", ascending=False, kind="stable"
    )
    return ordered.groupby([f"params.{p}" for p in by], sort=False, dropna=False).head(1)


def history(run_id, metric):
    """A metric logged per step in one run, as a Series indexed by step."""
    points = mlflow.MlflowClient().get_metric_history(run_id, metric)
    return pd.Series({p.step: p.value for p in points}, name=metric).sort_index()


def histories(selected_runs, metric, by):
    """The per-step metric of several runs side by side, one column per run, labelled by params."""
    columns = {
        tuple(r[f"params.{p}"] for p in by): history(r["run_id"], metric)
        for _, r in selected_runs.iterrows()
    }
    return pd.DataFrame(columns).rename_axis(index="step", columns=by)


def table(run_id, name):
    """A logged table (per_class.json or confusion.json), indexed by its first column."""
    frame = mlflow.load_table(name, run_ids=[run_id])
    return frame.set_index(frame.columns[0])


def mean_table(run_ids, name):
    """The element-wise mean of a logged table over runs, e.g. over the seeds of one setting."""
    return pd.concat([table(r, name) for r in run_ids]).groupby(level=0, sort=False).mean()


def overview(selection):
    """Validation scores per model, variant and weighting: the majority run, the best grid point of
    each LinearSVC variant and weighting, the mean over seeds of each main CamemBERT weighting."""
    keys = ["params.model", "params.variant", "params.weighting"]
    settings = ["params.C_log2", "params.ngram_range", "params.seed"]
    scores = [
        f"metrics.{m}" for m in ["macro_f1", "accuracy", *(f"f1_{c}" for c in evaluate.CLASSES)]
    ]
    model = selection["params.model"]
    linearsvc = best(selection[model.eq("linearsvc")], ["variant", "weighting"])
    camembert = selection[model.eq("camembert") & selection["params.stage"].eq("main")]
    camembert = (
        camembert.sort_values("params.seed")
        .groupby(keys, dropna=False, sort=False)
        .agg({**dict.fromkeys(scores, "mean"), "params.seed": ", ".join})
        .reset_index()
    )
    rows = pd.concat([selection[model.eq("majority")], linearsvc, camembert])[
        keys + settings + scores
    ]
    rows.columns = [c.split(".", 1)[1] for c in rows.columns]
    rows["model"] = pd.Categorical(rows["model"], MODELS)
    rows["weighting"] = pd.Categorical(rows["weighting"], OPTIONS)
    return rows.sort_values(["model", "variant", "weighting"]).reset_index(drop=True)


def learning_curves():
    """Best validation macro-F1 per training draw of both models: the retuned LinearSVC points, the
    CamemBERT curve runs, and the main CamemBERT runs of their weighting as its full set."""
    curve_runs = runs(LEARNING_CURVES)
    model = curve_runs["params.model"]
    linearsvc = best(curve_runs[model.eq("linearsvc")], ["fraction", "seed"])
    camembert = curve_runs[model.eq("camembert")]
    (weighting,) = camembert["params.weighting"].unique()
    full = runs(
        SELECTION,
        "params.model = 'camembert' and params.stage = 'main' "
        f"and params.weighting = '{weighting}'",
    ).assign(**{"params.fraction": "1.0"})
    columns = [
        "params.model",
        "params.fraction",
        "params.seed",
        "params.train_rows",
        "metrics.macro_f1",
    ]
    rows = pd.concat([linearsvc, camembert, full])[columns]
    rows.columns = [c.split(".", 1)[1] for c in rows.columns]
    rows = rows.astype({"fraction": float, "seed": int, "train_rows": int})
    return rows.sort_values(["model", "fraction", "seed"]).reset_index(drop=True)
