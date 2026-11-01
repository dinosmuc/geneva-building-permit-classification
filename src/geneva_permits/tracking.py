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


def load_table(run_id, name):
    """A table logged to a run, read from its own experiment whatever experiment is active."""
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    return mlflow.MlflowClient().load_table(experiment_id, name, run_ids=[run_id])


def table(run_id, name):
    """A logged table (per_class.json or confusion.json), indexed by its first column."""
    frame = load_table(run_id, name)
    return frame.set_index(frame.columns[0])


def mean_table(run_ids, name):
    """The element-wise mean of a logged table over runs, e.g. over the seeds of one setting."""
    return pd.concat([table(r, name) for r in run_ids]).groupby(level=0, sort=False).mean()


KEYS = ["params.model", "params.variant", "params.weighting"]
SETTINGS = ["params.C_log2", "params.ngram_range", "params.seed"]
SCORES = [f"metrics.{m}" for m in ["macro_f1", "accuracy", *(f"f1_{c}" for c in evaluate.CLASSES)]]


def tidy(rows):
    """Keys, settings and scores without the params./metrics. prefixes, in report order."""
    rows = rows.reindex(columns=KEYS + SETTINGS + SCORES)  # a setting a model lacks stays NaN
    rows.columns = [c.split(".", 1)[1] for c in rows.columns]
    rows["model"] = pd.Categorical(rows["model"], MODELS)
    rows["weighting"] = pd.Categorical(rows["weighting"], OPTIONS)
    return rows.sort_values(["model", "variant", "weighting", "seed"]).reset_index(drop=True)


def summarize(model_runs):
    """One row per model, variant and weighting: the majority run, the best LinearSVC grid point,
    the CamemBERT mean over seeds (seed column: the seeds averaged)."""
    model = model_runs["params.model"]
    linearsvc = best(model_runs[model.eq("linearsvc")], ["variant", "weighting"])
    camembert = (
        model_runs[model.eq("camembert")]
        .sort_values("params.seed")
        .groupby(KEYS, dropna=False, sort=False)
        .agg({**dict.fromkeys(SCORES, "mean"), "params.seed": ", ".join})
        .reset_index()
    )
    return tidy(pd.concat([model_runs[model.eq("majority")], linearsvc, camembert]))


def overview(selection):
    """Validation scores per model, variant and weighting: the majority run, the best grid point of
    each LinearSVC variant and weighting, the mean over seeds of each main CamemBERT weighting."""
    model, stage = selection["params.model"], selection["params.stage"]
    return summarize(selection[model.ne("camembert") | stage.eq("main")])


def test_runs():
    """The model runs of the test experiment, one row each, CamemBERT once per seed."""
    scored = runs(TEST)
    return tidy(scored[scored["params.model"].notna()])  # the bootstrap run carries no model


def test_overview():
    """Test scores per model, variant and weighting, CamemBERT as the mean over its seeds."""
    scored = runs(TEST)
    return summarize(scored[scored["params.model"].notna()])


def camembert_seeds():
    """Each CamemBERT seed on the test beside its stored main run: kept epoch, validation score."""
    tested = runs(TEST, "params.model = 'camembert'").set_index("params.stored_run")
    stored = runs(SELECTION, "params.model = 'camembert' and params.stage = 'main'")
    stored = stored.set_index("run_id").loc[tested.index]
    rows = pd.DataFrame(
        {
            "seed": tested["params.seed"].astype(int).to_numpy(),
            "kept_epoch": stored["metrics.kept_epoch"].astype(int).to_numpy(),
            "validation_macro_f1": stored["metrics.macro_f1"].to_numpy(),
            "test_macro_f1": tested["metrics.macro_f1"].to_numpy(),
            "test_accuracy": tested["metrics.accuracy"].to_numpy(),
        }
    )
    return rows.sort_values("seed").reset_index(drop=True)


def bootstrap_run():
    """The run holding the test's bootstrap tables."""
    return runs(TEST, "attributes.run_name = 'bootstrap'").iloc[0]


def bootstrap_table():
    """The stored interval table: rows (model, metric), columns comparison, estimate, low, high."""
    return load_table(bootstrap_run()["run_id"], "bootstrap.json").set_index(["model", "metric"])


def test_predictions():
    """Every model's test predictions beside the true labels, indexed by record key."""
    return load_table(bootstrap_run()["run_id"], "predictions.json").set_index("record_key")


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
