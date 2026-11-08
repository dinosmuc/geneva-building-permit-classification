"""One MLflow store for the whole project, under mlruns/ at the repository root.

Browse it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`. The
read helpers below return pandas tables for the results notebooks; they read CamemBERT runs only
through identity.current_runs, with the current fingerprint and learning rate.
"""

import subprocess

import mlflow
import pandas as pd

from geneva_permits import config, evaluate, source
from geneva_permits.identity import current_identity, current_runs
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


def camembert_runs(experiment, stage, identity=None):
    """The current CamemBERT runs of a stage (identity.current_runs)."""
    stage_runs = runs(experiment, f"params.model = 'camembert' and params.stage = '{stage}'")
    return current_runs(stage_runs, identity)


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
    the CamemBERT mean over seeds (seed column: the seeds averaged). Two CamemBERT runs of one
    weighting and seed raise rather than enter the mean."""
    model = model_runs["params.model"]
    linearsvc = best(model_runs[model.eq("linearsvc")], ["variant", "weighting"])
    camembert = model_runs[model.eq("camembert")]
    if camembert.reindex(columns=[*KEYS, "params.seed"]).duplicated().any():
        raise ValueError("two CamemBERT runs share a weighting and seed; refusing to average them")
    camembert = (
        camembert.sort_values("params.seed")
        .groupby(KEYS, dropna=False, sort=False)
        .agg({**dict.fromkeys(SCORES, "mean"), "params.seed": ", ".join})
        .reset_index()
    )
    return tidy(pd.concat([model_runs[model.eq("majority")], linearsvc, camembert]))


def overview(selection, identity=None):
    """Validation scores per model, variant and weighting: the majority run, the best grid point of
    each LinearSVC variant and weighting, the mean over seeds of each current main CamemBERT
    weighting."""
    main = current_runs(selection[selection["params.stage"].eq("main")], identity)
    return summarize(pd.concat([selection[selection["params.model"].ne("camembert")], main]))


# holdout_*, not test_*: pytest collects test_* functions that a test module imports
def holdout_models(identity=None):
    """The model runs of the test experiment; a CamemBERT run only when it scored a current main
    run."""
    scored = runs(TEST)
    scored = scored[scored["params.model"].notna()]  # the bootstrap run carries no model
    main = camembert_runs(SELECTION, "main", identity)
    stored = scored.reindex(columns=["params.stored_run"])["params.stored_run"]
    return scored[scored["params.model"].ne("camembert") | stored.isin(main["run_id"])]


def holdout_runs(identity=None):
    """The model runs of the test experiment, one row each, CamemBERT once per seed."""
    return tidy(holdout_models(identity))


def holdout_overview(identity=None):
    """Test scores per model, variant and weighting, CamemBERT as the mean over its seeds."""
    return summarize(holdout_models(identity))


def camembert_seeds(identity=None):
    """Each CamemBERT seed on the test beside its stored main run: kept epoch, validation score."""
    identity = identity or current_identity()
    tested = holdout_models(identity)
    tested = tested[tested["params.model"].eq("camembert")].set_index("params.stored_run")
    stored = camembert_runs(SELECTION, "main", identity).set_index("run_id").loc[tested.index]
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


def holdout_predictions():
    """Every model's test predictions beside the true labels, indexed by record key."""
    return load_table(bootstrap_run()["run_id"], "predictions.json").set_index("record_key")


def learning_curves(identity=None):
    """Best validation macro-F1 per training draw of both models: the retuned LinearSVC points, the
    current CamemBERT curve runs, and the current main runs of their weighting as its full set."""
    identity = identity or current_identity()
    curve_runs = runs(LEARNING_CURVES)
    linearsvc = best(curve_runs[curve_runs["params.model"].eq("linearsvc")], ["fraction", "seed"])
    camembert = current_runs(curve_runs, identity)
    (weighting,) = camembert["params.weighting"].unique()
    full = camembert_runs(SELECTION, "main", identity)
    full = full[full["params.weighting"].eq(weighting)].assign(**{"params.fraction": "1.0"})
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
