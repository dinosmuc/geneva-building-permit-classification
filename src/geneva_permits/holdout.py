"""The pre-registered 2013 test (config.yaml `evaluation.test`): every listed model scored once.

LinearSVC models are refit on the training rows at the grid point selected on validation; the
CamemBERT models are the stored main runs. Each model gets one run in the test experiment, and
a bootstrap run holds every model's predictions and the interval table.
"""

import mlflow
import pandas as pd
import torch

from geneva_permits import camembert, classical
from geneva_permits.baseline import majority_label
from geneva_permits.config import load_config
from geneva_permits.evaluate import bootstrap
from geneva_permits.identity import current_identity
from geneva_permits.preprocess import load_texts
from geneva_permits.splits import TRAIN, load_splits
from geneva_permits.tracking import (
    SELECTION,
    TEST,
    best,
    git_commit,
    log_evaluation,
    runs,
    set_experiment,
    start_run,
)


def dossiers(texts):
    """The base dossier of each row, from the split records: the bootstrap's cluster key."""
    records = load_splits().set_index("record_key")["base_dossier"]
    return texts["record_key"].map(records)


def score_majority(train, test):
    """The training majority label on every test row."""
    label = majority_label(train["label"])
    with start_run(run_name="majority"):
        mlflow.log_params({"model": "majority", "label": label})
        predictions = [label] * len(test)
        log_evaluation(test["label"], predictions)
    return predictions


def score_linearsvc(spec, train, test, settings):
    """A variant and weighting refit at its best validation grid point, scored on the test rows."""
    variant, weighting = spec["variant"], spec["weighting"]
    grid = runs(
        SELECTION,
        f"params.model = 'linearsvc' and params.variant = '{variant}' "
        f"and params.weighting = '{weighting}'",
    )
    point = best(grid, ["variant", "weighting"]).iloc[0]
    c_log2 = int(point["params.C_log2"])
    ngram_range = [int(n) for n in point["params.ngram_range"].split("-")]
    with start_run(run_name=f"linearsvc-{variant}-{weighting}"):
        pipeline, _ = classical.fit_point(variant, weighting, c_log2, ngram_range, train, settings)
        mlflow.log_params(
            {
                "model": "linearsvc",
                "variant": variant,
                "weighting": weighting,
                "C_log2": c_log2,
                "ngram_range": point["params.ngram_range"],
                "selected_run": point["run_id"],
            }
        )
        predictions = pipeline.predict(test[classical.VARIANTS[variant]])
        log_evaluation(test["label"], predictions)
    return predictions


def score_camembert(weighting, seed, test, config, identity):
    """The current main model of a weighting and seed, scored on the test rows."""
    run_id = camembert.stored_run(weighting, seed, identity)
    settings = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = camembert.load_stored(run_id).to(device)
    loader = camembert.batches(
        camembert.load_tokenizer(config["models"]["primary"]), test, settings
    )
    with start_run(run_name=f"camembert-{weighting}-{seed}"):
        mlflow.log_params(
            {
                "model": "camembert",
                "weighting": weighting,
                "seed": seed,
                "stored_run": run_id,
                "fingerprint": identity["fingerprint"],
            }
        )
        predictions = camembert.predict(model, loader, device, settings["precision"])
        log_evaluation(test["label"], predictions)
    return predictions


def run_test(texts=None, groups=None, config=None):
    """Score every model of the plan once on the test partition; the bootstrap table.

    Refuses a dirty tree, since the runs must carry a plain commit hash, and a second scoring
    once a bootstrap run has finished (config evaluation.test_policy: once). The partial runs
    of an interrupted attempt are deleted and the rerun is tagged.
    """
    config = config or load_config()
    plan, settings = config["evaluation"]["test"], config["evaluation"]["bootstrap"]
    if git_commit().endswith("-dirty"):
        raise RuntimeError("uncommitted changes: the test runs must carry a plain commit hash")
    set_experiment(TEST)
    previous = runs(TEST)
    if not previous.empty:
        names = previous.get("tags.mlflow.runName", pd.Series("", index=previous.index))
        if (names.eq("bootstrap") & previous["status"].eq("FINISHED")).any():
            raise RuntimeError(
                "the test partition was already scored (evaluation.test_policy: once)"
            )
        for run_id in previous["run_id"]:
            mlflow.delete_run(run_id)
    texts = load_texts() if texts is None else texts
    train = texts[texts["partition"].eq(TRAIN)]
    test = texts[texts["partition"].eq(plan["partition"])]
    groups = dossiers(test) if groups is None else groups
    models = plan["models"]
    predictions = {"majority": score_majority(train, test)}
    for spec in models["linearsvc"]:
        name = f"linearsvc {spec['variant']} {spec['weighting']}"
        predictions[name] = score_linearsvc(spec, train, test, config["classical"])
    weighting, identity = models["camembert"]["weighting"], current_identity(config, texts)
    predictions["camembert"] = [
        score_camembert(weighting, seed, test, config, identity)
        for seed in models["camembert"]["seeds"]
    ]
    comparisons = {
        "primary": [plan["primary_comparison"]],
        "secondary": plan["secondary_comparisons"],
    }
    pairs = [tuple(pair) for role in comparisons.values() for pair in role]
    roles = {
        " - ".join(pair): role
        for role, pairs_of_role in comparisons.items()
        for pair in pairs_of_role
    }
    table = bootstrap(test["label"], predictions, groups, pairs, settings)
    table.insert(0, "comparison", table.index.get_level_values("model").map(roles).fillna(""))
    with start_run(run_name="bootstrap"):
        mlflow.log_params(
            settings | {"groups": plan["groups"], "distinct_groups": pd.Series(groups).nunique()}
        )
        mlflow.log_table(table.reset_index(), "bootstrap.json")
        mlflow.log_table(
            prediction_table(test, predictions, models["camembert"]["seeds"]), "predictions.json"
        )
    if not previous.empty:
        note = f"after an interrupted attempt; {len(previous)} partial runs deleted"
        for run_id in runs(TEST)["run_id"]:
            mlflow.MlflowClient().set_tag(run_id, "rerun", note)
    return table


def prediction_table(test, predictions, seeds):
    """Every model's predictions beside the true labels, one column per model or seed."""
    columns = {"record_key": test["record_key"].to_numpy(), "label": test["label"].to_numpy()}
    for name, predicted in predictions.items():
        if name == "camembert":
            columns |= {f"camembert {seed}": p for seed, p in zip(seeds, predicted, strict=True)}
        else:
            columns[name] = predicted
    return pd.DataFrame(columns)
