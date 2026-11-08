"""What identifies a CamemBERT run beyond its point, and the stored runs that carry it.

The fingerprint hashes the training settings fit uses, the pinned model and the training and
validation rows the stage was given; a curve point draws its subset from those rows by its
fraction and seed. Every reader uses only finished runs with the current fingerprint and
learning rate.
"""

import hashlib
import json

import pandas as pd

from geneva_permits.config import load_config
from geneva_permits.preprocess import load_texts, train_validation

TRAINED_WITH = [
    "precision",
    "max_seq_length",
    "batch_size",
    "epochs",
    "checkpoint",
    "schedule",
    "warmup_ratio",
    "weight_decay",
    "adam_betas",
    "adam_epsilon",
    "max_grad_norm",
]
DATA_COLUMNS = ["record_key", "text", "label"]
POINT = ["stage", "weighting", "seed", "fraction"]  # tells apart the runs of one identity


def rows_sha256(rows):
    """SHA-256 of the rows' keys, texts and labels, in their order."""
    table = rows[DATA_COLUMNS].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(table.encode()).hexdigest()


def fingerprint(settings, model_settings, train, validation):
    """SHA-256 of the training settings fit uses, the pinned model and the rows it is given."""
    identity = {
        "training": {name: settings[name] for name in TRAINED_WITH},
        "model": {name: model_settings[name] for name in ["id", "revision"]},
        "data": {"train": rows_sha256(train), "validation": rows_sha256(validation)},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def current_identity(config=None, texts=None):
    """The fingerprint and learning rate of the configured runs on the texts, by default the
    stored ones."""
    config = config or load_config()
    settings = config["training"]
    train, validation = train_validation(load_texts() if texts is None else texts)
    return {
        "fingerprint": fingerprint(settings, config["models"]["primary"], train, validation),
        "learning_rate": settings["learning_rate"],
    }


def current_runs(frame, identity=None):
    """The finished CamemBERT runs of a runs table with the identity's fingerprint and learning
    rate; raises when two share a stage, weighting, seed and fraction rather than average them."""
    identity = identity or current_identity()
    names = ["model", "fingerprint", "learning_rate", *POINT]
    params = frame.reindex(columns=[f"params.{name}" for name in names])  # missing ones stay NaN
    params.columns = names
    keep = (
        params["model"].eq("camembert")
        & frame.reindex(columns=["status"])["status"].eq("FINISHED")
        & params["fingerprint"].eq(identity["fingerprint"])
        & pd.to_numeric(params["learning_rate"]).eq(identity["learning_rate"])
    ).to_numpy()
    shared = params.loc[keep, POINT].duplicated(keep=False).to_numpy()
    if shared.any():
        run_ids = sorted(frame.loc[keep, "run_id"][shared])
        raise ValueError(f"current CamemBERT runs share a point: {', '.join(run_ids)}")
    return frame[keep]
