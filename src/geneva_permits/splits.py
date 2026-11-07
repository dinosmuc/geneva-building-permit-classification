"""Calendar partitions and the overlap exclusions that keep evaluation records unseen."""

import numpy as np
import pandas as pd

from geneva_permits.config import ANNOTATIONS_DIR, SPLITS_DIR, load_config
from geneva_permits.prepare import load_records

REVIEW_PATH = ANNOTATIONS_DIR / "exploratory_review.csv"
SPLITS_PATH = SPLITS_DIR / "splits.csv"
TRAIN = "train_original"
VALIDATION = "validation"
POOL = "pool_unlabelled"  # takes unlabelled records, every other partition takes target ones
EVALUATION = ["validation", "test_historical", "test_recent"]


def candidate_partitions(records, splits):
    """Each record's partition by filing year and label status; NA outside every partition."""
    partition = pd.Series(pd.NA, index=records.index, dtype="string")
    for name, years in splits.items():
        status = "unlabelled" if name == POOL else "target"
        in_years = records["year"].between(years.get("from", -np.inf), years.get("to", np.inf))
        partition[records["record_status"].eq(status) & in_years.fillna(False)] = name
    return partition


def spans_partitions(candidates, key):
    """True where the row's key value occurs in more than one candidate partition."""
    return candidates.groupby(key)["partition"].transform("nunique").gt(1)


def exclusion_reasons(candidates, reviewed):
    """The first rule that applies to each candidate, else 'retained'.

    Every rule compares against all candidate partitions, so no rule depends on another.
    """
    evaluated = candidates["partition"].isin(EVALUATION)
    reviewed_dossier = candidates["base_dossier"].isin(reviewed["base_dossier"])
    reviewed_text = candidates["text_key"].isin(reviewed["text_key"])
    reasons = np.select(
        [
            spans_partitions(candidates, "base_dossier"),
            evaluated & spans_partitions(candidates, "text_key"),
            evaluated & (reviewed_dossier | reviewed_text),
        ],
        ["dossier_overlap", "text_overlap", "review_exposure"],
        default="retained",
    )
    return pd.Series(reasons, index=candidates.index, dtype="string")


def assign_partitions(records, reviewed_keys):
    """Candidate records with their partition and exclusion reason."""
    reviewed = records.loc[records["record_key"].isin(reviewed_keys)]
    if len(reviewed) != len(set(reviewed_keys)):
        raise ValueError("a reviewed record_key is not in the records")
    partition = candidate_partitions(records, load_config()["data"]["splits"])
    candidates = records.assign(partition=partition).dropna(subset="partition")
    candidates["exclusion_reason"] = exclusion_reasons(candidates, reviewed)
    return candidates


def load_candidates():
    """Every candidate record of the pinned snapshot, excluded ones too, for auditing."""
    review = pd.read_csv(REVIEW_PATH, dtype="string")
    return assign_partitions(load_records(), review["record_key"])


def load_splits():
    """The retained records of every partition: what models train and evaluate on."""
    candidates = load_candidates()
    return candidates.loc[candidates["exclusion_reason"].eq("retained")]


def split_table(candidates):
    """Key and partition of the retained rows, sorted by key: the stored form of the split."""
    retained = candidates.loc[candidates["exclusion_reason"].eq("retained")]
    return retained[["record_key", "partition"]].sort_values("record_key").reset_index(drop=True)


def write_splits(path=SPLITS_PATH):
    """Store the current split; the tests then require the code to keep reproducing it."""
    table = split_table(load_splits())
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return path, len(table)
