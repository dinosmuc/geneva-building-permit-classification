"""Partitions and overlap exclusions, on tiny synthetic frames and on the pinned snapshot."""

import pandas as pd
import pytest

from geneva_permits.config import load_config
from geneva_permits.splits import (
    EVALUATION,
    REVIEW_PATH,
    SPLITS_PATH,
    assign_partitions,
    load_splits,
    split_table,
)


def records_frame(*rows):
    """A records-like frame; each row gives (year, record_status, base_dossier, text_key)."""
    frame = pd.DataFrame(rows, columns=["year", "record_status", "base_dossier", "text_key"])
    frame["record_key"] = [f"key {index}" for index in range(len(frame))]
    text_columns = ["record_status", "base_dossier", "text_key", "record_key"]
    return frame.astype({"year": "Int64"} | dict.fromkeys(text_columns, "string"))


def reasons(frame, reviewed_keys=()):
    return assign_partitions(frame, list(reviewed_keys))["exclusion_reason"].tolist()


def test_partitions_follow_filing_year_and_label_status():
    # boundary years of config.yaml; every row has its own dossier and text
    frame = records_frame(
        (2010, "target", "DD 1", "a"),
        (2011, "target", "DD 2", "b"),
        (2012, "target", "DD 3", "c"),
        (2013, "target", "DD 4", "d"),
        (2014, "target", "DD 5", "e"),
        (2026, "target", "DD 6", "f"),
        (2027, "target", "DD 7", "g"),
        (2013, "unlabelled", "DD 8", "h"),
        (2014, "unlabelled", "DD 9", "i"),
        (2020, "am", "DD 10", "j"),
        (None, "no_description", "DD 11", None),
    )
    splits = assign_partitions(frame, [])
    assert dict(zip(splits["record_key"], splits["partition"], strict=True)) == {
        "key 0": "train_original",
        "key 1": "validation",
        "key 2": "validation",
        "key 3": "test_historical",
        "key 4": "test_recent",
        "key 5": "test_recent",
        "key 8": "pool_unlabelled",
    }


def test_dossier_overlap_applies_everywhere_and_comes_first():
    frame = records_frame(
        (2005, "target", "DD 1", "mur"),
        (2013, "target", "DD 1", "mur"),
        (2020, "unlabelled", "DD 1", "piscine"),
        (2020, "target", "DD 2", "villa"),
    )
    assert reasons(frame) == ["dossier_overlap", "dossier_overlap", "dossier_overlap", "retained"]


def test_text_overlap_excludes_evaluation_rows_only():
    frame = records_frame(
        (2005, "target", "DD 1", "mur"),
        (2020, "target", "DD 2", "mur"),
        (2020, "unlabelled", "DD 3", "mur"),
    )
    assert reasons(frame) == ["retained", "text_overlap", "retained"]


def test_review_exposure_excludes_evaluation_rows_only():
    frame = records_frame(
        (2020, "target", "DD 1", "mur"),  # reviewed
        (2020, "target", "DD 1", "piscine"),  # same dossier
        (2020, "target", "DD 2", "mur"),  # same text
        (2005, "target", "DD 3", "villa"),  # reviewed, in training
        (2020, "target", "DD 4", "garage"),
    )
    assert reasons(frame, ["key 0", "key 3"]) == [
        "review_exposure",
        "review_exposure",
        "review_exposure",
        "retained",
        "retained",
    ]


def test_unknown_review_key_raises():
    with pytest.raises(ValueError, match="record_key"):
        reasons(records_frame((2005, "target", "DD 1", "mur")), ["missing"])


@pytest.fixture(scope="module")
def review_keys():
    return pd.read_csv(REVIEW_PATH, dtype="string")["record_key"]


@pytest.fixture(scope="module")
def splits(records, review_keys):
    return assign_partitions(records, review_keys)


def test_exclusion_counts_in_the_snapshot(splits):
    counts = {
        partition: values.value_counts().to_dict()
        for partition, values in splits.groupby("partition")["exclusion_reason"]
    }
    assert counts == {
        "train_original": {"retained": 30_782, "dossier_overlap": 207},
        "validation": {
            "retained": 2_794,
            "text_overlap": 2_249,
            "dossier_overlap": 347,
            "review_exposure": 8,
        },
        "test_historical": {
            "retained": 1_320,
            "text_overlap": 978,
            "dossier_overlap": 313,
            "review_exposure": 6,
        },
        "test_recent": {
            "retained": 6_671,
            "dossier_overlap": 2_337,
            "text_overlap": 1_723,
            "review_exposure": 11,
        },
        "pool_unlabelled": {"retained": 58_482, "dossier_overlap": 2_359},
    }


def test_stored_split_matches_the_code(splits):
    # after a deliberate change: `uv run permits data splits`, then review the diff
    stored = pd.read_csv(SPLITS_PATH, dtype="string")
    pd.testing.assert_frame_equal(stored, split_table(splits))


def test_load_splits_returns_retained_rows_only(records):
    retained = load_splits()
    assert retained["exclusion_reason"].eq("retained").all()
    assert len(retained) == 30_782 + 2_794 + 1_320 + 6_671 + 58_482


def test_every_record_outside_the_partitions_has_a_reason(records, splits):
    outside = records.loc[~records["record_key"].isin(splits["record_key"])]
    assert outside["record_status"].value_counts().to_dict() == {
        "no_description": 68_418,
        "unlabelled": 18_181,
        "am": 38,
    }
    pool_start = load_config()["data"]["splits"]["pool_unlabelled"]["from"]
    assert outside.loc[outside["record_status"].eq("unlabelled"), "year"].lt(pool_start).all()


def test_retained_rows_share_nothing_across_partitions(splits, review_keys):
    retained = splits.loc[splits["exclusion_reason"].eq("retained")]
    reviewed = splits.loc[splits["record_key"].isin(review_keys)]
    for name in splits["partition"].unique():
        rows = retained.loc[retained["partition"].eq(name)]
        others = splits.loc[splits["partition"].ne(name)]  # every candidate, excluded rows too
        assert set(rows["base_dossier"]).isdisjoint(others["base_dossier"]), name
        if name in EVALUATION:
            assert set(rows["text_key"]).isdisjoint(others["text_key"]), name
            assert not rows["base_dossier"].isin(reviewed["base_dossier"]).any(), name
            assert not rows["text_key"].isin(reviewed["text_key"]).any(), name
