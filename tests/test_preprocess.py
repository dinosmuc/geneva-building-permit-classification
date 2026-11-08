"""Lemmatized texts: what the pipeline keeps, and the stored file's provenance check."""

import pandas as pd
import pytest
import spacy

from geneva_permits import preprocess
from geneva_permits.config import load_config
from geneva_permits.preprocess import (
    build_texts,
    lemmatize,
    load_nlp,
    load_texts,
    provenance,
    split_settings_sha256,
    versions,
    write_texts,
)
from geneva_permits.source import read_lock, sha256_of


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def test_lemmas_drop_stop_words_punctuation_and_inflection(nlp):
    texts = ["construction d'une villa avec piscine", "remplacement des fenêtres."]
    assert lemmatize(texts, nlp) == ["construction villa piscine", "remplacement fenêtre"]


def test_excluded_components_leave_the_lemmas_unchanged(nlp, records):
    # a sample for speed; all 30,782 training texts matched when checked once
    texts = records.loc[records["record_status"].eq("target"), "DESCRIPTION"].head(1_000)
    full = spacy.load(load_config()["preprocessing"]["spacy_model"])
    assert lemmatize(texts, nlp) == lemmatize(texts, full)


def test_build_texts_skips_the_pool_and_records_versions(nlp):
    splits = pd.DataFrame(
        {
            "record_key": ["a", "b"],
            "partition": ["train_original", "pool_unlabelled"],
            "label": ["VIL", None],
            "DESCRIPTION": ["construction d'une villa", "pose d'une tente"],
        }
    )
    texts = build_texts(splits, nlp)
    assert texts.to_dict("records") == [
        {
            "record_key": "a",
            "partition": "train_original",
            "label": "VIL",
            "text": "construction d'une villa",
            "lemmas": "construction villa",
        }
    ]
    assert texts.attrs == versions()


@pytest.fixture
def stored_split(tmp_path, monkeypatch):
    """A stored split of two records in place of data/splits/splits.csv."""
    path = tmp_path / "splits.csv"
    path.write_text("record_key,partition\na,train_original\nb,pool_unlabelled\n")
    monkeypatch.setattr(preprocess, "SPLITS_PATH", path)
    return path


SPLIT_RECORDS = pd.DataFrame(
    {
        "record_key": ["a", "b"],
        "partition": ["train_original", "pool_unlabelled"],
        "label": ["VIL", None],
        "DESCRIPTION": ["construction d'une villa", "pose d'une tente"],
        "exclusion_reason": ["retained", "retained"],
    }
)


def test_load_texts_refuses_other_versions(tmp_path):
    path = tmp_path / "texts.parquet"
    texts = pd.DataFrame({"lemmas": ["villa"]})
    texts.attrs = provenance()
    texts.to_parquet(path)
    pd.testing.assert_frame_equal(load_texts(path), texts)

    texts.attrs = provenance() | {"spacy": "0.0"}
    texts.to_parquet(path)
    with pytest.raises(ValueError, match="other spacy; rerun"):
        load_texts(path)


@pytest.mark.parametrize("missing", ["snapshot_sha256", "splits_sha256", "split_settings_sha256"])
def test_load_texts_refuses_texts_without_their_data_provenance(tmp_path, missing):
    path = tmp_path / "texts.parquet"
    texts = pd.DataFrame({"lemmas": ["villa"]})
    texts.attrs = {k: v for k, v in provenance().items() if k != missing}
    texts.to_parquet(path)
    with pytest.raises(ValueError, match=f"other {missing}; rerun"):
        load_texts(path)


@pytest.mark.parametrize(
    "change", ["splits", "dedup", "target_classes", "excluded_classes", "review file"]
)
def test_load_texts_refuses_texts_made_under_other_split_settings(tmp_path, monkeypatch, change):
    path = tmp_path / "texts.parquet"
    texts = pd.DataFrame({"lemmas": ["villa"]})
    texts.attrs = provenance()
    texts.to_parquet(path)
    if change == "review file":
        review = tmp_path / "exploratory_review.csv"
        review.write_text("record_key\n")
        monkeypatch.setattr(preprocess, "REVIEW_PATH", review)
    else:
        config = load_config()
        changed = config | {"data": config["data"] | {change: "changed"}}
        monkeypatch.setattr(preprocess, "load_config", lambda: changed)
    with pytest.raises(ValueError, match="other split_settings_sha256; rerun"):
        load_texts(path)


def test_write_texts_records_the_snapshot_and_split_and_refuses_a_changed_split(
    tmp_path, monkeypatch, nlp, stored_split
):
    monkeypatch.setattr(preprocess, "load_splits", lambda: SPLIT_RECORDS)
    monkeypatch.setattr(preprocess, "load_nlp", lambda: nlp)
    path = tmp_path / "texts.parquet"
    write_texts(path)
    texts = load_texts(path)
    assert texts["record_key"].tolist() == ["a"]
    assert texts.attrs["splits_sha256"] == sha256_of(stored_split)
    assert texts.attrs["snapshot_sha256"] == read_lock()["sha256"]
    assert texts.attrs["split_settings_sha256"] == split_settings_sha256()

    stored_split.write_text("record_key,partition\na,validation\nb,pool_unlabelled\n")
    with pytest.raises(ValueError, match="other splits_sha256"):
        load_texts(path)  # the texts no longer match the stored split
    with pytest.raises(ValueError, match="computed split differs"):
        write_texts(path)  # nor does the split the code computes
