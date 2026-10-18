"""Lemmatized texts: what the pipeline keeps, and the stored file's version check."""

import pandas as pd
import pytest
import spacy

from geneva_permits.config import load_config
from geneva_permits.preprocess import build_texts, lemmatize, load_nlp, load_texts, versions


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


def test_load_texts_refuses_other_versions(tmp_path):
    path = tmp_path / "texts.parquet"
    texts = pd.DataFrame({"lemmas": ["villa"]})
    texts.attrs = versions()
    texts.to_parquet(path)
    pd.testing.assert_frame_equal(load_texts(path), texts)

    texts.attrs = versions() | {"spacy": "0.0"}
    texts.to_parquet(path)
    with pytest.raises(ValueError, match="rerun"):
        load_texts(path)
