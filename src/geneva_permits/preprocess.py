"""Lemmatized descriptions for the classical models, computed once and stored."""

import hashlib
import json
from importlib.metadata import version

import pandas as pd
import spacy

from geneva_permits.config import PROCESSED_DIR, load_config
from geneva_permits.source import read_lock, sha256_of
from geneva_permits.splits import (
    POOL,
    REVIEW_PATH,
    SPLITS_PATH,
    TRAIN,
    VALIDATION,
    load_splits,
    split_table,
)

PROCESSED_PATH = PROCESSED_DIR / "texts.parquet"
UNUSED_PIPES = ["parser", "ner"]  # lemmas are identical without them


def load_nlp():
    """The pinned French pipeline without the components lemmas do not need."""
    return spacy.load(load_config()["preprocessing"]["spacy_model"], exclude=UNUSED_PIPES)


def is_kept(token):
    """Content tokens: not a stop word, punctuation or whitespace."""
    return not (token.is_stop or token.is_punct or token.is_space)


def lemmatize(texts, nlp):
    """Each text as the space-joined lemmas of its kept tokens."""
    return [" ".join(t.lemma_ for t in doc if is_kept(t)) for doc in nlp.pipe(texts)]


def versions():
    """The spaCy and model versions the texts are made with."""
    model = load_config()["preprocessing"]["spacy_model"]
    return {"spacy": spacy.__version__, model: version(model)}


def split_settings_sha256():
    """SHA-256 of what the split follows besides the snapshot: data.splits, dedup, the target and
    excluded classes, and the reviewed records."""
    data = load_config()["data"]
    settings = {
        name: data[name] for name in ["splits", "dedup", "target_classes", "excluded_classes"]
    }
    settings["review_sha256"] = sha256_of(REVIEW_PATH)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def data_provenance():
    """SHA-256 of the pinned snapshot, of the stored split and of the split settings."""
    return {
        "snapshot_sha256": read_lock()["sha256"],
        "splits_sha256": sha256_of(SPLITS_PATH),
        "split_settings_sha256": split_settings_sha256(),
    }


def provenance():
    """What the stored texts are made with: the spaCy versions, the snapshot, the split and its
    settings."""
    return versions() | data_provenance()


def build_texts(splits, nlp):
    """Original and lemmatized text of the retained records outside the unlabelled pool."""
    columns = ["record_key", "partition", "label", "DESCRIPTION"]
    texts = splits.loc[splits["partition"].ne(POOL), columns].reset_index(drop=True)
    texts = texts.rename(columns={"DESCRIPTION": "text"})
    texts["lemmas"] = lemmatize(texts["text"], nlp)
    texts.attrs = versions()
    return texts


def write_texts(path=PROCESSED_PATH):
    """Lemmatize the current split and store it with its provenance.

    Refuses when the split the code computes differs from the stored one, whose hash it records.
    """
    splits = load_splits()
    stored = pd.read_csv(SPLITS_PATH, dtype="string")
    if not split_table(splits).astype("string").equals(stored):
        raise ValueError(
            f"the computed split differs from {SPLITS_PATH}; after a deliberate change, run "
            "`permits data splits` and review the diff"
        )
    texts = build_texts(splits, load_nlp())
    texts.attrs |= data_provenance()
    path.parent.mkdir(parents=True, exist_ok=True)
    texts.to_parquet(path, index=False)
    return path, len(texts)


def load_texts(path=PROCESSED_PATH):
    """The stored texts; raises if spaCy, its model, the snapshot, the split or its settings
    changed since."""
    texts = pd.read_parquet(path)
    expected = provenance()
    changed = sorted(
        k for k in expected.keys() | texts.attrs.keys() if texts.attrs.get(k) != expected.get(k)
    )
    if changed:
        raise ValueError(
            f"{path} was made with other {', '.join(changed)}; rerun `permits data preprocess`"
        )
    return texts


def train_validation(texts):
    """The training and validation rows."""
    return (texts[texts["partition"].eq(p)] for p in [TRAIN, VALIDATION])
