"""Lemmatized descriptions for the classical models, computed once and stored."""

from importlib.metadata import version

import pandas as pd
import spacy

from geneva_permits.config import PROCESSED_DIR, load_config
from geneva_permits.splits import POOL, load_splits

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


def build_texts(splits, nlp):
    """Original and lemmatized text of the retained records outside the unlabelled pool."""
    columns = ["record_key", "partition", "label", "DESCRIPTION"]
    texts = splits.loc[splits["partition"].ne(POOL), columns].reset_index(drop=True)
    texts = texts.rename(columns={"DESCRIPTION": "text"})
    texts["lemmas"] = lemmatize(texts["text"], nlp)
    texts.attrs = versions()
    return texts


def write_texts(path=PROCESSED_PATH):
    """Lemmatize the current split and store it with the versions used."""
    texts = build_texts(load_splits(), load_nlp())
    path.parent.mkdir(parents=True, exist_ok=True)
    texts.to_parquet(path, index=False)
    return path, len(texts)


def load_texts(path=PROCESSED_PATH):
    """The stored texts; raises if spaCy or the model changed since they were written."""
    texts = pd.read_parquet(path)
    if texts.attrs != versions():
        raise ValueError(f"{path} was made with {texts.attrs}; rerun `permits data preprocess`")
    return texts
