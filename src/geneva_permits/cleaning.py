"""Field-level cleaning: placeholders, the text grouping key and operation labels."""

import re
import unicodedata

import pandas as pd

# Sentinels the export uses instead of empty cells, one per field.
MISSING_OPERATION_CODE = "--"
MISSING_DESCRIPTION = "non renseigné"
MISSING_STATUS = "-"
MISSING_OPERATION_LABEL = "Non renseigné (valeur par défaut à l'importation des données)"
PLACEHOLDERS = {
    "TYPE_OPERATION": MISSING_OPERATION_CODE,
    "DESCRIPTION": MISSING_DESCRIPTION,
    "STATUT": MISSING_STATUS,
    "OPERATION": MISSING_OPERATION_LABEL,
}


def is_missing(values, column):
    """True where a field is null or holds that field's placeholder."""
    missing = values.isna()
    if column in PLACEHOLDERS:
        missing |= values.eq(PLACEHOLDERS[column]).fillna(False)
    return missing


def normalize_description(text):
    """Grouping key: casefold, strip accents, collapse punctuation and whitespace.

    Used only for repetition and overlap checks; the model reads the original text.
    """
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[\W_]+", " ", text).strip()


def operation_labels(frame):
    """Official definition of each operation code, taken from the export itself.

    TYPE_OPERATION holds the code and OPERATION its French definition. A code with
    several label spellings keeps the longest, which is the uncorrupted one.
    """
    labels = {}
    for code, label in zip(frame["TYPE_OPERATION"], frame["OPERATION"], strict=True):
        if pd.isna(code) or pd.isna(label) or code == MISSING_OPERATION_CODE:
            continue
        if len(label) > len(labels.get(code, "")):
            labels[code] = label
    return dict(sorted(labels.items()))
