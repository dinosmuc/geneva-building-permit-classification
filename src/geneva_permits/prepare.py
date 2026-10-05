"""The record table: one row per source record, with a stable key, status, label and text key."""

import re
import unicodedata

import pandas as pd

from geneva_permits.config import load_config
from geneva_permits.source import load_source

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
AM_CODE = "AM"


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


def record_status(raw, target_classes):
    """no_description, else target, am or unlabelled by operation code; other codes raise."""
    code = raw["TYPE_OPERATION"]
    status = pd.Series(pd.NA, index=raw.index, dtype="string")
    status.loc[is_missing(code, "TYPE_OPERATION")] = "unlabelled"
    status.loc[code.eq(AM_CODE).fillna(False)] = "am"
    status.loc[code.isin(target_classes)] = "target"
    status.loc[is_missing(raw["DESCRIPTION"], "DESCRIPTION")] = "no_description"
    if status.isna().any():
        raise ValueError(f"unknown operation codes: {sorted(code[status.isna()].unique())}")
    return status


def prepare_records(raw):
    """SITG columns without OPERATION, which restates the label, plus six derived columns."""
    records = raw.drop(columns="OPERATION")
    records["record_key"] = records["ID_DOSSIER"] + "|" + records["E"] + "|" + records["N"]
    if records["record_key"].isna().any() or records["record_key"].duplicated().any():
        raise ValueError("record_key must be present and unique")
    records["base_dossier"] = records["TYPE_DOSSIER"] + " " + records["NO_DOSSIER"]
    records["year"] = records["DATE_DEPOT"].dt.year.astype("Int64")
    records["record_status"] = record_status(records, load_config()["data"]["target_classes"])
    records["label"] = records["TYPE_OPERATION"].where(records["record_status"].eq("target"))
    described = records["DESCRIPTION"].where(records["record_status"].ne("no_description"))
    records["text_key"] = described.map(normalize_description, na_action="ignore").astype("string")
    return records


def load_records():
    """The record table of the pinned snapshot, rebuilt in memory on every call."""
    return prepare_records(load_source())


def operation_definitions(raw):
    """Each operation code's OPERATION definition, by its most frequent spelling."""
    pairs = raw.loc[
        ~is_missing(raw["TYPE_OPERATION"], "TYPE_OPERATION")
        & ~is_missing(raw["OPERATION"], "OPERATION"),
        ["TYPE_OPERATION", "OPERATION"],
    ]
    definitions = {}
    for code, spellings in pairs.groupby("TYPE_OPERATION")["OPERATION"]:
        counts = spellings.value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            raise ValueError(f"{code}: equally frequent spellings {list(counts.index[:2])}")
        definitions[code] = counts.index[0]
    return definitions
