"""The record table, on tiny synthetic frames and on the pinned snapshot."""

import os

import pandas as pd
import pytest

from geneva_permits.config import RAW_DIR
from geneva_permits.prepare import (
    MISSING_DESCRIPTION,
    MISSING_OPERATION_CODE,
    MISSING_OPERATION_LABEL,
    is_missing,
    normalize_description,
    operation_definitions,
    prepare_records,
)
from geneva_permits.source import load_source, read_lock


def source_frame(*rows):
    """A raw-like frame; each row gives (TYPE_OPERATION, DESCRIPTION), keys are made unique."""
    return pd.DataFrame(
        {
            "OBJECTID": [str(index) for index in range(len(rows))],
            "ID_DOSSIER": [f"DD {index}/1" for index in range(len(rows))],
            "TYPE_DOSSIER": "DD",
            "NO_DOSSIER": [str(index) for index in range(len(rows))],
            "TYPE_OPERATION": [code for code, _ in rows],
            "DESCRIPTION": [description for _, description in rows],
            "OPERATION": "définition",
            "DATE_DEPOT": pd.Timestamp("2014-03-01"),
            "E": "2500000.00",
            "N": "1117000.00",
        },
        dtype="string",
    ).astype({"DATE_DEPOT": "datetime64[us]"})


def test_record_status_takes_one_value_per_row():
    raw = source_frame(
        ("REN", "rénovation d'un appartement"),
        ("AM", "réaménagement rues"),
        (MISSING_OPERATION_CODE, "piscine"),
        (None, "véranda"),
        ("REN", MISSING_DESCRIPTION),
        (MISSING_OPERATION_CODE, None),
    )
    status = prepare_records(raw)["record_status"].tolist()
    assert status == [
        "target",
        "am",
        "unlabelled",
        "unlabelled",
        "no_description",
        "no_description",
    ]


def test_unknown_operation_code_raises():
    with pytest.raises(ValueError, match="XYZ"):
        prepare_records(source_frame(("XYZ", "mur")))


def test_placeholders_count_as_missing():
    status = pd.Series(["-", pd.NA, "TERMINE"], dtype="string")
    assert is_missing(status, "STATUT").tolist() == [True, True, False]
    assert is_missing(pd.Series([MISSING_DESCRIPTION, "mur"]), "DESCRIPTION").tolist() == [
        True,
        False,
    ]


def test_label_and_text_key_are_null_outside_their_rows():
    records = prepare_records(
        source_frame(("PI", "Piscine -"), (MISSING_OPERATION_CODE, "mur"), ("PI", None))
    )
    assert records["label"].tolist() == ["PI", pd.NA, pd.NA]
    assert records["text_key"].tolist() == ["piscine", "mur", pd.NA]


def test_derived_columns_replace_operation():
    records = prepare_records(source_frame(("PI", "piscine")))
    assert "OPERATION" not in records
    assert records.loc[0, "record_key"] == "DD 0/1|2500000.00|1117000.00"
    assert records.loc[0, "base_dossier"] == "DD 0"
    assert records.loc[0, "year"] == 2014


def test_duplicate_record_key_raises():
    raw = source_frame(("PI", "piscine"), ("PI", "piscine"))
    raw["ID_DOSSIER"] = "DD 1/1"
    with pytest.raises(ValueError, match="record_key"):
        prepare_records(raw)


def test_normalize_description_merges_formatting_variants():
    variants = [
        "Rénovation d’un appartement ",
        "rénovation d'un appartement",
        "RENOVATION D'UN APPARTEMENT -",
    ]
    assert {normalize_description(text) for text in variants} == {"renovation d un appartement"}
    assert normalize_description("au 2ème étage") != normalize_description("au 3ème étage")


def definitions_frame(pairs):
    return pd.DataFrame(pairs, columns=["TYPE_OPERATION", "OPERATION"], dtype="string")


def test_operation_definitions_pick_the_majority_spelling():
    raw = definitions_frame(
        [
            ("EQU", "travaux (publics"),  # broken spelling, first and equally long
            ("EQU", "travaux (public)"),
            ("EQU", "travaux (public)"),
            (MISSING_OPERATION_CODE, MISSING_OPERATION_LABEL),
            (None, MISSING_OPERATION_LABEL),
            ("PI", MISSING_OPERATION_LABEL),
        ]
    )
    assert operation_definitions(raw) == {"EQU": "travaux (public)"}


def test_operation_definitions_raise_on_a_tie():
    raw = definitions_frame([("EQU", "travaux (publics"), ("EQU", "travaux (public)")])
    with pytest.raises(ValueError, match="EQU"):
        operation_definitions(raw)


@pytest.fixture(scope="module")
def raw():
    """The pinned snapshot; skipped locally when not fetched, a failure in CI."""
    if not (RAW_DIR / read_lock()["file"]).is_file():
        if os.environ.get("CI"):
            pytest.fail("pinned snapshot missing; CI must run `uv run permits data fetch` first")
        pytest.skip("pinned snapshot not fetched; run `uv run permits data fetch`")
    return load_source()


@pytest.fixture(scope="module")
def records(raw):
    return prepare_records(raw)


def test_record_status_counts_in_the_snapshot(records):
    assert records["record_status"].value_counts().to_dict() == {
        "unlabelled": 79_022,
        "no_description": 68_418,
        "target": 49_746,
        "am": 38,
    }
    assert len(records) == 197_224


def test_record_key_is_unique_in_the_snapshot(records):
    assert records["record_key"].is_unique


def test_operation_definitions_in_the_snapshot(raw):
    definitions = operation_definitions(raw)
    assert len(definitions) == 9
    assert definitions["EQU"].endswith(")")
