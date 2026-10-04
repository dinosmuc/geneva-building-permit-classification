"""Regression checks for the source loader."""

import pandas as pd

from src.data import is_missing, normalize_description, parse_dates


def test_parse_dates_handles_both_source_formats():
    raw = pd.Series(["20020226", "20211007105640", pd.NA], dtype="string")
    parsed = parse_dates(raw)
    assert parsed[0] == pd.Timestamp("2002-02-26")
    assert parsed[1] == pd.Timestamp("2021-10-07")  # time of day dropped
    assert pd.isna(parsed[2])


def test_normalize_description_merges_formatting_variants():
    variants = [
        "Rénovation d’un appartement ",
        "rénovation d'un appartement",
        "RENOVATION D'UN APPARTEMENT -",
    ]
    assert {normalize_description(text) for text in variants} == {"renovation d un appartement"}
    assert normalize_description("au 2ème étage") != normalize_description("au 3ème étage")


def test_is_missing_counts_nulls_and_field_placeholder():
    status = pd.Series(["-", pd.NA, "TERMINE"], dtype="string")
    assert is_missing(status, "STATUT").tolist() == [True, True, False]
