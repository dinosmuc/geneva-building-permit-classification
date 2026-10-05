"""Regression checks for field-level cleaning."""

import pandas as pd

from geneva_permits.cleaning import is_missing, normalize_description


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
