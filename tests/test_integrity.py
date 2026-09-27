"""Integrity gates. Split and annotation checks are added as the pipeline lands."""

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_parses():
    assert yaml.safe_load(CONFIG.read_text())


def test_eight_target_classes():
    cfg = yaml.safe_load(CONFIG.read_text())
    classes = cfg["data"]["target_classes"]
    assert len(classes) == len(set(classes)) == 8
    assert "AM" in cfg["data"]["excluded_classes"]


def test_partitions_do_not_overlap_in_time():
    splits = yaml.safe_load(CONFIG.read_text())["data"]["splits"]
    labelled = ["train_original", "validation", "test_historical", "test_recent"]
    spans = sorted((splits[k].get("from", 0), splits[k].get("to", 9999)) for k in labelled)
    for (_, end), (start, _) in zip(spans[:-1], spans[1:], strict=True):
        assert end < start, f"partitions overlap: {spans}"


def test_text_disjoint_covers_every_training_source():
    # a P description reaching the test subset would fake a distillation gain
    against = set(yaml.safe_load(CONFIG.read_text())["evaluation"]["text_disjoint_against"])
    assert {"train_original", "pool_accepted", "validation"} <= against
