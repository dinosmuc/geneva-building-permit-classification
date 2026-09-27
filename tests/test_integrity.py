"""Integrity gates for the experiment.

Planned checks, added as the pipeline lands:
  - no dossier group spans two partitions
  - the text-disjoint subset shares no folded description with O, accepted P or validation
  - accepted annotations validate against the output schema
  - per-source exposure counts match across the three enriched conditions
"""

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
    """Labelled partitions must occupy disjoint filing periods."""
    splits = yaml.safe_load(CONFIG.read_text())["data"]["splits"]
    labelled = ["train_original", "validation", "test_historical", "test_recent"]
    spans = [(splits[k].get("from", 0), splits[k].get("to", 9999)) for k in labelled]
    spans.sort()
    for (_, end), (start, _) in zip(spans[:-1], spans[1:], strict=True):
        assert end < start, f"partitions overlap: {spans}"


def test_text_disjoint_covers_every_training_source():
    """A P description leaking into the test subset would fake a distillation gain."""
    cfg = yaml.safe_load(CONFIG.read_text())
    against = set(cfg["evaluation"]["text_disjoint_against"])
    assert {"train_original", "pool_accepted", "validation"} <= against
