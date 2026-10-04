"""Integrity gates. Split and annotation checks are added as the pipeline lands."""

from geneva_permits.config import load_config


def test_config_parses():
    assert load_config()


def test_eight_target_classes():
    data = load_config()["data"]
    classes = data["target_classes"]
    assert len(classes) == len(set(classes)) == 8
    assert "AM" in data["excluded_classes"]


def test_partitions_do_not_overlap_in_time():
    splits = load_config()["data"]["splits"]
    labelled = ["train_original", "validation", "test_historical", "test_recent"]
    spans = sorted((splits[k].get("from", 0), splits[k].get("to", 9999)) for k in labelled)
    for (_, end), (start, _) in zip(spans[:-1], spans[1:], strict=True):
        assert end < start, f"partitions overlap: {spans}"


def test_text_disjoint_covers_every_training_source():
    # a P description reaching the test subset would fake a distillation gain
    against = set(load_config()["evaluation"]["text_disjoint_against"])
    assert {"train_original", "pool_accepted", "validation"} <= against
