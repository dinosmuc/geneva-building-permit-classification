"""Snapshot fetch, verification and reading, against a mocked HTTP layer (no network)."""

import hashlib
import io
import json
import zipfile
from datetime import datetime

import httpx
import pandas as pd
import pytest

from geneva_permits.source import (
    LOCK_NAME,
    SnapshotError,
    fetch_snapshot,
    load_source,
    parse_dates,
    read_lock,
    verify_snapshot,
)

MIRROR_URL = "https://mirror.test/snapshot_2026-09-27.zip"


def make_archive(csv):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SIT_AUTOR_DOSSIER.csv", csv)
    return buffer.getvalue()


PINNED = make_archive("OBJECTID;DATE_DEPOT;DATE_MAJ_2\n1;20020226;20020226\n")
NEWER = make_archive("OBJECTID;DATE_DEPOT;DATE_MAJ_2\n2;20261003;20261003\n")


@pytest.fixture
def raw_dir(tmp_path):
    lock = {
        "file": "snapshot_2026-09-27.zip",
        "bytes": len(PINNED),
        "sha256": hashlib.sha256(PINNED).hexdigest(),
        "mirror_url": MIRROR_URL,
    }
    (tmp_path / LOCK_NAME).write_text(json.dumps(lock), encoding="utf-8")
    return tmp_path


def serving(content, status=200):
    """A client whose every response is content, recording the URLs requested."""
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler)), requested


def snapshot_of(directory):
    """Name -> (bytes, mtime_ns, inode) for every file in directory."""
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
        for path in sorted(directory.iterdir())
    }


def test_valid_pinned_file_is_kept_and_never_overwritten(raw_dir):
    pinned = raw_dir / read_lock(raw_dir)["file"]
    pinned.write_bytes(PINNED)
    client, requested = serving(NEWER)
    before = snapshot_of(raw_dir)

    for _ in range(2):
        assert fetch_snapshot(raw_dir, client=client) == (pinned, "kept")

    assert snapshot_of(raw_dir) == before
    assert requested == []


def test_successful_download_is_verified_and_moved_into_place(raw_dir):
    client, requested = serving(PINNED)

    path, action = fetch_snapshot(raw_dir, client=client)

    assert action == "downloaded"
    assert requested == [MIRROR_URL]
    assert verify_snapshot(raw_dir) == path
    assert sorted(p.name for p in raw_dir.iterdir()) == sorted([LOCK_NAME, path.name])


@pytest.mark.parametrize(
    ("content", "status"),
    [
        (PINNED[:-1] + bytes([PINNED[-1] ^ 1]), 200),  # same size, one bit flipped
        (PINNED + b"x", 200),
        (b"Not Found", 404),
    ],
    ids=["bit-flip", "extra-byte", "http-404"],
)
def test_failed_download_leaves_nothing_behind(raw_dir, content, status):
    client, _ = serving(content, status)
    before = snapshot_of(raw_dir)

    with pytest.raises(SnapshotError):
        fetch_snapshot(raw_dir, client=client)

    assert snapshot_of(raw_dir) == before


def test_verify_rejects_missing_and_altered_archives(raw_dir):
    pinned = raw_dir / read_lock(raw_dir)["file"]
    with pytest.raises(SnapshotError, match="missing"):
        verify_snapshot(raw_dir)

    pinned.write_bytes(NEWER)
    with pytest.raises(SnapshotError, match="does not match"):
        verify_snapshot(raw_dir)


def test_load_source_refuses_an_unverified_archive(raw_dir):
    (raw_dir / read_lock(raw_dir)["file"]).write_bytes(NEWER)
    with pytest.raises(SnapshotError):
        load_source(raw_dir)


def test_load_source_reads_a_verified_archive(raw_dir):
    (raw_dir / read_lock(raw_dir)["file"]).write_bytes(PINNED)
    frame = load_source(raw_dir)
    assert frame["OBJECTID"].tolist() == ["1"]
    assert frame["DATE_DEPOT"].iloc[0] == datetime(2002, 2, 26)


def test_parse_dates_handles_both_source_formats():
    raw = pd.Series(["20020226", "20211007105640", pd.NA], dtype="string")
    parsed = parse_dates(raw)
    assert parsed[0] == pd.Timestamp("2002-02-26")
    assert parsed[1] == pd.Timestamp("2021-10-07")  # time of day dropped
    assert pd.isna(parsed[2])
