"""Snapshot fetch and verification, against a mocked HTTP layer (no network)."""

import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime

import httpx
import pytest

from geneva_permits.source import (
    LEGACY_NAME,
    LIVE_DIR_NAME,
    LIVE_MANIFEST_NAME,
    LOCK_NAME,
    SnapshotError,
    fetch_live_snapshot,
    fetch_snapshot,
    load_source,
    previous_friday,
    read_date_info,
    read_lock,
    verify_snapshot,
)

MIRROR_URL = "https://mirror.test/snapshot_2026-09-27.zip"
LIVE_URL = "https://live.test/SIT_AUTOR_DOSSIER-CSV.zip"
DATE_INFO = (
    "\nInformations sur les données Open Data du SITG que vous avez téléchargées:\n"
    "Date de création du fichier .zip téléchargé : 27.09.2026 08:27:23.0882805\n\n"
    "Les données contenues dans le .zip ont \nété extraites de la géodatabase du SITG "
    "le vendredi soir précédent le 27.09.2026\n"
)


def make_archive(csv="OBJECTID;DATE_DEPOT;DATE_MAJ_2\n1;20020226;20020226\n"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SIT_AUTOR_DOSSIER.csv", csv)
        archive.writestr("DOC/Informations_date.txt", DATE_INFO)
    return buffer.getvalue()


PINNED = make_archive()
NEWER = make_archive(csv="OBJECTID;DATE_DEPOT;DATE_MAJ_2\n2;20261003;20261003\n")


@pytest.fixture
def raw_dir(tmp_path):
    lock = {
        "file": "snapshot_2026-09-27.zip",
        "bytes": len(PINNED),
        "sha256": hashlib.sha256(PINNED).hexdigest(),
        "mirror_url": MIRROR_URL,
        "live_url": LIVE_URL,
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
    """Name -> (bytes, mtime_ns, inode) for every file below directory."""
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_existing_valid_file_is_kept(raw_dir):
    pinned = raw_dir / read_lock(raw_dir)["file"]
    pinned.write_bytes(PINNED)
    client, requested = serving(NEWER)
    before = snapshot_of(raw_dir)

    assert fetch_snapshot(raw_dir, client=client) == (pinned, "kept")
    assert requested == []
    assert snapshot_of(raw_dir) == before


def test_legacy_filename_is_adopted(raw_dir):
    (raw_dir / LEGACY_NAME).write_bytes(PINNED)
    client, requested = serving(NEWER)

    path, action = fetch_snapshot(raw_dir, client=client)

    assert action == "adopted"
    assert path.read_bytes() == PINNED
    assert not (raw_dir / LEGACY_NAME).exists()
    assert requested == []


def test_legacy_file_with_other_content_is_not_adopted(raw_dir):
    (raw_dir / LEGACY_NAME).write_bytes(NEWER)
    client, _ = serving(PINNED)

    assert fetch_snapshot(raw_dir, client=client)[1] == "downloaded"
    assert (raw_dir / LEGACY_NAME).read_bytes() == NEWER


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


def test_valid_pinned_file_is_never_overwritten(raw_dir):
    lock = read_lock(raw_dir)
    pinned = raw_dir / lock["file"]
    pinned.write_bytes(PINNED)
    (raw_dir / LEGACY_NAME).write_bytes(PINNED)
    client, requested = serving(NEWER)
    before = snapshot_of(raw_dir)

    for _ in range(2):
        assert fetch_snapshot(raw_dir, client=client) == (pinned, "kept")

    assert snapshot_of(raw_dir) == before
    assert requested == []


def test_live_fetch_never_touches_the_pin_or_the_lock(raw_dir):
    (raw_dir / read_lock(raw_dir)["file"]).write_bytes(PINNED)
    client, requested = serving(NEWER)
    before = snapshot_of(raw_dir)
    now = datetime(2026, 10, 4, 9, 30, tzinfo=UTC)

    archive, manifest = fetch_live_snapshot(raw_dir, client=client, now=now)

    after = snapshot_of(raw_dir)
    assert {name: after[name] for name in before} == before
    live = raw_dir / LIVE_DIR_NAME / "2026-10-04"
    assert set(after) - set(before) == {
        f"{LIVE_DIR_NAME}/2026-10-04/SIT_AUTOR_DOSSIER-CSV_2026-10-04.zip",
        f"{LIVE_DIR_NAME}/2026-10-04/{LIVE_MANIFEST_NAME}",
    }
    assert archive.parent == live and archive.read_bytes() == NEWER
    assert requested == [LIVE_URL]
    assert manifest["matches_pin"] is False
    assert json.loads((live / LIVE_MANIFEST_NAME).read_text(encoding="utf-8")) == manifest


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


def test_date_info_follows_sitg_wording(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(PINNED)
    assert read_date_info(archive) == {
        "sitg_zip_created": "2026-09-27T08:27:23",
        "sitg_extracted": "2026-09-25",  # 27.09.2026 is a Sunday
        "sitg_extraction_note": (
            "extraites de la géodatabase du SITG le vendredi soir précédent le 27.09.2026"
        ),
    }


@pytest.mark.parametrize(
    ("day", "friday"),
    [(date(2026, 9, 27), date(2026, 9, 25)), (date(2026, 9, 25), date(2026, 9, 18))],
)
def test_previous_friday_is_strictly_earlier(day, friday):
    assert previous_friday(day) == friday
