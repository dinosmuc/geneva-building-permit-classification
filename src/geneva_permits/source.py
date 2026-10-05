"""The pinned SITG snapshot: fetch, verify and read.

`source_manifest.json` is a lock file: code reads it and never writes it. Changing the pin is a
manual, reviewed edit.
"""

import contextlib
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from geneva_permits.paths import RAW_DIR

LOCK_NAME = "source_manifest.json"
CSV_MEMBER = "SIT_AUTOR_DOSSIER.csv"
CSV_SEPARATOR = ";"
DATE_COLUMNS = ["DATE_DEPOT", "DATE_MAJ_2"]
DATE_FORMAT = "%Y%m%d"  # DATE_DEPOT is stored as 20020226
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"  # three filing dates carry a time of day

# Tried in order; the first one that decodes is used.
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

BLOCK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 120.0


class SnapshotError(RuntimeError):
    """The archive is missing, failed to download, or does not match the lock."""


def sha256_of(path):
    """Hash a file in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def read_lock(raw_dir=RAW_DIR):
    """The tracked lock file describing the pinned snapshot."""
    return json.loads((Path(raw_dir) / LOCK_NAME).read_text(encoding="utf-8"))


def matches_lock(path, lock):
    """True if path is a file with the locked size and SHA-256."""
    path = Path(path)
    return (
        path.is_file()
        and path.stat().st_size == lock["bytes"]
        and sha256_of(path) == lock["sha256"]
    )


def verify_snapshot(raw_dir=RAW_DIR, lock=None):
    """Return the pinned archive's path, or raise SnapshotError if it is missing or altered."""
    lock = lock or read_lock(raw_dir)
    archive = Path(raw_dir) / lock["file"]
    if not archive.is_file():
        raise SnapshotError(f"{archive} is missing; run `uv run permits data fetch`.")
    if not matches_lock(archive, lock):
        raise SnapshotError(
            f"{archive} does not match {LOCK_NAME} "
            f"(expected {lock['bytes']} bytes, SHA-256 {lock['sha256']})."
        )
    return archive


def fetch_snapshot(raw_dir=RAW_DIR, lock=None, client=None):
    """Keep a pinned archive that matches the lock, else download and verify the mirror copy.

    Returns (path, action). A matching file is never replaced.
    """
    raw_dir = Path(raw_dir)
    lock = lock or read_lock(raw_dir)
    archive = raw_dir / lock["file"]
    if matches_lock(archive, lock):
        return archive, "kept"
    download(lock["mirror_url"], archive, lock, client)
    return archive, "downloaded"


def download(url, destination, expected, client=None):
    """Stream url into a temporary file, check its size and SHA-256, then move it into place.

    On any failure the temporary file is removed and destination is left as it was.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = None
    try:
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=".", suffix=".part")
        temporary = Path(name)
        with (
            os.fdopen(handle, "wb") as out,
            http_session(client) as http,
            http.stream("GET", url) as response,
        ):
            response.raise_for_status()
            for block in response.iter_bytes(BLOCK_SIZE):
                digest.update(block)
                out.write(block)
        size = temporary.stat().st_size
        if (size, digest.hexdigest()) != (expected["bytes"], expected["sha256"]):
            raise SnapshotError(
                f"{url} returned {size} bytes with SHA-256 {digest.hexdigest()}; "
                f"the lock expects {expected['bytes']} bytes with SHA-256 {expected['sha256']}."
            )
        os.replace(temporary, destination)
    except httpx.HTTPError as error:
        temporary.unlink(missing_ok=True)
        raise SnapshotError(f"download from {url} failed: {error}") from error
    except BaseException:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def http_session(client=None):
    """The caller's client, left open, or a fresh one closed after use."""
    if client is not None:
        return contextlib.nullcontext(client)
    return httpx.Client(follow_redirects=True, timeout=TIMEOUT_SECONDS)


def detect_encoding(archive, member=CSV_MEMBER):
    """Return the first candidate encoding that decodes the member cleanly."""
    with zipfile.ZipFile(archive) as opened:
        raw = opened.read(member)
    for encoding in CANDIDATE_ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("no candidate encoding decodes " + member)


def parse_dates(values):
    """Parse YYYYMMDD values, falling back to YYYYMMDDHHMMSS truncated to the day."""
    dates = pd.to_datetime(values, format=DATE_FORMAT, errors="coerce")
    timestamps = pd.to_datetime(values, format=TIMESTAMP_FORMAT, errors="coerce")
    return dates.fillna(timestamps.dt.normalize())


def load_source(raw_dir=RAW_DIR, member=CSV_MEMBER):
    """Read the verified pinned export as text, parsing only the two date columns."""
    archive = verify_snapshot(raw_dir)
    encoding = detect_encoding(archive, member)
    with zipfile.ZipFile(archive) as opened:
        with opened.open(member) as handle:
            frame = pd.read_csv(handle, sep=CSV_SEPARATOR, encoding=encoding, dtype="string")
    for column in DATE_COLUMNS:
        frame[column] = parse_dates(frame[column])
    return frame
