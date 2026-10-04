"""The pinned SITG snapshot: fetch, verify and read.

`source_manifest.json` is a lock file: code reads it and never writes it. Changing the pin is a
manual, reviewed edit. Only `fetch_live_snapshot` contacts SITG, and it writes outside the pin.
"""

import contextlib
import hashlib
import json
import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from geneva_permits.cleaning import parse_dates
from geneva_permits.paths import RAW_DIR

LOCK_NAME = "source_manifest.json"
LEGACY_NAME = "SIT_AUTOR_DOSSIER-CSV.zip"  # unversioned name used before the pin
LIVE_DIR_NAME = "live"
LIVE_MANIFEST_NAME = "manifest.json"

CSV_MEMBER = "SIT_AUTOR_DOSSIER.csv"
CSV_SEPARATOR = ";"
DATE_COLUMNS = ["DATE_DEPOT", "DATE_MAJ_2"]
DATE_INFO_MEMBER = "DOC/Informations_date.txt"

# Tried in order; the first one that decodes is used.
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

BLOCK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 120.0
FRIDAY = 4  # date.weekday()


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
    """Make the pinned archive available and return (path, action).

    A file that already matches the lock is never replaced. A matching archive under the legacy
    name is adopted; otherwise the mirror copy is downloaded and verified before it is moved in.
    """
    raw_dir = Path(raw_dir)
    lock = lock or read_lock(raw_dir)
    archive = raw_dir / lock["file"]
    if matches_lock(archive, lock):
        return archive, "kept"
    legacy = raw_dir / LEGACY_NAME
    if matches_lock(legacy, lock):
        os.replace(legacy, archive)
        return archive, "adopted"
    download(lock["mirror_url"], archive, client, expected=lock)
    return archive, "downloaded"


def fetch_live_snapshot(raw_dir=RAW_DIR, client=None, now=None):
    """Download today's SITG export into live/<date>/ with its own manifest.

    The pinned archive and the lock are left untouched; adopting a new snapshot means
    editing the lock by hand.
    """
    raw_dir = Path(raw_dir)
    lock = read_lock(raw_dir)
    now = now or datetime.now(UTC)
    url = lock["live_url"]
    target = raw_dir / LIVE_DIR_NAME / now.date().isoformat()
    archive = target / f"{Path(url).stem}_{now.date().isoformat()}.zip"

    headers = download(url, archive, client)
    with zipfile.ZipFile(archive) as opened:
        members = sorted(opened.namelist())
    manifest = {
        "url": url,
        "downloaded_at": now.isoformat(timespec="seconds"),
        "last_modified": headers.get("last-modified"),
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_of(archive),
        "members": members,
        **read_date_info(archive),
    }
    manifest["matches_pin"] = manifest["sha256"] == lock["sha256"]
    (target / LIVE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return archive, manifest


def download(url, destination, client=None, expected=None):
    """Stream url into a temporary file beside destination, check it, then move it into place.

    `expected` holds `bytes` and `sha256`; on any failure the temporary file is removed and
    destination is left as it was. Returns the response headers.
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
        if expected and (size, digest.hexdigest()) != (expected["bytes"], expected["sha256"]):
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
    return response.headers


def http_session(client=None):
    """The caller's client, left open, or a fresh one closed after use."""
    if client is not None:
        return contextlib.nullcontext(client)
    return httpx.Client(follow_redirects=True, timeout=TIMEOUT_SECONDS)


def read_date_info(archive):
    """SITG's own dating of an archive, from DOC/Informations_date.txt.

    The file states the zip creation time and that the data was extracted "le vendredi soir
    précédent" the download day; that Friday is derived here.
    """
    with zipfile.ZipFile(archive) as opened:
        text = opened.read(DATE_INFO_MEMBER).decode("utf-8")
    text = " ".join(text.split())
    created = re.search(r"fichier \.zip téléchargé : (\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2})", text)
    extracted = re.search(
        r"(extraites de la géodatabase du SITG le [^.]*?\d{2}\.\d{2}\.\d{4})", text
    )
    friday = re.search(r"le vendredi soir précédent le (\d{2}\.\d{2}\.\d{4})", text)
    info = {"sitg_zip_created": None, "sitg_extracted": None, "sitg_extraction_note": None}
    if created:
        stamp = datetime.strptime(created.group(1), "%d.%m.%Y %H:%M:%S")
        info["sitg_zip_created"] = stamp.isoformat()
    if extracted:
        info["sitg_extraction_note"] = extracted.group(1)
    if friday:
        reference = datetime.strptime(friday.group(1), "%d.%m.%Y").date()
        info["sitg_extracted"] = previous_friday(reference).isoformat()
    return info


def previous_friday(day):
    """The last Friday strictly before day."""
    return day - timedelta(days=(day.weekday() - FRIDAY - 1) % 7 + 1)


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
