"""Download, verify and read the SITG export."""

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from geneva_permits.cleaning import parse_dates
from geneva_permits.paths import RAW_DIR

# https://sitg.ge.ch/donnees/sit-autor-dossier — CSV, no geometry needed.
# The service rebuilds daily, so only a recorded hash makes a run reproducible.
SOURCE_URL = "https://ge.ch/sitg/geodata/SITG/OPENDATA/SIT_AUTOR_DOSSIER-CSV.zip"
MANIFEST_NAME = "source_manifest.json"
ARCHIVE_PATH = RAW_DIR / SOURCE_URL.split("/")[-1]

CSV_MEMBER = "SIT_AUTOR_DOSSIER.csv"
CSV_SEPARATOR = ";"
DATE_COLUMNS = ["DATE_DEPOT", "DATE_MAJ_2"]

# Tried in order; the first one that decodes is recorded in the manifest.
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def sha256_of(path):
    """Hash a file in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_source(raw_dir=RAW_DIR, url=SOURCE_URL):
    """Download the export and write a provenance manifest beside it."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / url.split("/")[-1]

    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    archive.write_bytes(response.content)  # 8 MB, no need to stream

    with zipfile.ZipFile(archive) as opened:
        members = sorted(opened.namelist())

    manifest = {
        "url": url,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_modified": response.headers.get("last-modified"),
        "bytes": archive.stat().st_size,
        "sha256": sha256_of(archive),
        "members": members,
    }
    manifest_path = raw_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return archive, manifest


def detect_encoding(archive=ARCHIVE_PATH, member=CSV_MEMBER):
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


def load_source(archive=ARCHIVE_PATH, member=CSV_MEMBER):
    """Read the export as text, parsing only the two date columns."""
    encoding = detect_encoding(archive, member)
    with zipfile.ZipFile(archive) as opened:
        with opened.open(member) as handle:
            frame = pd.read_csv(handle, sep=CSV_SEPARATOR, encoding=encoding, dtype="string")
    for column in DATE_COLUMNS:
        frame[column] = parse_dates(frame[column])
    return frame


if __name__ == "__main__":
    archive, manifest = download_source()
    print(json.dumps(manifest, indent=2))
    print("encoding:", detect_encoding(archive))
