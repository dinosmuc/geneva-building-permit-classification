"""Load, clean, split and freeze the SITG export. Leakage checks live here."""

import hashlib
import json
import re
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

# https://sitg.ge.ch/donnees/sit-autor-dossier — CSV, no geometry needed.
# The service rebuilds daily, so only a recorded hash makes a run reproducible.
SOURCE_URL = "https://ge.ch/sitg/geodata/SITG/OPENDATA/SIT_AUTOR_DOSSIER-CSV.zip"
MANIFEST_NAME = "source_manifest.json"

# Anchored on this file, so every path works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / SOURCE_URL.split("/")[-1]

CSV_MEMBER = "SIT_AUTOR_DOSSIER.csv"
CSV_SEPARATOR = ";"
DATE_FORMAT = "%Y%m%d"  # DATE_DEPOT is stored as 20020226
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"  # three filing dates carry a time of day
DATE_COLUMNS = ["DATE_DEPOT", "DATE_MAJ_2"]

# Tried in order; the first one that decodes is recorded in the manifest.
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

# Sentinels the export uses instead of empty cells, one per field.
MISSING_OPERATION_CODE = "--"
MISSING_DESCRIPTION = "non renseigné"
MISSING_STATUS = "-"
MISSING_OPERATION_LABEL = "Non renseigné (valeur par défaut à l'importation des données)"
PLACEHOLDERS = {
    "TYPE_OPERATION": MISSING_OPERATION_CODE,
    "DESCRIPTION": MISSING_DESCRIPTION,
    "STATUT": MISSING_STATUS,
    "OPERATION": MISSING_OPERATION_LABEL,
}


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


def parse_dates(values):
    """Parse YYYYMMDD values, falling back to YYYYMMDDHHMMSS truncated to the day."""
    dates = pd.to_datetime(values, format=DATE_FORMAT, errors="coerce")
    timestamps = pd.to_datetime(values, format=TIMESTAMP_FORMAT, errors="coerce")
    return dates.fillna(timestamps.dt.normalize())


def load_source(archive=ARCHIVE_PATH, member=CSV_MEMBER):
    """Read the export as text, parsing only the two date columns."""
    encoding = detect_encoding(archive, member)
    with zipfile.ZipFile(archive) as opened:
        with opened.open(member) as handle:
            frame = pd.read_csv(handle, sep=CSV_SEPARATOR, encoding=encoding, dtype="string")
    for column in DATE_COLUMNS:
        frame[column] = parse_dates(frame[column])
    return frame


def is_missing(values, column):
    """True where a field is null or holds that field's placeholder."""
    missing = values.isna()
    if column in PLACEHOLDERS:
        missing |= values.eq(PLACEHOLDERS[column]).fillna(False)
    return missing


def normalize_description(text):
    """Grouping key: casefold, strip accents, collapse punctuation and whitespace.

    Used only for repetition and overlap checks; the model reads the original text.
    """
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[\W_]+", " ", text).strip()


def operation_labels(frame):
    """Official definition of each operation code, taken from the export itself.

    TYPE_OPERATION holds the code and OPERATION its French definition. A code with
    several label spellings keeps the longest, which is the uncorrupted one.
    """
    labels = {}
    for code, label in zip(frame["TYPE_OPERATION"], frame["OPERATION"], strict=True):
        if pd.isna(code) or pd.isna(label) or code == MISSING_OPERATION_CODE:
            continue
        if len(label) > len(labels.get(code, "")):
            labels[code] = label
    return dict(sorted(labels.items()))


if __name__ == "__main__":
    archive, manifest = download_source()
    print(json.dumps(manifest, indent=2))
    print("encoding:", detect_encoding(archive))
