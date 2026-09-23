"""
Fetch the raw dataset from the UCI Machine Learning Repository.

Makes the project reproducible from a clean clone: ``data/raw`` is git-ignored,
so this script is how a reviewer obtains the exact same input file.

Usage::

    python scripts/download_data.py            # skip if already present
    python scripts/download_data.py --force    # re-download and overwrite
    python scripts/download_data.py --verify   # check an existing copy only

Dataset
-------
Default of Credit Card Clients (UCI dataset 350), Taiwan, April-September 2005.
Licensed CC BY 4.0. See ``src/schema.py`` for the full citation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.schema import (  # noqa: E402
    DATASET_CITATION,
    DATASET_DOWNLOAD_URL,
    DATASET_LANDING_URL,
    DATASET_LICENCE,
    DATASET_NAME,
    EXPECTED_RAW_COLUMNS,
    EXPECTED_RAW_ROWS,
)

logger = get_logger(__name__)

ARCHIVE_NAME = "uci_default_credit_card.zip"
EXPECTED_DATA_FILE = "default of credit card clients.xls"
# Plain urllib gets a 403 from some CDNs without a browser-ish agent.
USER_AGENT = "FinRiskAI/0.1 (academic project; +https://archive.ics.uci.edu)"
DOWNLOAD_TIMEOUT_SECONDS = 180


def _human_size(num_bytes: int) -> str:
    """Format a byte count for log output."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _sha256(path: Path) -> str:
    """SHA-256 of a file, streamed so memory stays flat."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path, force: bool = False) -> Path:
    """Download the dataset archive.

    Args:
        destination: Directory to write the archive into.
        force: Re-download even when the archive is already present.

    Returns:
        Path to the downloaded archive.

    Raises:
        SystemExit: On a network or HTTP failure, with actionable guidance.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / ARCHIVE_NAME

    if archive_path.exists() and not force:
        logger.info(
            "Archive already present (%s). Use --force to re-download.",
            _human_size(archive_path.stat().st_size),
        )
        return archive_path

    logger.info("Downloading %s", DATASET_DOWNLOAD_URL)
    request = urllib.request.Request(
        DATASET_DOWNLOAD_URL, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Download failed with HTTP {exc.code} ({exc.reason}).\n"
            f"The dataset can be downloaded manually from:\n  {DATASET_LANDING_URL}\n"
            f"Save '{EXPECTED_DATA_FILE}' into {destination}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Download failed - could not reach the server: {exc.reason}\n"
            "Check your internet connection, or download manually from:\n"
            f"  {DATASET_LANDING_URL}"
        ) from exc
    except TimeoutError as exc:
        raise SystemExit(
            f"Download timed out after {DOWNLOAD_TIMEOUT_SECONDS}s. "
            f"Retry, or download manually from:\n  {DATASET_LANDING_URL}"
        ) from exc

    if not payload:
        raise SystemExit("Download returned an empty response; nothing was written.")

    archive_path.write_bytes(payload)
    logger.info("Saved archive: %s (%s)", archive_path.name, _human_size(len(payload)))
    return archive_path


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract the dataset file from the archive.

    Args:
        archive_path: Downloaded ``.zip``.
        destination: Directory to extract into.

    Returns:
        Path to the extracted data file.

    Raises:
        SystemExit: If the archive is corrupt or holds no recognisable data file.
    """
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            logger.info("Archive contains: %s", ", ".join(names))

            data_members = [
                name
                for name in names
                if name.lower().endswith((".xls", ".xlsx", ".csv"))
                and not name.startswith("__MACOSX")
            ]
            if not data_members:
                raise SystemExit(
                    f"No .xls/.xlsx/.csv file inside {archive_path.name}. "
                    f"Contents: {names}"
                )

            member = data_members[0]
            # Guard against a path-traversal entry in an untrusted archive.
            safe_name = Path(member).name
            target = destination / safe_name
            with archive.open(member) as source, target.open("wb") as sink:
                sink.write(source.read())

            logger.info("Extracted: %s (%s)", target.name, _human_size(target.stat().st_size))
            return target
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"{archive_path.name} is not a valid zip archive - the download may be "
            "truncated. Re-run with --force."
        ) from exc


def verify_dataset(data_path: Path) -> bool:
    """Load the file and confirm it matches the expected contract.

    Args:
        data_path: Path to the extracted data file.

    Returns:
        True when the dataset loads and validates.
    """
    # Imported here so a download failure is not masked by an import error.
    from src.data_loader import DataLoadError, load_dataset

    logger.info("Verifying %s", data_path.name)
    try:
        result = load_dataset(data_path, validate=True, raise_on_invalid=False)
    except DataLoadError as exc:
        logger.error("Verification failed: %s", exc)
        return False

    profile = result.profile
    print("\n" + "=" * 68)
    print(f"  {DATASET_NAME}")
    print("=" * 68)
    print(f"  File            : {data_path.name}")
    print(f"  SHA-256         : {_sha256(data_path)}")
    print(f"  Rows            : {profile.n_rows:,}   (expected {EXPECTED_RAW_ROWS:,})")
    print(f"  Columns         : {profile.n_columns}      (expected {EXPECTED_RAW_COLUMNS})")
    print(f"  Missing cells   : {profile.total_missing_values:,}")
    print(f"  Duplicate rows  : {profile.duplicate_rows:,}")
    print(f"  Dup. excl. ID   : {profile.duplicate_rows_excluding_id:,}")
    print(f"  Numeric cols    : {len(profile.numeric_columns)}")
    print(f"  Validation      : {result.validation.summary()}")
    for warning in result.validation.warnings:
        print(f"    warning: {warning}")
    for error in result.validation.errors:
        print(f"    ERROR  : {error}")
    print("-" * 68)
    print(f"  Source   : {DATASET_LANDING_URL}")
    print(f"  Licence  : {DATASET_LICENCE}")
    print(f"  Citation : {DATASET_CITATION}")
    print("=" * 68 + "\n")

    return result.validation.is_valid


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Download the UCI Default of Credit Card Clients dataset."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist."
    )
    parser.add_argument(
        "--verify", action="store_true", help="Only verify the existing local copy."
    )
    args = parser.parse_args()

    raw_dir = settings.raw_dir
    data_path = raw_dir / EXPECTED_DATA_FILE

    if args.verify:
        if not data_path.exists():
            logger.error(
                "No dataset at %s. Run this script without --verify to download it.",
                data_path,
            )
            return 1
        return 0 if verify_dataset(data_path) else 1

    if data_path.exists() and not args.force:
        logger.info("Dataset already present at %s", data_path)
    else:
        archive = download_archive(raw_dir, force=args.force)
        data_path = extract_archive(archive, raw_dir)

    return 0 if verify_dataset(data_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
