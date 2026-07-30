"""Download and extract the UCSD Anomaly Detection Dataset.

The archive is ~740 MB and expands to image sequences for Ped1 and Ped2.
Re-running is safe: existing downloads and extractions are reused.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
ARCHIVE_NAME = "UCSD_Anomaly_Dataset.tar.gz"
REPO = Path(__file__).resolve().parent.parent


def _report(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100.0, 100.0 * done * block / total)
    mb = done * block / 1e6
    sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} MB")
    sys.stdout.flush()


def download(dest: Path, url: str = URL) -> Path:
    archive = dest / ARCHIVE_NAME
    if archive.exists() and archive.stat().st_size > 0:
        print(f"archive already present ({archive.stat().st_size / 1e6:.0f} MB)")
        return archive
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = archive.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    tmp.replace(archive)
    print("\ndownload complete")
    return archive


def extract(archive: Path, dest: Path) -> Path:
    marker = dest / "UCSD_Anomaly_Dataset.v1p2"
    if marker.is_dir():
        print("already extracted")
        return marker
    print("extracting (this takes a minute)")
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse absolute or parent-escaping member paths.
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(dest)
    print("extraction complete")
    return marker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=REPO / "data" / "raw")
    p.add_argument("--url", default=URL)
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    archive = download(args.dest, args.url)
    root = extract(archive, args.dest)

    if not args.keep_archive and archive.exists():
        archive.unlink()
        print("removed archive (pass --keep-archive to keep it)")

    ped2 = root / "UCSDped2"
    if not ped2.is_dir():
        raise SystemExit(f"Expected {ped2} after extraction")
    n_train = len([d for d in (ped2 / "Train").iterdir() if d.is_dir()])
    n_test = len([d for d in (ped2 / "Test").iterdir() if d.is_dir() and not d.name.endswith("_gt")])
    print(f"\nUCSDped2 ready: {n_train} train clips, {n_test} test clips")
    print(f"root: {root}")


if __name__ == "__main__":
    main()
