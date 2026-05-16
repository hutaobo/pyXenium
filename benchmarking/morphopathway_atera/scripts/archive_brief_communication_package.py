from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


EXCLUDED_NAMES = {
    "package_file_manifest.csv",
    "package_file_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SHA256 checksums and a zip archive for a morphopathway Brief Communication package."
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--archive-path", type=Path, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _read_json_or_empty(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _archive_manifest_matches(archive_path: Path, archive_manifest_path: Path) -> bool:
    if not archive_path.exists() or not archive_manifest_path.exists():
        return False
    archive_manifest = _read_json_or_empty(archive_manifest_path)
    return (
        archive_manifest.get("archive_sha256") == _sha256(archive_path)
        and int(archive_manifest.get("archive_size_bytes", -1)) == int(archive_path.stat().st_size)
    )


def _payload_files(package_dir: Path, archive_path: Path) -> list[Path]:
    files: list[Path] = []
    archive_path = archive_path.resolve()
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if path.resolve() == archive_path:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(package_dir).as_posix())


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    if not package_dir.exists() or not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory does not exist: {package_dir}")

    archive_path = args.archive_path
    if archive_path is None:
        archive_path = package_dir.parent / f"{package_dir.name}.zip"
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files = _payload_files(package_dir, archive_path)
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(package_dir).as_posix(),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )

    manifest_frame = pd.DataFrame(rows)
    manifest_csv = package_dir / "package_file_manifest.csv"
    manifest_json = package_dir / "package_file_manifest.json"
    archive_manifest_path = archive_path.with_suffix(".manifest.json")

    existing_manifest = _read_json_or_empty(manifest_json)
    if (
        existing_manifest.get("files") == rows
        and manifest_csv.exists()
        and _archive_manifest_matches(archive_path, archive_manifest_path)
    ):
        print(archive_manifest_path.read_text(encoding="utf-8"))
        return

    _write_text_if_changed(manifest_csv, manifest_frame.to_csv(index=False))
    manifest = {
        "package_dir": str(package_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "payload_file_count": int(len(manifest_frame)),
        "payload_size_bytes": int(manifest_frame["size_bytes"].sum()) if not manifest_frame.empty else 0,
        "files": rows,
    }
    _write_text_if_changed(manifest_json, json.dumps(manifest, indent=2))

    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in _payload_files(package_dir, archive_path):
            archive.write(path, arcname=f"{package_dir.name}/{path.relative_to(package_dir).as_posix()}")
        archive.write(manifest_csv, arcname=f"{package_dir.name}/{manifest_csv.name}")
        archive.write(manifest_json, arcname=f"{package_dir.name}/{manifest_json.name}")

    archive_manifest = {
        "archive_path": str(archive_path),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_sha256": _sha256(archive_path),
        "package_dir": str(package_dir),
        "package_file_manifest_csv": str(manifest_csv),
        "package_file_manifest_json": str(manifest_json),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "payload_file_count": int(len(manifest_frame)),
    }
    archive_manifest_path.write_text(json.dumps(archive_manifest, indent=2), encoding="utf-8")
    print(json.dumps(archive_manifest, indent=2))


if __name__ == "__main__":
    main()
