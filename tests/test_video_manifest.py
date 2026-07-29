import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_curated_video_manifest_matches_checked_in_artifacts():
    video_directory = Path(__file__).resolve().parents[1] / "videos"
    manifest_path = video_directory / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("curated videos are repository release artifacts")
    manifest = json.loads(manifest_path.read_text())

    for filename, description in manifest["artifacts"].items():
        path = video_directory / filename
        assert path.is_file()
        assert path.stat().st_size == description["bytes"]
        assert _sha256(path) == description["sha256"]


def test_curated_json_metadata_contains_no_absolute_paths():
    video_directory = Path(__file__).resolve().parents[1] / "videos"
    if not video_directory.is_dir():
        pytest.skip("curated videos are repository release artifacts")

    for path in video_directory.rglob("*.json"):
        text = path.read_text()
        assert "/private/" not in text
        assert "/Users/" not in text
        assert "/home/" not in text
