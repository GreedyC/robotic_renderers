"""Keep isolated worker locks and reported runtime versions in sync."""

import tomllib
from pathlib import Path

import pytest

from robotic_renderers.backends import mitsuba, ovrtx
from robotic_renderers.workflows import cosmos3

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ("warp", "mitsuba", "ovrtx"))
def test_shared_runtime_dependencies_match_the_main_lock(backend):
    main = tomllib.loads((ROOT / "uv.lock").read_text())
    remote = tomllib.loads((ROOT / "runtime" / backend / "uv.lock").read_text())
    versions = {package["name"]: package["version"] for package in main["package"]}
    for package in remote["package"]:
        if package["name"] in versions:
            assert package["version"] == versions[package["name"]], package["name"]


def test_reported_remote_versions_match_runtime_pins():
    for backend, pins in (
        ("mitsuba", {"mitsuba": mitsuba.MITSUBA_VERSION}),
        ("ovrtx", {"ovrtx": ovrtx.OVRTX_VERSION, "ovstage": ovrtx.OVSTAGE_VERSION}),
    ):
        project = tomllib.loads((ROOT / "runtime" / backend / "pyproject.toml").read_text())
        lock = tomllib.loads((ROOT / "runtime" / backend / "uv.lock").read_text())
        locked = {package["name"]: package["version"] for package in lock["package"]}
        for name, version in pins.items():
            assert f"{name}=={version}" in project["project"]["dependencies"]
            assert locked[name] == version


def test_cosmos_setup_matches_manifest_and_resolves_packages_together():
    manifest = tomllib.loads((ROOT / "runtime/cosmos/manifest.toml").read_text())
    assert manifest["python-packages"]["vllm"] == cosmos3.VLLM_VERSION
    assert manifest["python-packages"]["vllm-omni-commit"] == cosmos3.VLLM_OMNI_COMMIT
    script = cosmos3._remote_setup_script()
    assert script.count('"$uv_bin" pip install') == 1
    assert f"vllm=={cosmos3.VLLM_VERSION}" in script
    assert "direct_url.json" in script
    assert 'origin.get("vcs_info", {}).get("commit_id")' in script
