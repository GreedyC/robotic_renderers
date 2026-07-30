import io
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_renderers import cli as main
from robotic_renderers import defaults
from robotic_renderers.backends import mitsuba as render_mitsuba


def test_parser_accepts_mitsuba_for_every_render_command():
    for command in ("setup", "still", "live", "record", "benchmark"):
        args = main.build_parser().parse_args([command, "--backend", "mitsuba"])

        assert args.backend == "mitsuba"
        assert args.host == "render-host"


def test_mitsuba_bundle_preserves_canonical_scene_and_animation(tmp_path: Path):
    bundle_path = render_mitsuba.prepare_mitsuba_bundle(
        tmp_path,
        times=[0.0, 1.0 / 30.0],
        width=160,
        height=90,
        fps=30,
        samples=8,
    )
    bundle = json.loads(bundle_path.read_text())
    manifest = json.loads((tmp_path / bundle["manifest"]).read_text())
    geometry = np.load(tmp_path / bundle["geometry"])

    assert bundle["variant"] == "cuda_ad_rgb"
    assert bundle["samples"] == 8
    assert bundle["frames"] == 2
    assert bundle["width"] == 160
    assert bundle["height"] == 90
    assert bundle["max_depth"] == 12
    assert bundle["denoise"] is True
    assert len(bundle["shapes"]) == len(manifest["geoms"])
    assert len(bundle["shapes"]) >= 30
    assert len(manifest["frames"]) == 2
    assert {geom["name"] for geom in manifest["geoms"]} >= {
        "red_box",
        "blue_sphere",
        "gold_cylinder",
        "green_torus_mesh",
        "gold_gem_mesh",
    }
    assert sum(shape["kind"] == "plane" for shape in bundle["shapes"]) == 1
    assert all(
        (tmp_path / shape["filename"]).is_file()
        for shape in bundle["shapes"]
        if shape["kind"] == "mesh"
    )
    assert any(name.startswith("vertices_") for name in geometry.files)
    assert (tmp_path / bundle["sky"]).is_file()


def test_mitsuba_worker_count_uses_one_persistent_worker_per_gpu(monkeypatch):
    monkeypatch.setattr(
        render_mitsuba,
        "_gpu_inventory",
        lambda host: [f"GPU {index}" for index in range(8)],
    )

    assert render_mitsuba._worker_count("render-host", frame_count=48, workers=None) == 8
    assert render_mitsuba._worker_count("render-host", frame_count=3, workers=None) == 3
    assert render_mitsuba._worker_count("render-host", frame_count=48, workers=2) == 2


def test_mitsuba_jobs_use_the_writable_platform_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(defaults, "cache_directory", lambda: tmp_path)

    assert render_mitsuba._job_root() == tmp_path / "jobs" / "mitsuba"


def test_mitsuba_batch_command_pins_the_requested_gpu():
    command = render_mitsuba._batch_worker_command(
        ".jobs/example",
        gpu=5,
        indices=[5, 13],
        output_directory=".jobs/example/frames",
    )

    assert "CUDA_VISIBLE_DEVICES=5" in command
    assert "--indices 5,13" in command
    assert "cuda_ad_rgb" in command
    assert "PYTHONPATH=" in command
    assert "-m robotic_renderers.backends.mitsuba __worker" in command


def test_mitsuba_rejects_oversized_remote_error_before_allocating(monkeypatch):
    stdout = io.BytesIO(
        render_mitsuba.STREAM_ERROR + struct.pack("!I", render_mitsuba.MAX_ERROR_BYTES + 1)
    )
    process = type(
        "FakeProcess",
        (),
        {"stdout": stdout, "stderr": io.BytesIO()},
    )()
    monkeypatch.setattr(render_mitsuba.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        render_mitsuba,
        "_terminate_remote_job_workers",
        lambda host, remote_job: None,
    )
    monkeypatch.setattr(render_mitsuba, "_terminate_and_reap", lambda process: None)

    frames = render_mitsuba._remote_stream_frames("render-host", ".jobs/example", 4, 3, False)
    with pytest.raises(RuntimeError, match="outside the allowed range"):
        next(frames)


def test_mitsuba_remote_setup_uses_the_frozen_official_runtime():
    script = render_mitsuba._remote_install_script("0123456789abcdef", force=False)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "uv_bin" in script
    assert "sync --directory" in script
    assert "--frozen --no-dev" in script
    assert 'mv "$incoming/source" "$root/source"' in script
    assert 'mv "$incoming/runtime" "$root/runtime"' in script
    assert "cuda_ad_rgb" in script


def test_mitsuba_benchmark_command_uses_the_packaged_worker_module():
    command = render_mitsuba._benchmark_worker_command(
        ".jobs/example",
        gpu=2,
        count=4,
        warmup=1,
        start_index=2,
        stride=4,
    )

    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert "PYTHONPATH=" in command
    assert "-m robotic_renderers.backends.mitsuba __worker" in command


def _fake_worker_process(return_code=None, stdout="", stderr=""):
    process = SimpleNamespace(
        returncode=return_code,
        terminate_count=0,
        kill_count=0,
        wait_timeouts=[],
    )

    def poll():
        return process.returncode

    def terminate():
        process.terminate_count += 1

    def kill():
        process.kill_count += 1
        process.returncode = -9

    def wait(timeout=None):
        process.wait_timeouts.append(timeout)
        if process.returncode is None:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return process.returncode

    process.poll = poll
    process.terminate = terminate
    process.kill = kill
    process.wait = wait
    process.communicate = lambda: (stdout, stderr)
    return process


def test_mitsuba_partial_worker_spawn_terminates_escalates_and_reaps(monkeypatch):
    first_process = _fake_worker_process()
    spawn_count = 0
    remote_terminations = []

    def popen(*args, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 1:
            return first_process
        raise OSError("second SSH spawn failed")

    monkeypatch.setattr(render_mitsuba.subprocess, "Popen", popen)
    monkeypatch.setattr(
        render_mitsuba,
        "_terminate_remote_job_workers",
        lambda host, remote_job: remote_terminations.append((host, remote_job)),
    )

    with pytest.raises(OSError, match="second SSH spawn failed"):
        render_mitsuba._spawn_worker_processes(
            "render-host",
            ".jobs/example",
            ["worker zero", "worker one"],
        )

    assert remote_terminations == [("render-host", ".jobs/example")]
    assert first_process.terminate_count == 1
    assert first_process.kill_count == 1
    assert first_process.wait_timeouts == [3.0, None]
    assert first_process.poll() == -9


def test_mitsuba_failed_worker_collection_cleans_every_process(monkeypatch):
    failed_process = _fake_worker_process(
        return_code=7,
        stderr="worker zero exploded",
    )
    sibling_process = _fake_worker_process()
    remote_terminations = []
    monkeypatch.setattr(
        render_mitsuba,
        "_terminate_remote_job_workers",
        lambda host, remote_job: remote_terminations.append((host, remote_job)),
    )

    with pytest.raises(RuntimeError, match="worker zero exploded"):
        render_mitsuba._collect_processes_fail_fast(
            "render-host",
            ".jobs/example",
            [failed_process, sibling_process],
            "batch worker",
        )

    assert remote_terminations == [("render-host", ".jobs/example")]
    assert failed_process.wait_timeouts == [3.0]
    assert sibling_process.terminate_count == 1
    assert sibling_process.kill_count == 1
    assert sibling_process.wait_timeouts == [3.0, None]
    assert sibling_process.poll() == -9


def test_mitsuba_upload_failure_removes_the_new_remote_job(tmp_path: Path, monkeypatch):
    commands = []

    def run_ssh(host, command, **kwargs):
        commands.append((host, command, kwargs))
        return SimpleNamespace(returncode=0)

    def fail_upload(command):
        raise OSError("rsync failed")

    monkeypatch.setattr(render_mitsuba, "run_ssh", run_ssh)
    monkeypatch.setattr(render_mitsuba, "_run", fail_upload)

    with pytest.raises(OSError, match="rsync failed"):
        render_mitsuba._upload_job("render-host", tmp_path, "upload-example")

    assert len(commands) == 2
    assert "install -d" in commands[0][1]
    assert 'rm -rf "$HOME/code/robot_rendering_mitsuba/.jobs/upload-example"' in commands[1][1]


def test_mitsuba_scene_lights_are_translated_from_the_shared_manifest(tmp_path: Path):
    manifest = {
        "camera": {"fovy": 44.0},
        "frames": [
            {
                "camera": {
                    "position": [1.0, 2.0, 3.0],
                    "rotation": np.eye(3).tolist(),
                },
                "transforms": [],
            }
        ],
        "geoms": [],
        "lights": [
            {
                "name": "custom_key",
                "type": "directional",
                "position": [9.0, 8.0, 7.0],
                "direction": [0.1, 0.2, -0.9],
                "diffuse": [0.2, 0.3, 0.4],
                "intensity": 2.5,
                "cast_shadow": True,
            },
            {
                "name": "custom_fill",
                "type": "point",
                "position": [-1.0, -2.0, 1.5],
                "direction": [0.0, 0.0, -1.0],
                "diffuse": [0.4, 0.5, 0.6],
                "intensity": 3.0,
                "cast_shadow": False,
            },
            {
                "name": "custom_rim",
                "type": "spot",
                "position": [0.5, 1.0, 2.0],
                "direction": [-0.25, 0.5, -1.0],
                "diffuse": [0.7, 0.2, 0.1],
                "intensity": 4.0,
                "cutoff": 42.0,
                "exponent": 3.0,
                "cast_shadow": True,
            },
        ],
    }
    bundle = {
        "manifest": "manifest.json",
        "geometry": "geometry.npz",
        "sky": "sky.exr",
        "shapes": [],
        "width": 160,
        "height": 90,
        "samples": 8,
        "max_depth": 12,
        "rr_depth": 5,
        "denoise": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    bundle_path = tmp_path / "mitsuba_bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    np.savez(tmp_path / "geometry.npz")
    loaded = {}

    def look_at(**kwargs):
        return ("look_at", kwargs)

    def load_dict(description, optimize):
        loaded.update(description)
        assert optimize is False
        return "scene"

    fake_mitsuba = SimpleNamespace(
        ScalarTransform4f=SimpleNamespace(
            look_at=look_at,
            rotate=lambda axis, angle: ("rotate", axis, angle),
        ),
        load_dict=load_dict,
        traverse=lambda scene: {},
    )

    runtime = render_mitsuba._worker_build_scene(fake_mitsuba, bundle_path)

    np.testing.assert_allclose(loaded["custom_key"]["direction"], [0.1, 0.2, -0.9])
    np.testing.assert_allclose(
        loaded["custom_key"]["irradiance"]["value"],
        [0.5, 0.75, 1.0],
    )
    assert loaded["custom_fill"]["type"] == "point"
    assert loaded["custom_fill"]["position"] == [-1.0, -2.0, 1.5]
    np.testing.assert_allclose(
        loaded["custom_fill"]["intensity"]["value"],
        [1.2, 1.5, 1.8],
    )
    spot = loaded["custom_rim"]
    assert spot["type"] == "spot"
    assert spot["to_world"] == (
        "look_at",
        {
            "origin": [0.5, 1.0, 2.0],
            "target": [0.25, 1.5, 1.0],
            "up": [0.0, 0.0, 1.0],
        },
    )
    np.testing.assert_allclose(spot["intensity"]["value"], [5.6, 1.6, 0.8])
    assert spot["cutoff_angle"] == 42.0
    assert spot["beam_width"] == pytest.approx(render_mitsuba._worker_spot_beam_width(42.0, 3.0))
    assert runtime["light_cast_shadow"] == {
        "custom_key": True,
        "custom_fill": False,
        "custom_rim": True,
    }
    assert "sun_key" not in loaded
