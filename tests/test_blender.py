import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from robotic_renderers import defaults
from robotic_renderers.backends import blender


def _patch_job(monkeypatch, job: Path) -> None:
    job.mkdir()
    monkeypatch.setattr(blender, "_new_job", lambda _prefix: job)


def _patch_bundle(monkeypatch, job: Path, captured: dict | None = None) -> Path:
    manifest = job / "manifest.json"
    manifest.write_text("{}")

    def prepare(_destination, times, width, height, fps):
        if captured is not None:
            captured.update(
                {
                    "times": list(times),
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            )
        return manifest

    monkeypatch.setattr(blender, "prepare_blender_bundle", prepare)
    return manifest


def test_new_job_uses_platform_cache_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(defaults, "cache_directory", lambda: tmp_path)

    job = blender._new_job("unit-")
    try:
        assert job.parent == tmp_path / "jobs" / "blender"
        assert job.name.startswith("unit-")
    finally:
        shutil.rmtree(job)


def test_still_is_published_atomically(tmp_path: Path, monkeypatch):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    _patch_bundle(monkeypatch, job)
    output = tmp_path / "still.png"
    output.write_bytes(b"previous artifact")

    def render(_mode, _manifest, staged_output, _samples, warmup=0):
        assert warmup == 0
        Image.new("RGB", (16, 9), color=(12, 34, 56)).save(staged_output)
        return 1.25

    monkeypatch.setattr(blender, "_run_blender", render)

    result = blender.render_still(output, width=16, height=9, samples=8)

    assert Image.open(output).getpixel((0, 0)) == (12, 34, 56)
    assert result["output"] == str(output)
    assert result["seconds"] == pytest.approx(1.25)
    assert not job.exists()
    assert not list(tmp_path.glob(".*.part-*"))


def test_still_failure_preserves_existing_output(tmp_path: Path, monkeypatch):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    _patch_bundle(monkeypatch, job)
    output = tmp_path / "still.png"
    original = b"previous artifact"
    output.write_bytes(original)

    def fail(_mode, _manifest, staged_output, _samples, warmup=0):
        Path(staged_output).write_bytes(b"incomplete")
        raise RuntimeError("synthetic Blender failure")

    monkeypatch.setattr(blender, "_run_blender", fail)

    with pytest.raises(RuntimeError, match="synthetic Blender failure"):
        blender.render_still(output, width=16, height=9)

    assert output.read_bytes() == original
    assert not job.exists()
    assert not list(tmp_path.glob(".*.part-*"))


def test_video_result_separates_timeline_and_render_fps(tmp_path: Path, monkeypatch):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    _patch_bundle(monkeypatch, job)

    def render(mode, _manifest, frame_directory, _samples, warmup=0):
        assert mode == "frames"
        assert warmup == 0
        for index, color in enumerate(((20, 30, 40), (50, 60, 70)), start=1):
            Image.new("RGB", (16, 16), color=color).save(
                Path(frame_directory) / f"frame_{index:06d}.png"
            )
        return 0.5

    monkeypatch.setattr(blender, "_run_blender", render)
    output = tmp_path / "video.mp4"

    result = blender.record_video(
        output,
        duration=1.0,
        fps=2,
        width=16,
        height=16,
        samples=8,
    )

    assert output.is_file()
    assert result["frames"] == 2
    assert result["video_fps"] == 2
    assert result["render_fps"] == pytest.approx(4.0)
    assert "fps" not in result
    assert not job.exists()


def test_finite_live_view_terminates_reaps_and_cleans_job(tmp_path: Path, monkeypatch):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    _patch_bundle(monkeypatch, job)
    state = {"return_code": None}

    def wait(timeout=None):
        assert timeout == pytest.approx(0.01)
        raise subprocess.TimeoutExpired("blender", timeout)

    process = SimpleNamespace(
        pid=1234,
        wait=wait,
        poll=lambda: state["return_code"],
    )
    monkeypatch.setattr(blender, "_start_blender", lambda *args, **kwargs: process)

    def terminate(processes):
        assert processes == {"blender": process}
        state["return_code"] = -15
        return {"blender": -15}

    monkeypatch.setattr("robotic_renderers.processes.terminate_and_reap", terminate)

    result = blender.render_live(width=16, height=9, fps=1, duration=0.01)

    assert result["return_code"] == -15
    assert result["stopped_at_deadline"] is True
    assert not job.exists()


def test_live_view_reports_early_process_failure_and_cleans_job(tmp_path: Path, monkeypatch):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    _patch_bundle(monkeypatch, job)
    process = SimpleNamespace(
        pid=1234,
        wait=lambda timeout=None: 7,
        poll=lambda: 7,
    )
    monkeypatch.setattr(blender, "_start_blender", lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError, match="status 7"):
        blender.render_live(width=16, height=9, fps=1, duration=0.01)

    assert not job.exists()


def test_benchmark_authors_warmup_frames_but_times_only_measured_frames(
    tmp_path: Path,
    monkeypatch,
):
    job = tmp_path / "job"
    _patch_job(monkeypatch, job)
    captured = {}
    _patch_bundle(monkeypatch, job, captured)

    def render(mode, _manifest, output, _samples, warmup=0):
        assert mode == "benchmark"
        assert warmup == 2
        (Path(output) / "benchmark.json").write_text(
            '{"device":"Cycles/TEST","frames":3,"warmup":2,'
            '"init_seconds":1.5,"render_seconds":0.75}'
        )
        return 3.0

    monkeypatch.setattr(blender, "_run_blender", render)

    result = blender.benchmark(
        frame_count=3,
        width=16,
        height=9,
        fps=10,
        warmup=2,
        samples=8,
    )

    assert captured["times"] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert result["warmup"] == 2
    assert result["init_seconds"] == pytest.approx(1.5)
    assert result["render_seconds"] == pytest.approx(0.75)
    assert result["process_seconds"] == pytest.approx(3.0)
    assert result["render_fps"] == pytest.approx(4.0)
    assert not job.exists()


@pytest.mark.parametrize(
    ("frame_count", "warmup", "message"),
    ((0, 0, "frame_count"), (1, -1, "warmup")),
)
def test_benchmark_rejects_invalid_counts(frame_count, warmup, message):
    with pytest.raises(ValueError, match=message):
        blender.benchmark(frame_count=frame_count, warmup=warmup)


def test_worker_lighting_consumes_manifest_values(monkeypatch):
    created_data = {}
    created_objects = {}
    linked_objects = []
    orientations = []

    def new_light(*, name, type):
        light_data = SimpleNamespace(name=name, type=type)
        created_data[name] = light_data
        return light_data

    def new_object(*, name, object_data):
        light = SimpleNamespace(name=name, data=object_data, location=None)
        created_objects[name] = light
        return light

    fake_bpy = SimpleNamespace(
        data=SimpleNamespace(
            lights=SimpleNamespace(new=new_light),
            objects=SimpleNamespace(new=new_object),
        ),
        context=SimpleNamespace(
            collection=SimpleNamespace(
                objects=SimpleNamespace(link=linked_objects.append),
            )
        ),
    )
    fake_mathutils = SimpleNamespace(Vector=lambda values: tuple(values))
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    monkeypatch.setattr(
        blender,
        "_worker_point_light",
        lambda light, direction: orientations.append((light.name, direction)),
    )

    manifest_lights = [
        {
            "name": "edited_rim",
            "type": "spot",
            "position": [1.25, -2.5, 3.75],
            "direction": [-0.2, 0.4, -0.8],
            "ambient": [0.01, 0.02, 0.03],
            "diffuse": [0.12, 0.34, 0.56],
            "specular": [0.65, 0.43, 0.21],
            "intensity": 7.0,
            "range": 3.25,
            "bulb_radius": 0.04,
            "cutoff": 37.0,
            "exponent": 4.0,
            "cast_shadow": False,
        }
    ]

    blender._worker_add_lighting(manifest_lights)

    light_data = created_data["edited_rim"]
    light = created_objects["edited_rim"]
    assert linked_objects == [light]
    assert light_data.type == "SPOT"
    assert light_data.color == (0.12, 0.34, 0.56)
    assert light_data.energy == pytest.approx(140.0)
    assert light_data.use_shadow is False
    assert light_data.use_custom_distance is True
    assert light_data.cutoff_distance == pytest.approx(3.25)
    assert light_data.shadow_soft_size == pytest.approx(0.16)
    assert light_data.spot_size == pytest.approx(math.radians(37.0))
    assert light_data.spot_blend == pytest.approx(0.2)
    assert light.location == (1.25, -2.5, 3.75)
    assert orientations == [("edited_rim", (-0.2, 0.4, -0.8))]


@pytest.mark.parametrize(
    ("light_type", "expected_type", "expected_energy", "expected_size_key"),
    (
        ("directional", "SUN", 2.5, "angle"),
        ("point", "POINT", 25.0, "radius"),
        ("spot", "SPOT", 50.0, "radius"),
    ),
)
def test_worker_light_unit_compensation_is_type_specific(
    light_type,
    expected_type,
    expected_energy,
    expected_size_key,
):
    description = {
        "name": f"edited_{light_type}",
        "type": light_type,
        "position": [0.0, 1.0, 2.0],
        "direction": [0.0, 0.0, -1.0],
        "diffuse": [0.2, 0.3, 0.4],
        "intensity": 2.5,
        "range": 6.0,
        "bulb_radius": 0.03,
        "cutoff": 42.0,
        "exponent": 3.0,
        "cast_shadow": True,
    }

    settings = blender._worker_light_settings(description)

    assert settings["type"] == expected_type
    assert settings["energy"] == pytest.approx(expected_energy)
    assert expected_size_key in settings
    if light_type == "directional":
        assert settings["angle"] == pytest.approx(math.radians(3.0))
    elif light_type == "point":
        assert settings["radius"] == pytest.approx(0.03)
    else:
        assert settings["radius"] == pytest.approx(0.12)


def test_worker_light_settings_rejects_unknown_canonical_type():
    with pytest.raises(RuntimeError, match="Unsupported canonical light type"):
        blender._worker_light_settings({"type": "area"})
