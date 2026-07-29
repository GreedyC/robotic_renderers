import ast
from pathlib import Path

import numpy as np
import pytest

import robotic_renderers
from robotic_renderers import cli as main
from robotic_renderers.backends import blender as render_blender
from robotic_renderers.backends import mujoco_opengl as render_mujoco_opengl


def test_required_function_api_exists():
    for backend in main.BACKEND_MODULES:
        module = main._module(backend)
        for function_name in (
            "backend_status",
            "render_frame",
            "render_still",
            "render_live",
            "record_video",
            "benchmark",
        ):
            assert callable(getattr(module, function_name))


def test_public_package_exposes_functional_backend_discovery():
    assert robotic_renderers.backend_names() == (
        "opengl",
        "warp",
        "blender",
        "ovrtx",
        "mitsuba",
    )
    assert robotic_renderers.load_backend("opengl").BACKEND_NAME == "opengl"


def test_application_sources_do_not_define_classes():
    root = Path(__file__).resolve().parents[1] / "src" / "robotic_renderers"
    for source_path in root.rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


@pytest.mark.integration
def test_opengl_smoke_frame():
    frame = render_mujoco_opengl.render_frame(160, 90, time_s=0.5)

    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 5.0


@pytest.mark.integration
def test_blender_installation_is_detected():
    blender = render_blender.backend_status()

    assert blender["available"] is True
    assert blender["version"]


def test_blender_status_reports_missing_binary_without_failing(monkeypatch):
    monkeypatch.setattr(render_blender, "find_blender", lambda: None)

    status = render_blender.backend_status()

    assert status["backend"] == "blender"
    assert status["available"] is False
    assert status["binary"] is None


def test_opengl_record_separates_render_and_encode_timing(tmp_path, monkeypatch):
    context = {"renderer": object()}
    closed = []
    monkeypatch.setattr(render_mujoco_opengl, "create_renderer", lambda **_kwargs: context)
    monkeypatch.setattr(
        render_mujoco_opengl,
        "render_context_frame",
        lambda _context, _time_s: np.zeros((2, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(render_mujoco_opengl, "trajectory_times", lambda *_args: [0.0, 0.5])
    monkeypatch.setattr(
        render_mujoco_opengl,
        "close_renderer",
        lambda closed_context: closed.append(closed_context),
    )

    def fake_write_video(frames, output_path, fps):
        assert fps == 2
        assert len(list(frames)) == 2
        return Path(output_path), 2, 3.0

    monkeypatch.setattr(render_mujoco_opengl, "write_video", fake_write_video)

    result = render_mujoco_opengl.record_video(
        tmp_path / "opengl.mp4",
        duration=1.0,
        fps=2,
        width=3,
        height=2,
    )

    assert result["video_fps"] == 2
    assert result["render_seconds"] >= 0.0
    assert result["encode_seconds"] >= 0.0
    assert result["render_seconds"] + result["encode_seconds"] == pytest.approx(result["seconds"])
    assert result["render_fps"] == pytest.approx(2 / max(result["render_seconds"], 1e-9))
    assert closed == [context]
