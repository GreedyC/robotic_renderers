import ast
from pathlib import Path

import numpy as np

import main
import render_blender
import render_mujoco_opengl


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


def test_application_sources_do_not_define_classes():
    root = Path(__file__).resolve().parents[1]
    for source_path in root.glob("*.py"):
        tree = ast.parse(source_path.read_text())
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def test_opengl_smoke_frame():
    frame = render_mujoco_opengl.render_frame(160, 90, time_s=0.5)

    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 5.0


def test_blender_installation_is_detected():
    blender = render_blender.backend_status()

    assert blender["available"] is True
    assert blender["version"] == "5.1.2"
