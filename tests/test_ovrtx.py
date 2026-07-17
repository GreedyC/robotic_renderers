import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf

import main
import render_ovrtx


def test_parser_accepts_ovrtx_for_every_render_command():
    for command in ("setup", "still", "live", "record", "benchmark"):
        args = main.build_parser().parse_args([command, "--backend", "ovrtx"])
        assert args.backend == "ovrtx"
        assert args.host == "l4"


def test_ovrtx_matrix_uses_usd_row_vector_convention():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    matrix = render_ovrtx._matrix([1.0, 2.0, 3.0], rotation)

    np.testing.assert_array_equal(matrix[:3, :3], rotation.T)
    np.testing.assert_array_equal(matrix[3, :3], [1.0, 2.0, 3.0])


def test_ovrtx_floor_matches_blenders_200_meter_checker_plane():
    vertices, faces, uvs = render_ovrtx._plane_mesh()

    np.testing.assert_array_equal(vertices.min(axis=0), [-100.0, -100.0, 0.0])
    np.testing.assert_array_equal(vertices.max(axis=0), [100.0, 100.0, 0.0])
    assert faces.shape == (2, 3)
    assert uvs[:, 0].max() == 300.0
    assert uvs[:, 1].max() == 300.0


def test_ovrtx_bundle_contains_canonical_scene_and_animation(tmp_path: Path):
    bundle_path = render_ovrtx.prepare_ovrtx_bundle(
        tmp_path,
        times=[0.0, 1.0 / 30.0],
        width=160,
        height=90,
        fps=30,
        samples=8,
    )
    bundle = json.loads(bundle_path.read_text())
    scene_layer = Sdf.Layer.FindOrOpen(str(tmp_path / bundle["scene"]))
    assert scene_layer is not None
    scene = scene_layer.ExportToString()
    animation = np.load(tmp_path / bundle["animation"])

    assert bundle["render_mode"] == "PathTracing"
    assert bundle["animated_in_usd"] is True
    assert bundle["scene"].endswith(".usdc")
    assert bundle["frames"] == 2
    geom_count = len(bundle["geom_paths"])
    assert geom_count >= 30
    assert animation["geom_transforms"].shape == (2, geom_count, 4, 4)
    assert animation["camera_transforms"].shape == (2, 1, 4, 4)
    assert 'token omni:rtx:rendermode = "PathTracing"' in scene
    assert "int omni:rtx:pt:samplesPerPixel = 8" in scene
    assert 'token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]' in scene
    assert "startTimeCode = 1" in scene
    assert 'def Camera "Camera"' in scene
    assert 'def Mesh "Tiles_0"' in scene
    assert "asset inputs:diffuse_texture = @./floor_checker.png@" in scene
    assert 'def DiskLight "WarmRim"' in scene
    assert 'token omni:rtx:background:source:type = "domeLight"' in scene
    assert "bool omni:rtx:autoExposure:enabled = 0" in scene
    assert (tmp_path / "studio_gradient.png").is_file()
    assert (tmp_path / "floor_checker.png").is_file()


def test_worker_evaluates_each_authored_usd_frame_before_rendering(monkeypatch):
    usd_times = []

    def step(*, render_products, delta_time):
        assert render_products == {"/Render/Camera"}
        assert delta_time == 1.0 / 24.0
        mapping = SimpleNamespace(unmap=lambda: None)
        render_var = SimpleNamespace(map=lambda device: mapping)
        frame = SimpleNamespace(render_vars={"LdrColor": render_var})
        return {"/Render/Camera": SimpleNamespace(frames=[frame])}

    fake_ovrtx = SimpleNamespace(Device=SimpleNamespace(CPU="cpu"))
    monkeypatch.setitem(sys.modules, "ovrtx", fake_ovrtx)
    monkeypatch.setattr(
        np,
        "from_dlpack",
        lambda mapping: np.zeros((2, 3, 4), dtype=np.uint8),
    )
    renderer = SimpleNamespace(update_from_usd_time=usd_times.append, step=step)
    bundle = {
        "animated_in_usd": True,
        "fps": 24,
        "height": 2,
        "width": 3,
        "render_product": "/Render/Camera",
    }

    render_ovrtx._worker_frame(renderer, bundle, {}, index=0)
    render_ovrtx._worker_frame(renderer, bundle, {}, index=11)

    assert usd_times == [1.0 / 24.0, 12.0 / 24.0]


def test_malformed_handshake_does_not_block_and_reaps_stubborn_ssh(monkeypatch):
    class FakeStderr:
        read_count = 0

        def read(self):
            self.read_count += 1
            raise AssertionError("stderr must not be read while the child is alive")

    class FakeProcess:
        def __init__(self):
            self.stdout = SimpleNamespace(readline=lambda: b"not-a-handshake\n")
            self.stderr = FakeStderr()
            self.alive = True
            self.terminate_count = 0
            self.kill_count = 0
            self.wait_timeouts = []

        def poll(self):
            return None if self.alive else -9

        def terminate(self):
            self.terminate_count += 1

        def kill(self):
            self.kill_count += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if not self.kill_count:
                raise subprocess.TimeoutExpired("ssh", timeout)
            self.alive = False
            return -9

    process = FakeProcess()
    remote_terminations = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        render_ovrtx,
        "_terminate_remote_worker",
        lambda host, remote_job: remote_terminations.append((host, remote_job)),
    )

    frames = render_ovrtx._remote_stream_frames("l4", ".jobs/test", 4, 3, False)
    with pytest.raises(RuntimeError, match="not-a-handshake"):
        next(frames)

    assert process.stderr.read_count == 0
    assert remote_terminations == [("l4", ".jobs/test")]
    assert process.terminate_count == 1
    assert process.kill_count == 1
    assert process.wait_timeouts == [5.0, None]
    assert process.poll() == -9
