import io
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf

from robotic_renderers import cli as main
from robotic_renderers import defaults, scene_bundle
from robotic_renderers.backends import ovrtx as render_ovrtx


def test_parser_accepts_ovrtx_for_every_render_command():
    for command in ("setup", "still", "live", "record", "benchmark"):
        args = main.build_parser().parse_args([command, "--backend", "ovrtx"])
        assert args.backend == "ovrtx"
        assert args.host == "render-host"


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


def test_ovrtx_jobs_and_default_aovs_use_writable_user_directories(
    tmp_path: Path,
    monkeypatch,
):
    cache = tmp_path / "cache"
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(defaults, "cache_directory", lambda: cache)
    monkeypatch.setattr(defaults, "output_directory", lambda: outputs)

    assert render_ovrtx._job_root() == cache / "jobs" / "ovrtx"
    assert render_ovrtx._aov_directory(None, "live") == outputs / "ovrtx_live_aovs"


def test_ovrtx_status_requires_local_openusd(monkeypatch):
    monkeypatch.setattr(
        render_ovrtx,
        "_ssh_probe",
        lambda host, command: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA L40S",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        render_ovrtx,
        "_openusd_error",
        lambda: render_ovrtx.OPENUSD_INSTALL_HINT,
    )

    status = render_ovrtx.backend_status("render-host")

    assert status["available"] is False
    assert status["local_openusd"] is False
    assert status["device"] == "NVIDIA L40S"
    assert "robotic-renderers[ovrtx]" in status["error"]


def test_load_sdf_reports_an_actionable_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pxr", None)

    with pytest.raises(RuntimeError, match=r"robotic-renderers\[ovrtx\]"):
        render_ovrtx._load_sdf()


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


def test_ovrtx_lights_follow_scene_bundle_values(monkeypatch, tmp_path: Path):
    original_prepare = scene_bundle.prepare_scene_bundle

    def prepare_with_sentinel_lights(destination, times, width, height, fps):
        path = original_prepare(destination, times, width, height, fps)
        manifest = json.loads(path.read_text())
        directional, point, spot = manifest["lights"]
        directional.update(
            position=[4.0, 5.0, 6.0],
            direction=[0.0, 0.0, -1.0],
            diffuse=[0.11, 0.22, 0.33],
            intensity=1.5,
            cast_shadow=False,
        )
        point.update(
            position=[7.0, 8.0, 9.0],
            diffuse=[0.12, 0.23, 0.34],
            intensity=2.0,
            bulb_radius=0.42,
            cast_shadow=True,
        )
        spot.update(
            position=[-4.0, -5.0, 6.0],
            direction=[0.0, 0.0, -1.0],
            diffuse=[0.13, 0.24, 0.35],
            intensity=3.0,
            bulb_radius=0.17,
            cutoff=66.0,
            exponent=3.25,
            cast_shadow=False,
        )
        path.write_text(json.dumps(manifest, separators=(",", ":")))
        return path

    monkeypatch.setattr(scene_bundle, "prepare_scene_bundle", prepare_with_sentinel_lights)
    bundle_path = render_ovrtx.prepare_ovrtx_bundle(
        tmp_path,
        times=[0.0],
        width=160,
        height=90,
        fps=30,
        samples=8,
    )
    bundle = json.loads(bundle_path.read_text())
    layer = Sdf.Layer.FindOrOpen(str(tmp_path / bundle["scene"]))
    assert layer is not None
    scene = layer.ExportToString()

    assert "color3f inputs:color = (0.11, 0.22, 0.33)" in scene
    assert "float inputs:intensity = 315" in scene
    assert "bool inputs:shadow:enable = 0" in scene
    assert "color3f inputs:color = (0.12, 0.23, 0.34)" in scene
    assert "float inputs:radius = 0.42" in scene
    assert "double3 xformOp:translate = (7, 8, 9)" in scene
    assert "color3f inputs:color = (0.13, 0.24, 0.35)" in scene
    assert "float inputs:radius = 0.17" in scene
    assert "float inputs:shaping:cone:angle = 33" in scene
    assert "float inputs:shaping:focus = 3.25" in scene
    assert "inputs:shaping:cone:softness" not in scene


def test_ovrtx_realtime_bundle_authors_requested_annotation_aovs(tmp_path: Path):
    bundle_path = render_ovrtx.prepare_ovrtx_bundle(
        tmp_path,
        times=[0.0],
        width=160,
        height=90,
        fps=30,
        samples=8,
        render_mode="realtime",
        aovs=("depth", "semantic", "instance", "depth"),
    )
    bundle = json.loads(bundle_path.read_text())
    scene_layer = Sdf.Layer.FindOrOpen(str(tmp_path / bundle["scene"]))
    assert scene_layer is not None
    scene = scene_layer.ExportToString()

    assert bundle["render_mode_name"] == "realtime"
    assert bundle["render_mode"] == "RealTimePathTracing"
    assert bundle["aovs"] == ["depth", "semantic", "instance"]
    assert bundle["annotation_product"] == "/Render/Annotations"
    assert bundle["annotation_render_vars"] == [
        "LdrColor",
        "DistanceToImagePlaneSD",
        "SemanticSegmentation",
        "SemanticIdMap",
    ]
    assert 'def RenderProduct "Annotations"' in scene
    assert 'token omni:rtx:rendermode = "RealTimePathTracing"' in scene
    assert 'def RenderVar "DistanceToImagePlaneSD"' in scene
    assert 'def RenderVar "SemanticSegmentation"' in scene
    assert 'def RenderVar "SemanticIdMap"' in scene
    assert "SemanticsAPI:class" in scene
    assert "SemanticsAPI:label" in scene


def test_ovrtx_rejects_unknown_render_modes_and_aovs():
    with pytest.raises(ValueError, match="Unsupported ovrtx render mode"):
        render_ovrtx._normalize_render_mode("raster")
    with pytest.raises(ValueError, match="Unsupported ovrtx AOV"):
        render_ovrtx._normalize_aovs(("depth", "optical-flow"))


def test_aov_video_writer_streams_lossless_numbered_frames_and_manifest(
    monkeypatch,
    tmp_path: Path,
):
    writers = []

    class FakeWriter:
        def __init__(self, output_path):
            self.output_path = Path(output_path)
            self.frames = []
            self.closed = False

        def append_data(self, frame):
            self.frames.append(np.asarray(frame).copy())

        def close(self):
            self.closed = True
            self.output_path.write_bytes(b"encoded")

    def video_writer(output_path, fps):
        assert fps == 12
        writer = FakeWriter(output_path)
        writers.append((Path(output_path), writer))
        return writer

    monkeypatch.setattr(render_ovrtx, "_video_writer", video_writer)
    frames = [
        {
            "rgb": np.full((2, 3, 3), index, dtype=np.uint8),
            "depth": np.full((2, 3), 2.0 + index, dtype=np.float32),
            "semantic": np.full((2, 3), index + 1, dtype=np.uint32),
            "_labels": {"semantic": {"0": "unlabeled", "1": "robot"}},
        }
        for index in range(2)
    ]
    aov_dir = tmp_path / "aovs"
    stale_dir = aov_dir / "rgb_depth_frames"
    stale_dir.mkdir(parents=True)
    np.save(stale_dir / "frame_999999.npy", np.ones((1,), dtype=np.float32))
    (aov_dir / "rgb_albedo_manifest.json").write_text('{"stale": true}\n')

    _, frame_count, _, outputs = render_ovrtx._write_video_and_aovs(
        iter(frames),
        tmp_path / "rgb.mp4",
        12,
        ("depth", "semantic"),
        aov_dir,
    )

    assert frame_count == 2
    assert len(writers) == 3
    assert all(writer.closed for _, writer in writers)
    assert not (stale_dir / "frame_999999.npy").exists()
    assert not (aov_dir / "rgb_albedo_manifest.json").exists()
    np.testing.assert_array_equal(
        np.load(aov_dir / "rgb_depth_frames" / "frame_000001.npy"),
        frames[1]["depth"],
    )
    np.testing.assert_array_equal(
        np.load(aov_dir / "rgb_semantic_frames" / "frame_000000.npy"),
        frames[0]["semantic"],
    )
    depth_manifest = json.loads((aov_dir / "rgb_depth_manifest.json").read_text())
    assert depth_manifest == {
        "aov": "depth",
        "backend": "ovrtx",
        "dtype": "float32",
        "fps": 12,
        "frame_pattern": "rgb_depth_frames/frame_%06d.npy",
        "frames": 2,
        "preview_range_m": [1.5, 30.0],
        "preview_video": "rgb_depth.mp4",
        "render_var": "DistanceToImagePlaneSD",
        "shape": [2, 3],
    }
    assert Path(outputs["depth"]["manifest"]).is_file()
    assert Path(outputs["semantic"]["labels"]).is_file()


def test_ovrtx_video_failure_preserves_an_existing_destination(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "rgb.mp4"
    output.write_bytes(b"previous-video")

    class FailingWriter:
        def __init__(self, output_path):
            self.output_path = Path(output_path)

        def append_data(self, frame):
            self.output_path.write_bytes(b"partial-video")
            raise RuntimeError("encoder failed")

        def close(self):
            pass

    monkeypatch.setattr(
        render_ovrtx,
        "_video_writer",
        lambda output_path, fps: FailingWriter(output_path),
    )

    with pytest.raises(RuntimeError, match="encoder failed"):
        render_ovrtx._write_video_and_aovs(
            iter(({"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},)),
            output,
            12,
            (),
            None,
        )

    assert output.read_bytes() == b"previous-video"
    assert not list(tmp_path.glob(".rgb.part-*.mp4"))


def test_ovrtx_aov_failure_preserves_the_complete_previous_set(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "rgb.mp4"
    output.write_bytes(b"previous-rgb")
    aov_dir = tmp_path / "aovs"
    frames_dir = aov_dir / "rgb_depth_frames"
    frames_dir.mkdir(parents=True)
    (aov_dir / "rgb_depth.mp4").write_bytes(b"previous-preview")
    (aov_dir / "rgb_depth_manifest.json").write_text('{"previous": true}\n')
    np.save(frames_dir / "frame_000000.npy", np.asarray([42], dtype=np.float32))

    class SelectivelyFailingWriter:
        def __init__(self, output_path):
            self.output_path = Path(output_path)

        def append_data(self, frame):
            if self.output_path.name == "rgb_depth.mp4":
                self.output_path.write_bytes(b"partial-preview")
                raise RuntimeError("AOV encoder failed")

        def close(self):
            self.output_path.write_bytes(b"staged-video")

    monkeypatch.setattr(
        render_ovrtx,
        "_video_writer",
        lambda output_path, fps: SelectivelyFailingWriter(output_path),
    )
    frame = {
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth": np.ones((2, 2), dtype=np.float32),
    }

    with pytest.raises(RuntimeError, match="AOV encoder failed"):
        render_ovrtx._write_video_and_aovs(
            iter((frame,)),
            output,
            12,
            ("depth",),
            aov_dir,
        )

    assert output.read_bytes() == b"previous-rgb"
    assert (aov_dir / "rgb_depth.mp4").read_bytes() == b"previous-preview"
    assert (aov_dir / "rgb_depth_manifest.json").read_text() == '{"previous": true}\n'
    np.testing.assert_array_equal(
        np.load(frames_dir / "frame_000000.npy"),
        np.asarray([42], dtype=np.float32),
    )
    assert not list(tmp_path.glob(".aovs.part-*"))


def test_ovrtx_aov_publication_rolls_back_an_interrupted_swap(
    monkeypatch,
    tmp_path: Path,
):
    output_dir = tmp_path / "aovs"
    output_dir.mkdir()
    (output_dir / "rgb_depth.mp4").write_bytes(b"previous-preview")
    (output_dir / "rgb_depth_manifest.json").write_bytes(b"previous-manifest")
    staging = render_ovrtx._new_aov_staging_directory(output_dir)
    (staging / "rgb_depth.mp4").write_bytes(b"new-preview")
    (staging / "rgb_depth_manifest.json").write_bytes(b"new-manifest")

    real_replace = render_ovrtx.os.replace

    def fail_second_publication(source, destination):
        source = Path(source)
        if source.parent == staging and source.name == "rgb_depth_manifest.json":
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(render_ovrtx.os, "replace", fail_second_publication)

    with pytest.raises(OSError, match="simulated publication failure"):
        render_ovrtx._publish_staged_aovs(staging, output_dir, "rgb")

    assert (output_dir / "rgb_depth.mp4").read_bytes() == b"previous-preview"
    assert (output_dir / "rgb_depth_manifest.json").read_bytes() == b"previous-manifest"


def test_aov_video_writer_rejects_an_empty_frame_source(monkeypatch, tmp_path: Path):
    writer = SimpleNamespace(append_data=lambda frame: None, close=lambda: None)
    monkeypatch.setattr(render_ovrtx, "_video_writer", lambda output_path, fps: writer)

    with pytest.raises(RuntimeError, match="produced no frames"):
        render_ovrtx._write_video_and_aovs(
            iter(()),
            tmp_path / "rgb.mp4",
            12,
            ("depth",),
            tmp_path / "aovs",
        )


def test_ovrtx_video_timing_separates_remote_production_from_local_encode(
    monkeypatch,
    tmp_path: Path,
):
    clock = [0.0]

    def frames():
        clock[0] += 2.0
        try:
            yield {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
        finally:
            clock[0] += 3.0

    class TimedWriter:
        def __init__(self, output_path):
            self.output_path = Path(output_path)

        def append_data(self, frame):
            clock[0] += 0.5

        def close(self):
            clock[0] += 0.25
            self.output_path.write_bytes(b"encoded")

    monkeypatch.setattr(render_ovrtx.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        render_ovrtx,
        "_video_writer",
        lambda output_path, fps: TimedWriter(output_path),
    )
    timing = {}

    _, frame_count, encode_seconds, _ = render_ovrtx._write_video_and_aovs(
        frames(),
        tmp_path / "rgb.mp4",
        12,
        (),
        None,
        timings=timing,
        expected_frame_count=1,
    )

    assert frame_count == 1
    assert encode_seconds == pytest.approx(0.75)
    assert timing["remote_production_seconds"] == pytest.approx(2.0)
    assert timing["remote_cleanup_seconds"] == pytest.approx(3.0)
    assert timing["render_seconds"] == pytest.approx(2.0)
    assert timing["local_encode_seconds"] == pytest.approx(0.75)
    assert timing["roundtrip_seconds"] == pytest.approx(5.75)


def test_ovrtx_record_result_uses_render_and_roundtrip_timing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        render_ovrtx,
        "_frame_source",
        lambda *args, **kwargs: iter(()),
    )

    def write_video(
        frames,
        output_path,
        fps,
        aovs,
        output_dir,
        timings,
        expected_frame_count,
    ):
        assert expected_frame_count == 24
        timings.update(
            render_seconds=2.0,
            remote_production_seconds=2.0,
            remote_cleanup_seconds=0.5,
            local_encode_seconds=1.25,
            aov_publication_seconds=0.25,
            publication_seconds=0.25,
        )
        return Path(output_path), 4, 1.25, {}

    monkeypatch.setattr(render_ovrtx, "_write_video_and_aovs", write_video)
    ticks = iter((10.0, 20.0))
    monkeypatch.setattr(render_ovrtx.time, "perf_counter", lambda: next(ticks))

    result = render_ovrtx.record_video(
        tmp_path / "rgb.mp4",
        duration=1.0,
        fps=24,
        width=16,
        height=9,
    )

    assert result["video_fps"] == 24
    assert result["render_seconds"] == 2.0
    assert result["remote_production_seconds"] == 2.0
    assert result["remote_cleanup_seconds"] == 0.5
    assert result["encode_seconds"] == 1.25
    assert result["local_encode_seconds"] == 1.25
    assert result["aov_publication_seconds"] == 0.25
    assert result["publication_seconds"] == 0.25
    assert result["render_fps"] == 2.0
    assert result["roundtrip_seconds"] == 10.0
    assert result["roundtrip_fps"] == 0.4


@pytest.mark.parametrize(
    ("warmup", "effective_warmup"),
    ((0, 1), (2, 2)),
)
def test_ovrtx_benchmark_excludes_setup_and_warmup(
    monkeypatch,
    warmup,
    effective_warmup,
):
    requested_times = []

    def frame_source(times, *args, **kwargs):
        requested_times.extend(times)
        for index in range(len(times)):
            yield {"rgb": np.full((2, 2, 3), index, dtype=np.uint8)}

    monkeypatch.setattr(render_ovrtx, "_frame_source", frame_source)
    ticks = iter((0.0, 10.0, 12.0, 15.0))
    monkeypatch.setattr(render_ovrtx.time, "perf_counter", lambda: next(ticks))

    result = render_ovrtx.benchmark(
        frame_count=3,
        width=2,
        height=2,
        fps=10,
        warmup=warmup,
        samples=1,
    )

    assert len(requested_times) == effective_warmup + 3
    assert result["warmup"] == warmup
    assert result["effective_warmup"] == effective_warmup
    assert result["setup_warmup_seconds"] == 10.0
    assert result["render_seconds"] == 2.0
    assert result["render_fps"] == 1.5
    assert result["roundtrip_seconds"] == 15.0
    assert result["roundtrip_fps"] == 0.2


def test_ovrtx_remote_setup_uses_official_locked_wheels_without_archive_shims():
    script = render_ovrtx._remote_install_script("0123456789abcdef")

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "sync --directory" in script
    assert "--frozen --no-dev" in script
    assert "import ovrtx" in script
    assert "import ovstage" in script
    assert "libOpenGL.so.0" in script
    assert "libopengl0" in script
    assert 'mv "$incoming/source" "$root/source"' in script
    assert 'mv "$incoming/runtime" "$root/runtime"' in script
    assert "manylinux_2_35" not in script
    assert "site-packages" not in script


def test_ovrtx_failed_job_upload_removes_the_created_remote_directory(
    monkeypatch,
    tmp_path: Path,
):
    cleanup_calls = []
    monkeypatch.setattr(render_ovrtx, "run_ssh", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        render_ovrtx,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(23, "rsync")),
    )
    monkeypatch.setattr(
        render_ovrtx,
        "_cleanup_remote_job",
        lambda host, job_id: cleanup_calls.append((host, job_id)),
    )

    with pytest.raises(subprocess.CalledProcessError):
        render_ovrtx._upload_job("render-host", tmp_path, "0123456789abcdef")

    assert cleanup_calls == [("render-host", "0123456789abcdef")]


def test_ovrtx_remote_command_uses_the_packaged_worker_module():
    command = render_ovrtx._remote_worker_command(".jobs/example", loop=False)

    assert "PYTHONPATH=" in command
    assert "-m robotic_renderers.backends.ovrtx __worker" in command


def test_ovrtx_rejects_oversized_aov_payload_before_allocating(monkeypatch):
    limit = render_ovrtx._aov_payload_limit(4, 3)
    stdout = io.BytesIO(render_ovrtx.STREAM_MAGIC_AOV + struct.pack("!I", limit + 1))
    process = type(
        "FakeProcess",
        (),
        {"stdout": stdout, "stderr": io.BytesIO()},
    )()
    monkeypatch.setattr(render_ovrtx.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        render_ovrtx,
        "_terminate_remote_worker",
        lambda host, remote_job: None,
    )
    monkeypatch.setattr(render_ovrtx, "_terminate_and_reap", lambda process: None)

    frames = render_ovrtx._remote_stream_frames(
        "render-host",
        ".jobs/example",
        4,
        3,
        False,
        ("depth",),
    )
    with pytest.raises(RuntimeError, match="outside the allowed range"):
        next(frames)


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


def test_worker_advances_ovstage_time_and_renders_at_the_same_ordinal(monkeypatch):
    updates = []
    advances = []
    waits = []
    steps = []

    def update_from_usd_time(stage, *, ordinal, time_code):
        updates.append((stage, ordinal, time_code))

    def advance_write_floor(ordinal, scope):
        advances.append((ordinal, scope))
        return SimpleNamespace(wait=lambda: waits.append(ordinal))

    def step(*, render_products, delta_time, ordinal):
        steps.append((render_products, delta_time, ordinal))
        mapping = SimpleNamespace(unmap=lambda: None)
        render_var = SimpleNamespace(map=lambda device: mapping)
        frame = SimpleNamespace(render_vars={"LdrColor": render_var})
        return {"/Render/Camera": SimpleNamespace(frames=[frame])}

    fake_ovrtx = SimpleNamespace(Device=SimpleNamespace(CPU="cpu"))
    fake_ovstage = SimpleNamespace(
        Scope=SimpleNamespace(ALL="all"),
        population=SimpleNamespace(update_from_usd_time=update_from_usd_time),
    )
    monkeypatch.setitem(sys.modules, "ovrtx", fake_ovrtx)
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    monkeypatch.setattr(
        np,
        "from_dlpack",
        lambda mapping: np.zeros((2, 3, 4), dtype=np.uint8),
    )
    renderer = SimpleNamespace(step=step)
    stage = SimpleNamespace(advance_write_floor=advance_write_floor)
    bundle = {
        "animated_in_usd": True,
        "fps": 24,
        "height": 2,
        "width": 3,
        "render_product": "/Render/Camera",
    }

    render_ovrtx._worker_frame(
        renderer,
        bundle,
        {},
        index=11,
        stage=stage,
        ordinal=7,
    )

    assert updates == [(stage, 7, 0.5)]
    assert advances == [(7, "all")]
    assert waits == [7]
    assert steps == [({"/Render/Camera"}, 1.0 / 24.0, 7)]


def test_malformed_handshake_does_not_block_and_reaps_stubborn_ssh(monkeypatch):
    class FakeStderr:
        read_count = 0

        def read(self):
            self.read_count += 1
            raise AssertionError("stderr must not be read while the child is alive")

    class FakeProcess:
        def __init__(self):
            self.stdout = SimpleNamespace(readline=lambda *args: b"not-a-handshake\n")
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

    frames = render_ovrtx._remote_stream_frames("backup-render-host", ".jobs/test", 4, 3, False)
    with pytest.raises(RuntimeError, match="not-a-handshake"):
        next(frames)

    assert process.stderr.read_count == 0
    assert remote_terminations == [("backup-render-host", ".jobs/test")]
    assert process.terminate_count == 1
    assert process.kill_count == 1
    assert process.wait_timeouts == [5.0, None]
    assert process.poll() == -9
