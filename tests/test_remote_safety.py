import io
import struct
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_renderers.backends import mujoco_warp as render_mujoco_warp
from robotic_renderers.workflows import cosmos3 as render_cosmos3


def _minimal_warp_source(root: Path) -> None:
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text(f"# root {name} must not be uploaded\n")
    runtime = root / "runtime" / "warp"
    runtime.mkdir(parents=True)
    (runtime / "pyproject.toml").write_text("[project]\nname='warp-runtime'\n")
    (runtime / "uv.lock").write_text("version = 1\n")
    (runtime / "README.md").write_text("# Warp runtime\n")
    backend = root / "src" / "robotic_renderers" / "backends"
    backend.mkdir(parents=True)
    (root / "src" / "robotic_renderers" / "__init__.py").write_text("")
    (root / "src" / "robotic_renderers" / "scene.py").write_text("# scene.py\n")
    (backend / "mujoco_warp.py").write_text("# mujoco_warp.py\n")


def _minimal_installed_warp_package(package: Path) -> Path:
    backend = package / "backends"
    backend.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "scene.py").write_text("# scene.py\n")
    (backend / "mujoco_warp.py").write_text("# mujoco_warp.py\n")
    fr3 = package / "data" / "franka_fr3"
    fr3.mkdir(parents=True)
    (fr3 / "fr3.xml").write_text("<mujoco/>\n")
    runtime = package / "data" / "runtime" / "warp"
    runtime.mkdir(parents=True)
    (runtime / "pyproject.toml").write_text("[project]\nname='warp-runtime'\n")
    (runtime / "uv.lock").write_text("version = 1\n")
    (runtime / "README.md").write_text("# Warp runtime\n")
    return runtime


def test_remote_commands_use_noninteractive_keepalive_options():
    for module in (render_cosmos3, render_mujoco_warp):
        command = module._ssh_command("gpu-host", "true")

        assert command[:2] == ["ssh", "-o"]
        assert "BatchMode=yes" in command
        assert "ConnectTimeout=8" in command
        assert "ServerAliveInterval=15" in command
        assert "ServerAliveCountMax=3" in command
        assert command[-2:] == ["gpu-host", "true"]


def test_warp_bundle_stages_only_explicit_safe_sources(tmp_path: Path):
    root = tmp_path / "repository"
    root.mkdir()
    _minimal_warp_source(root)
    (root / "src" / "robotic_renderers" / ".env").write_text("TOKEN=secret\n")
    (root / "src" / "robotic_renderers" / "signing.key").write_text("secret\n")
    (root / "src" / "robotic_renderers" / "__pycache__").mkdir()
    (root / "src" / "robotic_renderers" / "__pycache__" / "module.pyc").write_bytes(b"x")
    (root / "runtime" / "other").mkdir()
    (root / "runtime" / "other" / "unrelated.toml").write_text("[runtime]\n")
    (root / "third_party" / "fr3").mkdir(parents=True)
    (root / "third_party" / "fr3" / "link0.obj").write_text("mesh\n")
    (root / "outputs").mkdir()
    (root / "outputs" / "frame.png").write_bytes(b"render")
    (root / "notes.txt").write_text("unrelated\n")
    (root / "credentials.json").write_text('{"token": "secret"}\n')

    staged = tmp_path / "staged"
    entries = render_mujoco_warp._stage_remote_bundle(staged, root)
    relative_entries = {path.as_posix() for path in entries}

    assert "src/robotic_renderers/backends/mujoco_warp.py" in relative_entries
    assert "src/robotic_renderers/__init__.py" in relative_entries
    assert "runtime/warp/pyproject.toml" in relative_entries
    assert "runtime/warp/uv.lock" in relative_entries
    assert "runtime/warp/README.md" in relative_entries
    assert "third_party/fr3/link0.obj" in relative_entries
    assert "pyproject.toml" not in relative_entries
    assert "uv.lock" not in relative_entries
    assert "runtime/other/unrelated.toml" not in relative_entries
    assert not (staged / "src" / "robotic_renderers" / ".env").exists()
    assert not (staged / "src" / "robotic_renderers" / "signing.key").exists()
    assert not (staged / "outputs").exists()
    assert not (staged / "notes.txt").exists()
    assert not (staged / "credentials.json").exists()


def test_warp_bundle_fingerprint_changes_with_source(tmp_path: Path):
    _minimal_warp_source(tmp_path)
    first = render_mujoco_warp._remote_bundle_fingerprint(tmp_path)
    (tmp_path / "src" / "robotic_renderers" / "scene.py").write_text(
        "# changed scene with a different size\n"
    )

    assert render_mujoco_warp._remote_bundle_fingerprint(tmp_path) != first


def test_installed_warp_bundle_reconstructs_remote_source_layout(tmp_path: Path):
    package = tmp_path / "site-packages" / "robotic_renderers"
    runtime = _minimal_installed_warp_package(package)
    (package / ".env").write_text("TOKEN=secret\n")
    unrelated = package / "data" / "runtime" / "ovrtx"
    unrelated.mkdir(parents=True)
    (unrelated / "uv.lock").write_text("version = 1\n")

    sources = render_mujoco_warp._installed_bundle_sources(package, runtime)
    entries = {path.as_posix() for path in sources}

    assert "src/robotic_renderers/__init__.py" in entries
    assert "src/robotic_renderers/backends/mujoco_warp.py" in entries
    assert "src/robotic_renderers/data/franka_fr3/fr3.xml" in entries
    assert "runtime/warp/pyproject.toml" in entries
    assert "runtime/warp/uv.lock" in entries
    assert "src/robotic_renderers/.env" not in entries
    assert "src/robotic_renderers/data/runtime/ovrtx/uv.lock" not in entries


def test_warp_worker_runs_from_isolated_remote_source():
    command = render_mujoco_warp._remote_shell_command(
        ["still", "--output", ".jobs/example/frame.png"]
    )

    assert "$HOME/.cache/robot_rendering/.venv/bin/python" in command
    assert "-m robotic_renderers.backends.mujoco_warp __worker" in command
    assert 'cd "$root"' in command
    assert 'PYTHONPATH="$project/src"' in command


def test_warp_setup_syncs_only_the_frozen_worker_runtime():
    script = render_mujoco_warp._remote_sync_script("source.upload-example")

    assert 'runtime="$incoming/runtime/warp"' in script
    assert 'test -f "$runtime/pyproject.toml"' in script
    assert 'test -f "$runtime/uv.lock"' in script
    assert '--directory "$runtime" --python 3.12 --frozen --no-dev' in script
    assert 'UV_PROJECT_ENVIRONMENT="$root/.venv"' in script
    assert 'PYTHONPATH="$incoming/src"' in script
    assert "from pxr import Sdf" in script
    assert 'cd "$incoming"' not in script


def test_warp_runtime_is_pinned_and_linux_only():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime" / "warp"
    project = tomllib.loads((runtime / "pyproject.toml").read_text())
    lock = tomllib.loads((runtime / "uv.lock").read_text())

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["project"]["dependencies"] == [
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
        "mujoco==3.11.0",
        "mujoco-warp==3.11.0",
        "numpy==2.5.1",
        "pillow==12.3.0",
        "usd-core==26.8",
        "warp-lang==1.15.0",
    ]
    assert lock["requires-python"] == "==3.12.*"
    assert lock["supported-markers"] == ["platform_machine == 'x86_64' and sys_platform == 'linux'"]


def _warp_stream_process(payload: bytes, return_code: int):
    process = SimpleNamespace(stdout=io.BytesIO(payload))
    process.poll = lambda: return_code
    process.wait = lambda timeout=None: return_code
    return process


def test_remote_warp_stream_rejects_nonzero_worker_exit(monkeypatch):
    frame = bytes(range(6))
    payload = (
        render_mujoco_warp.STREAM_MAGIC
        + struct.pack("!I", len(frame))
        + frame
        + struct.pack("!I", len(frame))
        + frame
    )
    process = _warp_stream_process(payload, return_code=7)
    reaped = []
    monkeypatch.setattr(render_mujoco_warp.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        render_mujoco_warp,
        "terminate_and_reap",
        lambda processes, grace_seconds: reaped.append((processes, grace_seconds)),
    )

    frames = render_mujoco_warp._remote_stream_frames("render-host", 2, 1, 2, 1.0, 1)
    with pytest.raises(RuntimeError, match="exited with 7"):
        list(frames)

    assert reaped == [({"warp": process}, 3.0)]


def test_remote_warp_stream_reports_nonzero_exit_before_handshake(monkeypatch):
    process = _warp_stream_process(b"", return_code=9)
    monkeypatch.setattr(render_mujoco_warp.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(render_mujoco_warp, "terminate_and_reap", lambda *args, **kwargs: None)

    frames = render_mujoco_warp._remote_stream_frames("render-host", 2, 1, 2, 1.0, 1)
    with pytest.raises(RuntimeError, match="exited with 9 before its handshake"):
        next(frames)


def test_remote_warp_stream_requires_exact_finite_frame_count(monkeypatch):
    frame = bytes(range(6))
    payload = render_mujoco_warp.STREAM_MAGIC + struct.pack("!I", len(frame)) + frame
    process = _warp_stream_process(payload, return_code=0)
    monkeypatch.setattr(render_mujoco_warp.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(render_mujoco_warp, "terminate_and_reap", lambda *args, **kwargs: None)

    frames = render_mujoco_warp._remote_stream_frames("render-host", 2, 1, 2, 1.0, 1)
    with pytest.raises(RuntimeError, match="produced 1 frames; expected 2"):
        list(frames)


def test_warp_recording_separates_render_and_encode_timing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        render_mujoco_warp,
        "create_renderer",
        lambda *args, **kwargs: {"device": "cpu"},
    )
    monkeypatch.setattr(
        render_mujoco_warp,
        "render_context_frame",
        lambda context, time_s: np.zeros((1, 2, 3), dtype=np.uint8),
    )
    timestamps = iter((0.0, 0.2, 0.2, 0.5))
    monkeypatch.setattr(render_mujoco_warp.time, "perf_counter", lambda: next(timestamps))

    def write_video(frames, output_path, fps):
        assert len(list(frames)) == 2
        return Path(output_path), 2, 1.5

    monkeypatch.setattr(render_mujoco_warp, "write_video", write_video)

    result = render_mujoco_warp.record_video_local(
        tmp_path / "warp.mp4",
        duration=1.0,
        fps=2,
        width=2,
        height=1,
        supersample=1,
    )

    assert result["seconds"] == pytest.approx(1.5)
    assert result["render_seconds"] == pytest.approx(0.5)
    assert result["encode_seconds"] == pytest.approx(1.0)
    assert result["render_fps"] == pytest.approx(4.0)
    assert result["video_fps"] == 2


def test_remote_warp_recording_preserves_worker_timing(tmp_path: Path, monkeypatch):
    worker_result = {
        "backend": "warp",
        "frames": 2,
        "seconds": 1.5,
        "render_seconds": 0.5,
        "encode_seconds": 1.0,
        "render_fps": 4.0,
        "video_fps": 2,
    }
    monkeypatch.setattr(render_mujoco_warp, "setup_remote", lambda host: None)
    monkeypatch.setattr(
        render_mujoco_warp,
        "_run_remote_worker",
        lambda host, arguments: worker_result,
    )
    monkeypatch.setattr(
        render_mujoco_warp,
        "_copy_remote_output",
        lambda host, remote_path, output_path: Path(output_path),
    )
    monkeypatch.setattr(render_mujoco_warp, "_cleanup_remote_job", lambda *args: None)

    result = render_mujoco_warp.record_video(
        tmp_path / "remote-warp.mp4",
        duration=1.0,
        fps=2,
        width=2,
        height=1,
        supersample=1,
        host="render-host",
        local=False,
    )

    for field in ("render_seconds", "encode_seconds", "render_fps", "video_fps"):
        assert result[field] == worker_result[field]
