from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from robotic_renderers import cli as main
from robotic_renderers.media import write_video
from robotic_renderers.workflows import tiled as render_tiled
from robotic_renderers.workflows.tiled import (
    BACKEND_ORDER,
    BACKEND_TITLES,
    _backend_command,
    compose_tiled_video,
)


def test_record_parser_accepts_all_backends():
    args = main.build_parser().parse_args(["record", "--backend", "all"])

    assert args.backend == "all"
    assert args.keep_components is False


def test_tiled_compositor_labels_five_synchronized_videos(tmp_path: Path):
    assert tuple(main.BACKEND_MODULES) == BACKEND_ORDER
    assert "Mitsuba" in BACKEND_TITLES["mitsuba"]

    paths = {}
    for index, backend in enumerate(BACKEND_ORDER, start=1):
        path = tmp_path / f"{backend}.mp4"
        frame = np.full((36, 64, 3), index * 30, dtype=np.uint8)
        write_video((frame.copy() for _ in range(2)), path, fps=2)
        paths[backend] = path

    result = compose_tiled_video(paths, tmp_path / "tiled.mp4", 1.0, 2, 64, 36)
    reader = imageio.get_reader(result["output"])
    try:
        first = reader.get_data(0)
        second = reader.get_data(1)
    finally:
        reader.close()

    assert first.shape == (72, 192, 3)
    assert second.shape == first.shape
    assert result["frames"] == 2
    assert result["width"] == 192
    assert result["height"] == 72
    assert result["columns"] == 3
    assert result["rows"] == 2
    assert result["video_fps"] == 2


def test_record_all_forwards_selected_ovrtx_mode(monkeypatch, tmp_path: Path):
    calls = []

    def record_all_backends(**kwargs):
        calls.append(kwargs)
        return {"backend": "all"}

    monkeypatch.setattr(render_tiled, "record_all_backends", record_all_backends)
    output = tmp_path / "all.mp4"
    args = main.build_parser().parse_args(
        [
            "record",
            "--backend",
            "all",
            "--output",
            str(output),
            "--duration",
            "2",
            "--fps",
            "12",
            "--width",
            "320",
            "--height",
            "180",
            "--supersample",
            "3",
            "--samples",
            "19",
            "--host",
            "render-host",
            "--remote-warp",
            "--keep-components",
            "--render-mode",
            "realtime",
        ]
    )

    main.run_record(args)

    assert calls == [
        {
            "output_path": output,
            "duration": 2.0,
            "fps": 12,
            "width": 320,
            "height": 180,
            "supersample": 3,
            "samples": 19,
            "host": "render-host",
            "remote_warp": True,
            "keep_components": True,
            "render_mode": "realtime",
        }
    ]


def test_tiled_worker_commands_route_mode_only_to_ovrtx(tmp_path: Path):
    commands = {
        backend: _backend_command(
            backend,
            tmp_path / f"{backend}.mp4",
            2.0,
            12,
            320,
            180,
            3,
            19,
            "render-host",
            True,
            "realtime",
        )
        for backend in BACKEND_ORDER
    }

    assert commands["ovrtx"][-6:] == [
        "--samples",
        "19",
        "--host",
        "render-host",
        "--render-mode",
        "realtime",
    ]
    assert commands["mitsuba"][-4:] == [
        "--samples",
        "19",
        "--host",
        "render-host",
    ]
    assert all(
        "--render-mode" not in command
        for backend, command in commands.items()
        if backend != "ovrtx"
    )
    assert all(
        command[1:4] == ["-m", "robotic_renderers", "record"] for command in commands.values()
    )


def test_tiled_local_warp_command_does_not_pass_an_ignored_host(tmp_path: Path):
    local = _backend_command(
        "warp",
        tmp_path / "warp.mp4",
        2.0,
        12,
        320,
        180,
        3,
        19,
        "render-host",
        False,
        "pathtraced",
    )
    remote = _backend_command(
        "warp",
        tmp_path / "warp-remote.mp4",
        2.0,
        12,
        320,
        180,
        3,
        19,
        "render-host",
        True,
        "pathtraced",
    )

    assert "--host" not in local
    assert "--remote-warp" not in local
    assert remote[-3:] == ["--remote-warp", "--host", "render-host"]


def test_concurrent_workers_use_cache_and_inherit_callers_working_directory(
    monkeypatch,
    tmp_path: Path,
):
    popen_calls = []
    cache = tmp_path / "cache"

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return object()

    monkeypatch.setattr(render_tiled, "cache_directory", lambda: cache)
    monkeypatch.setattr(
        render_tiled,
        "_preflight",
        lambda host: {
            backend: {"backend": backend, "available": True} for backend in BACKEND_ORDER
        },
    )
    monkeypatch.setattr(render_tiled.subprocess, "Popen", popen)
    monkeypatch.setattr(
        render_tiled,
        "wait_fail_fast",
        lambda processes: {backend: 0 for backend in processes},
    )
    monkeypatch.setattr(
        render_tiled,
        "compose_tiled_video",
        lambda *args: {"output": str(tmp_path / "all.mp4"), "frames": 1},
    )

    render_tiled.record_all_backends(tmp_path / "all.mp4", duration=1.0, fps=1)

    assert (cache / "tiled").is_dir()
    assert len(popen_calls) == len(BACKEND_ORDER)
    assert all("cwd" not in kwargs for _, kwargs in popen_calls)
    assert all(command[1:4] == ["-m", "robotic_renderers", "record"] for command, _ in popen_calls)


def test_tiled_compositor_rejects_mismatched_frame_rate(tmp_path: Path):
    paths = {}
    for index, backend in enumerate(BACKEND_ORDER, start=1):
        path = tmp_path / f"{backend}.mp4"
        frame = np.full((36, 64, 3), index * 30, dtype=np.uint8)
        component_fps = 3 if backend == "mitsuba" else 2
        write_video((frame.copy() for _ in range(2)), path, fps=component_fps)
        paths[backend] = path

    with pytest.raises(ValueError, match="Mitsuba|mitsuba"):
        compose_tiled_video(paths, tmp_path / "tiled.mp4", 1.0, 2, 64, 36)
