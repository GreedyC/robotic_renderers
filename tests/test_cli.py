from pathlib import Path
from types import SimpleNamespace

import pytest

from robotic_renderers import cli

EXPECTED_BACKENDS = ("opengl", "warp", "blender", "ovrtx", "mitsuba")


def _recording_function(calls, name, result=None):
    def record(*args, **kwargs):
        calls.append((name, args, kwargs))
        return result if result is not None else {"backend": name}

    return record


def test_status_probes_all_five_backends(monkeypatch):
    calls = []
    modules = {
        backend: SimpleNamespace(
            backend_status=_recording_function(
                calls,
                backend,
                {"backend": backend, "available": True},
            )
        )
        for backend in EXPECTED_BACKENDS
    }
    monkeypatch.setattr(cli, "_module", modules.__getitem__)

    result = cli.run_status()

    assert tuple(cli.BACKEND_MODULES) == EXPECTED_BACKENDS
    assert tuple(result) == EXPECTED_BACKENDS
    assert calls == [
        ("opengl", (), {}),
        ("warp", ("render-host",), {}),
        ("blender", (), {}),
        ("ovrtx", ("render-host",), {}),
        ("mitsuba", ("render-host",), {}),
    ]


def test_registry_uses_installable_package_module_paths():
    assert cli.BACKEND_MODULES == {
        "opengl": "robotic_renderers.backends.mujoco_opengl",
        "warp": "robotic_renderers.backends.mujoco_warp",
        "blender": "robotic_renderers.backends.blender",
        "ovrtx": "robotic_renderers.backends.ovrtx",
        "mitsuba": "robotic_renderers.backends.mitsuba",
    }


@pytest.mark.parametrize(
    ("command", "expected_mode"),
    (
        ("still", "pathtraced"),
        ("live", "realtime"),
        ("record", "pathtraced"),
        ("benchmark", "pathtraced"),
    ),
)
def test_render_mode_defaults_match_command_usage(command, expected_mode):
    args = cli.build_parser().parse_args([command, "--backend", "ovrtx"])

    assert args.host == "render-host"
    assert (args.width, args.height) == (1280, 720)
    assert args.render_mode == expected_mode
    assert args.aovs == []
    assert args.aov_output_dir is None
    if command == "record":
        assert args.duration == 30.0


def test_parser_collects_repeatable_ovrtx_aovs(tmp_path: Path):
    output_dir = tmp_path / "aovs"
    args = cli.build_parser().parse_args(
        [
            "still",
            "--backend",
            "ovrtx",
            "--render-mode",
            "realtime",
            "--aov",
            "depth",
            "--aov",
            "normals",
            "--aov-output-dir",
            str(output_dir),
        ]
    )

    assert args.render_mode == "realtime"
    assert args.aovs == ["depth", "normals"]
    assert args.aov_output_dir == output_dir


@pytest.mark.parametrize(
    "arguments",
    (
        ["still", "--backend", "opengl", "--width", "0"],
        ["record", "--backend", "opengl", "--duration", "-1"],
        ["record", "--backend", "opengl", "--fps", "0"],
        ["benchmark", "--backend", "opengl", "--warmup", "-1"],
    ),
)
def test_parser_rejects_invalid_dimensions_and_timing(arguments):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_status_can_probe_one_backend_and_forward_host(monkeypatch):
    calls = []
    module = SimpleNamespace(
        backend_status=_recording_function(
            calls,
            "mitsuba",
            {"backend": "mitsuba", "available": True},
        )
    )
    monkeypatch.setattr(cli, "_module", lambda backend: module)
    args = cli.build_parser().parse_args(
        ["status", "--backend", "mitsuba", "--host", "render-host"]
    )

    result = cli.run_status(args)

    assert tuple(result) == ("mitsuba",)
    assert calls == [("mitsuba", ("render-host",), {})]


def test_mitsuba_setup_uses_remote_host_and_force(monkeypatch):
    calls = []
    module = SimpleNamespace(setup_remote=_recording_function(calls, "setup"))
    monkeypatch.setattr(cli, "_module", lambda backend: module)
    args = cli.build_parser().parse_args(
        ["setup", "--backend", "mitsuba", "--host", "render-host", "--force"]
    )

    cli.run_setup(args)

    assert calls == [("setup", ("render-host",), {"force": True})]


def test_mitsuba_render_commands_forward_supported_remote_options(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    module = SimpleNamespace(
        render_still=_recording_function(calls, "still"),
        render_live=_recording_function(calls, "live"),
        record_video=_recording_function(calls, "record"),
        benchmark=_recording_function(calls, "benchmark"),
    )
    monkeypatch.setattr(cli, "_module", lambda backend: module)
    still_output = tmp_path / "still.png"
    video_output = tmp_path / "video.mp4"

    still_args = cli.build_parser().parse_args(
        [
            "still",
            "--backend",
            "mitsuba",
            "--output",
            str(still_output),
            "--width",
            "320",
            "--height",
            "180",
            "--time",
            "2.5",
            "--samples",
            "17",
            "--host",
            "render-host",
        ]
    )
    live_args = cli.build_parser().parse_args(
        [
            "live",
            "--backend",
            "mitsuba",
            "--width",
            "321",
            "--height",
            "181",
            "--fps",
            "12",
            "--duration",
            "1.5",
            "--samples",
            "18",
            "--host",
            "render-host",
        ]
    )
    record_args = cli.build_parser().parse_args(
        [
            "record",
            "--backend",
            "mitsuba",
            "--output",
            str(video_output),
            "--duration",
            "2.5",
            "--fps",
            "13",
            "--width",
            "322",
            "--height",
            "182",
            "--samples",
            "19",
            "--host",
            "render-host",
        ]
    )
    benchmark_args = cli.build_parser().parse_args(
        [
            "benchmark",
            "--backend",
            "mitsuba",
            "--frames",
            "5",
            "--width",
            "323",
            "--height",
            "183",
            "--fps",
            "14",
            "--warmup",
            "2",
            "--samples",
            "20",
            "--host",
            "render-host",
        ]
    )

    cli.run_still(still_args)
    cli.run_live(live_args)
    cli.run_record(record_args)
    benchmark_result = cli.run_benchmark(benchmark_args)

    assert calls == [
        (
            "still",
            (),
            {
                "output_path": still_output,
                "width": 320,
                "height": 180,
                "time_s": 2.5,
                "samples": 17,
                "host": "render-host",
            },
        ),
        (
            "live",
            (),
            {
                "width": 321,
                "height": 181,
                "fps": 12,
                "duration": 1.5,
                "samples": 18,
                "host": "render-host",
            },
        ),
        (
            "record",
            (),
            {
                "output_path": video_output,
                "duration": 2.5,
                "fps": 13,
                "width": 322,
                "height": 182,
                "samples": 19,
                "host": "render-host",
            },
        ),
        (
            "benchmark",
            (),
            {
                "frame_count": 5,
                "width": 323,
                "height": 183,
                "fps": 14,
                "warmup": 2,
                "samples": 20,
                "host": "render-host",
            },
        ),
    ]
    assert benchmark_result["results"]["mitsuba"] == {"backend": "benchmark"}


@pytest.mark.parametrize(
    "arguments",
    (
        ["still", "--backend", "mitsuba", "--aov", "depth"],
        ["still", "--backend", "blender", "--render-mode", "realtime"],
        ["still", "--backend", "opengl", "--samples", "8"],
        ["still", "--backend", "opengl", "--supersample", "2"],
        ["still", "--backend", "warp", "--host", "render-host"],
        ["record", "--backend", "mitsuba", "--keep-frames"],
        ["record", "--backend", "blender", "--keep-components"],
        [
            "record",
            "--backend",
            "all",
            "--aov",
            "depth",
            "--output",
            "comparison.mp4",
        ],
    ),
)
def test_cli_rejects_backend_options_that_would_be_ignored(arguments, capsys):
    assert cli.main(arguments) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_requires_aov_when_aov_output_directory_is_explicit(capsys):
    assert (
        cli.main(
            [
                "still",
                "--backend",
                "ovrtx",
                "--aov-output-dir",
                "outputs/aovs",
            ]
        )
        == 1
    )
    assert "--aov-output-dir requires" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "function_name", "expected_common"),
    (
        (
            "still",
            "render_still",
            lambda output: {
                "output_path": output,
                "width": 320,
                "height": 180,
                "time_s": 2.5,
                "samples": 17,
                "host": "render-host",
            },
        ),
        (
            "live",
            "render_live",
            lambda output: {
                "width": 320,
                "height": 180,
                "fps": 12,
                "duration": 1.5,
                "samples": 17,
                "host": "render-host",
            },
        ),
        (
            "record",
            "record_video",
            lambda output: {
                "output_path": output,
                "duration": 1.5,
                "fps": 12,
                "width": 320,
                "height": 180,
                "samples": 17,
                "host": "render-host",
            },
        ),
        (
            "benchmark",
            "benchmark",
            lambda output: {
                "frame_count": 5,
                "width": 320,
                "height": 180,
                "fps": 12,
                "warmup": 2,
                "samples": 17,
                "host": "render-host",
            },
        ),
    ),
)
def test_ovrtx_render_commands_forward_mode_and_aovs(
    monkeypatch,
    tmp_path: Path,
    command,
    function_name,
    expected_common,
):
    calls = []
    module = SimpleNamespace(**{function_name: _recording_function(calls, function_name)})
    monkeypatch.setattr(cli, "_module", lambda backend: module)
    primary_output = tmp_path / ("result.json" if command == "benchmark" else "result")
    aov_output_dir = tmp_path / "auxiliary"
    arguments = [
        command,
        "--backend",
        "ovrtx",
        "--width",
        "320",
        "--height",
        "180",
        "--samples",
        "17",
        "--host",
        "render-host",
        "--render-mode",
        "realtime",
        "--aov",
        "depth",
        "--aov",
        "albedo",
        "--aov-output-dir",
        str(aov_output_dir),
    ]
    if command == "still":
        arguments.extend(("--time", "2.5", "--output", str(primary_output)))
    elif command == "live":
        arguments.extend(("--fps", "12", "--duration", "1.5"))
    elif command == "record":
        arguments.extend(("--fps", "12", "--duration", "1.5", "--output", str(primary_output)))
    else:
        arguments.extend(
            (
                "--fps",
                "12",
                "--frames",
                "5",
                "--warmup",
                "2",
                "--output",
                str(primary_output),
            )
        )
    args = cli.build_parser().parse_args(arguments)

    if command == "still":
        cli.run_still(args)
    elif command == "live":
        cli.run_live(args)
    elif command == "record":
        cli.run_record(args)
    else:
        cli.run_benchmark(args)

    assert calls == [
        (
            function_name,
            (),
            {
                **expected_common(primary_output),
                "render_mode": "realtime",
                "aovs": ("depth", "albedo"),
                "aov_output_dir": aov_output_dir,
            },
        )
    ]


def test_benchmark_reports_total_failure_for_cli_exit(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(cli, "_module", lambda backend: SimpleNamespace(benchmark=fail))
    args = cli.build_parser().parse_args(["benchmark", "--backend", "opengl"])

    result = cli.run_benchmark(args)

    assert result["success"] is False
    assert result["partial"] is False
    assert result["failed_backends"] == ["opengl"]


def test_benchmark_keeps_partial_all_backend_results(monkeypatch):
    def module_for(backend):
        if backend == "mitsuba":
            return SimpleNamespace(
                benchmark=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
            )
        return SimpleNamespace(benchmark=lambda **kwargs: {"backend": backend})

    monkeypatch.setattr(cli, "_module", module_for)
    args = cli.build_parser().parse_args(["benchmark", "--backend", "all"])

    result = cli.run_benchmark(args)

    assert result["success"] is True
    assert result["partial"] is True
    assert result["failed_backends"] == ["mitsuba"]
