import json
from pathlib import Path

import numpy as np
import pytest

from robotic_renderers.backends import mitsuba, mujoco_warp, ovrtx
from robotic_renderers.benchmarks import throughput


def test_benchmark_bundle_destination_is_confined_to_build_root(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(throughput, "ROOT", tmp_path)
    valid = tmp_path / "build" / "fps_benchmark_trial"

    assert throughput._validated_benchmark_destination(valid) == valid.resolve()

    for invalid in (
        tmp_path,
        tmp_path / "build",
        tmp_path / "build" / "unmarked",
        tmp_path.parent / "fps_benchmark_escape",
    ):
        with pytest.raises(ValueError, match="fps_benchmark"):
            throughput._validated_benchmark_destination(invalid)


def test_benchmark_parser_defaults_are_portable_and_full_resolution():
    args = throughput.build_parser().parse_args([])

    assert args.host == "render-host"
    assert (args.width, args.height) == (1280, 720)
    assert args.backends == ["opengl", "warp", "blender", "ovrtx", "mitsuba"]
    assert args.samples == 32
    assert args.blender_workers[-3:] == [48, 64, 96]
    assert args.blender_cpu_threads is None
    assert args.blender_denoiser == "OPTIX"
    assert args.blender_persistent_data is True
    assert args.blender_numa_bind is True
    assert args.mitsuba_workers == [1, 2, 4, 8, 16, 32]
    assert args.output == Path("outputs/fps_benchmark.json")
    assert args.report == Path("outputs/fps_benchmark.md")


def test_ovrtx_multi_environment_stage_preserves_per_environment_resolution(
    tmp_path: Path,
):
    bundle_path = tmp_path / "ovrtx_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "scene": "scene.usdc",
                "frames": 12,
                "fps": 30,
                "width": 1280,
                "height": 720,
                "samples": 32,
            }
        )
    )

    stage_path, render_product = throughput._write_ovrtx_multi_env_stage(bundle_path, 4)
    stage = stage_path.read_text()

    assert render_product == "/Render/MultiEnvironment"
    assert "int2 resolution = (2560, 1440)" in stage
    assert stage.count("prepend references") == 4
    assert stage.count("</World/Environment_") == 4
    assert "uint[] deviceIds = [0]" in stage


def test_mitsuba_workers_are_distributed_round_robin_across_selected_gpus():
    assert throughput._mitsuba_worker_gpus(7, [1, 3, 5]) == [1, 3, 5, 1, 3, 5, 1]

    with pytest.raises(ValueError, match="At least one GPU"):
        throughput._mitsuba_worker_gpus(1, [])


def test_mitsuba_guided_denoiser_receives_rgb_albedo_and_normals():
    image = np.arange(2 * 3 * 9, dtype=np.float32).reshape(2, 3, 9)
    received = {}

    def denoiser(rgb, albedo, normals):
        received["rgb"] = rgb
        received["albedo"] = albedo
        received["normals"] = normals
        return rgb + albedo + normals

    denoised = throughput._mitsuba_denoise_rgb(image, denoiser)

    np.testing.assert_array_equal(received["rgb"], image[:, :, :3])
    np.testing.assert_array_equal(received["albedo"], image[:, :, 3:6])
    np.testing.assert_array_equal(received["normals"], image[:, :, 6:9])
    np.testing.assert_array_equal(
        denoised,
        image[:, :, :3] + image[:, :, 3:6] + image[:, :, 6:9],
    )
    np.testing.assert_array_equal(
        throughput._mitsuba_denoise_rgb(image, None),
        image[:, :, :3],
    )


def test_mitsuba_summary_reports_render_readback_and_throughput_fps():
    results = [
        {
            "worker": 0,
            "gpu": 3,
            "render_seconds": 2.0,
            "readback_seconds": 0.5,
            "output_shape": [720, 1280, 3],
            "checksum": 11,
        },
        {
            "worker": 1,
            "gpu": 1,
            "render_seconds": 3.0,
            "readback_seconds": 0.6,
            "output_shape": [720, 1280, 3],
            "checksum": 22,
        },
        {
            "worker": 2,
            "gpu": 3,
            "render_seconds": 2.5,
            "readback_seconds": 0.4,
            "output_shape": [720, 1280, 3],
            "checksum": 33,
        },
    ]

    summary = throughput._summarize_mitsuba_results(results, workers=3, frames=4, wall_seconds=4.25)

    assert summary["worker_gpus"] == [3, 1, 3]
    assert summary["gpus_used"] == [1, 3]
    assert summary["total_frames"] == 12
    assert summary["render_fps"] == pytest.approx(4.0)
    assert summary["readback_fps"] == pytest.approx(20.0)
    assert summary["throughput_fps"] == pytest.approx(12 / 3.6)
    assert summary["wall_seconds"] == 4.25
    assert summary["checksums"] == [11, 22, 33]


def test_setup_remote_sets_up_each_runtime(monkeypatch):
    monkeypatch.setattr(
        mujoco_warp,
        "setup_remote",
        lambda host, force: {"host": host, "force": force},
    )
    monkeypatch.setattr(ovrtx, "setup_remote", lambda host: {"backend": "ovrtx", "host": host})
    monkeypatch.setattr(
        mitsuba,
        "setup_remote",
        lambda host: {"backend": "mitsuba", "host": host},
    )
    monkeypatch.setattr(
        throughput,
        "_setup_blender",
        lambda host: {"backend": "blender", "host": host},
    )

    setup = throughput.setup_remote("render-host")

    assert setup["warp_project"] == {"host": "render-host", "force": True}
    assert setup["ovrtx"] == {"backend": "ovrtx", "host": "render-host"}
    assert setup["mitsuba"] == {"backend": "mitsuba", "host": "render-host"}
    assert setup["blender"] == {"backend": "blender", "host": "render-host"}


@pytest.mark.parametrize(
    ("command_factory", "working_directory", "pythonpath"),
    [
        (throughput._remote_uv_command, ".cache/robot_rendering", "$PWD/source/src"),
        (throughput._remote_ovrtx_command, "code/robot_rendering_ovrtx", "$PWD/source"),
        (
            throughput._remote_mitsuba_command,
            "code/robot_rendering_mitsuba",
            "$PWD/source",
        ),
    ],
)
def test_remote_benchmark_commands_use_packaged_module(
    command_factory,
    working_directory,
    pythonpath,
):
    command = command_factory(["__hardware"])

    assert f'cd "$HOME/{working_directory}"' in command
    assert f'PYTHONPATH="{pythonpath}"' in command
    assert ".venv/bin/python -m robotic_renderers.benchmarks.throughput __hardware" in command


def test_run_suite_routes_mitsuba_sweep_to_remote_runtime(monkeypatch):
    args = throughput.build_parser().parse_args(
        [
            "--backends",
            "mitsuba",
            "--no-setup",
            "--gpus",
            "1,3",
            "--mitsuba-workers",
            "2,4",
            "--frames",
            "5",
        ]
    )
    commands = []

    def fake_remote_result(host, command, timeout=None):
        commands.append(command)
        if "__hardware" in command:
            return {
                "hostname": host,
                "platform": "linux",
                "architecture": "x86_64",
                "logical_cpus": 64,
                "gpus": [
                    {"index": index, "name": "RTX", "driver": "test", "memory_mib": 1}
                    for index in range(4)
                ],
            }
        if "__prepare" in command:
            return {
                "blender_manifest": "/tmp/blender.json",
                "ovrtx_bundle": "/tmp/ovrtx.json",
                "mitsuba_bundle": "/tmp/mitsuba.json",
            }
        workers = 2 if "--workers 2" in command else 4
        return {
            "workers": workers,
            "gpus_used": [1, 3],
            "total_frames": workers * 5,
            "render_fps": float(workers),
            "readback_fps": 100.0,
            "rgb_fps": float(workers),
            "throughput_fps": float(workers),
            "checksums": list(range(workers)),
        }

    monkeypatch.setattr(throughput, "_remote_result", fake_remote_result)

    result = throughput.run_suite(args)

    trials = result["results"]["mitsuba"]["trials"]
    assert [trial["workers"] for trial in trials] == [2, 4]
    assert result["results"]["mitsuba"]["best"]["workers"] == 4
    mitsuba_commands = [command for command in commands if "__mitsuba" in command]
    assert len(mitsuba_commands) == 2
    assert all(
        'cd "$HOME/code/robot_rendering_mitsuba"' in command
        and 'PYTHONPATH="$PWD/source"' in command
        and "--gpus 1,3" in command
        for command in mitsuba_commands
    )


def test_markdown_report_ranks_best_render_fps():
    result = {
        "generated_at": "2026-07-15T00:00:00+00:00",
        "host": "render-host",
        "hardware": {
            "hostname": "test-render-host",
            "logical_cpus": 384,
            "architecture": "x86_64",
            "gpus": [{"name": "Test RTX GPU"}],
        },
        "configuration": {
            "scene": "fr3_renderer_evaluation",
            "width": 1280,
            "height": 720,
            "path_tracing_samples": 32,
            "warmup": 3,
            "timed_steps_or_frames": 12,
        },
        "results": {
            "opengl": {
                "strategy": "multiprocess",
                "trials": [
                    {
                        "workers": 2,
                        "gpus_used": [0, 1],
                        "render_fps": 100.0,
                        "rgb_fps": 100.0,
                    }
                ],
                "best": {
                    "workers": 2,
                    "render_fps": 100.0,
                    "rgb_fps": 100.0,
                },
            },
            "warp": {
                "strategy": "batched",
                "trials": [
                    {
                        "batch_per_gpu": 4,
                        "gpu_processes": 2,
                        "total_environments": 8,
                        "gpus_used": [0, 1],
                        "render_fps": 500.0,
                        "rgb_fps": 400.0,
                    }
                ],
                "best": {
                    "batch_per_gpu": 4,
                    "gpu_processes": 2,
                    "render_fps": 500.0,
                    "rgb_fps": 400.0,
                },
            },
        },
    }

    report = throughput.format_markdown(result)

    assert report.index("| 1 | warp") < report.index("| 2 | opengl")
    assert "1280×720 per environment" in report
    assert "500.00" in report
    assert "Readback FPS" in report
