import json
from pathlib import Path

import benchmark_fps


def test_benchmark_parser_defaults_to_full_resolution_g4():
    args = benchmark_fps.build_parser().parse_args([])

    assert args.host == "g4"
    assert (args.width, args.height) == (1280, 720)
    assert args.backends == ["opengl", "warp", "blender", "ovrtx"]
    assert args.samples == 32
    assert args.blender_workers[-3:] == [48, 64, 96]
    assert args.blender_cpu_threads is None
    assert args.blender_denoiser == "OPTIX"
    assert args.blender_persistent_data is True
    assert args.blender_numa_bind is True


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

    stage_path, render_product = benchmark_fps._write_ovrtx_multi_env_stage(bundle_path, 4)
    stage = stage_path.read_text()

    assert render_product == "/Render/MultiEnvironment"
    assert "int2 resolution = (2560, 1440)" in stage
    assert stage.count("prepend references") == 4
    assert stage.count("</World/Environment_") == 4
    assert "uint[] deviceIds = [0]" in stage


def test_markdown_report_ranks_best_render_fps():
    result = {
        "generated_at": "2026-07-15T00:00:00+00:00",
        "host": "g4",
        "hardware": {
            "hostname": "test-g4",
            "logical_cpus": 384,
            "architecture": "x86_64",
            "gpus": [{"name": "RTX PRO 6000"}],
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

    report = benchmark_fps.format_markdown(result)

    assert report.index("| 1 | warp") < report.index("| 2 | opengl")
    assert "1280×720 per environment" in report
    assert "500.00" in report
