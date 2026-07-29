"""Command-line app for the FR3 renderer evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from robotic_renderers.defaults import (
    DEFAULT_BENCHMARK_SAMPLES,
    DEFAULT_HEIGHT,
    DEFAULT_LIVE_SAMPLES,
    DEFAULT_PATH_TRACING_SAMPLES,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_RENDER_HOST,
    DEFAULT_VIDEO_FPS,
    DEFAULT_WIDTH,
    output_directory,
)
from robotic_renderers.registry import backend_modules, backend_names, backend_spec, load_backend

BACKEND_MODULES = backend_modules()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _module(backend: str):
    return load_backend(backend)


def _print_result(result) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _default_output(backend: str, extension: str) -> Path:
    return output_directory() / f"{backend}.{extension}"


def _add_backend(parser: argparse.ArgumentParser, allow_all: bool = False) -> None:
    choices = list(BACKEND_MODULES)
    if allow_all:
        choices.append("all")
    parser.add_argument("--backend", choices=choices, required=True)


def _add_dimensions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default=DEFAULT_RENDER_HOST,
        help="Linux GPU SSH host for remote Warp, ovrtx, or Mitsuba rendering",
    )
    parser.add_argument(
        "--remote-warp",
        action="store_true",
        help="Use Warp remotely over SSH instead of the local macOS CPU device",
    )


def _add_quality_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--supersample", type=_positive_int, default=2)
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=DEFAULT_PATH_TRACING_SAMPLES,
        help="Blender, ovrtx, or Mitsuba path-tracing samples",
    )


def _add_ovrtx_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--render-mode",
        choices=("pathtraced", "realtime"),
        help="ovrtx renderer mode (defaults to realtime for live, pathtraced otherwise)",
    )
    parser.add_argument(
        "--aov",
        dest="aovs",
        action="append",
        choices=("depth", "normals", "albedo", "semantic", "instance"),
        default=[],
        help="Additional ovrtx output; repeat to request multiple AOVs",
    )
    parser.add_argument(
        "--aov-output-dir",
        type=Path,
        help="Directory for ovrtx raw NumPy AOV data and PNG/MP4 previews",
    )


def _add_render_options(parser: argparse.ArgumentParser) -> None:
    _add_remote_options(parser)
    _add_quality_options(parser)
    _add_ovrtx_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one canonical Franka FR3 scene with five pipelines."
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Probe renderer installations and devices")
    status_parser.add_argument("--backend", choices=backend_names())
    status_parser.add_argument("--host", default=DEFAULT_RENDER_HOST)

    scene_parser = subparsers.add_parser("scene", help="Inspect or export the canonical MjSpec")
    scene_parser.add_argument("--export", type=Path, help="Write a portable MJCF bundle")
    _add_dimensions(scene_parser)

    setup_parser = subparsers.add_parser("setup", help="Prepare an external renderer")
    _add_backend(setup_parser)
    _add_remote_options(setup_parser)
    setup_parser.add_argument("--force", action="store_true")

    still_parser = subparsers.add_parser("still", help="Render one PNG")
    _add_backend(still_parser)
    _add_dimensions(still_parser)
    _add_render_options(still_parser)
    still_parser.add_argument("--time", type=float, default=1.0)
    still_parser.add_argument("--output", type=Path)
    still_parser.set_defaults(render_mode="pathtraced")

    live_parser = subparsers.add_parser("live", help="Open a live renderer view")
    _add_backend(live_parser)
    _add_dimensions(live_parser)
    _add_render_options(live_parser)
    live_parser.set_defaults(supersample=1, samples=DEFAULT_LIVE_SAMPLES)
    live_parser.set_defaults(render_mode="realtime")
    live_parser.add_argument("--fps", type=_positive_int, default=DEFAULT_VIDEO_FPS)
    live_parser.add_argument("--duration", type=_positive_float)

    record_parser = subparsers.add_parser("record", help="Record an MP4")
    _add_backend(record_parser, allow_all=True)
    _add_dimensions(record_parser)
    _add_render_options(record_parser)
    record_parser.add_argument("--fps", type=_positive_int, default=DEFAULT_VIDEO_FPS)
    record_parser.add_argument(
        "--duration",
        type=_positive_float,
        default=DEFAULT_RECORDING_DURATION,
    )
    record_parser.add_argument("--output", type=Path)
    record_parser.add_argument("--keep-frames", action="store_true")
    record_parser.set_defaults(render_mode="pathtraced")
    record_parser.add_argument(
        "--keep-components",
        action="store_true",
        help="Keep the five source MP4s next to an --backend all output",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Measure renderer throughput (encoding excluded)"
    )
    _add_backend(benchmark_parser, allow_all=True)
    _add_dimensions(benchmark_parser)
    _add_render_options(benchmark_parser)
    benchmark_parser.set_defaults(samples=DEFAULT_BENCHMARK_SAMPLES, supersample=1)
    benchmark_parser.set_defaults(render_mode="pathtraced")
    benchmark_parser.add_argument("--fps", type=_positive_int, default=DEFAULT_VIDEO_FPS)
    benchmark_parser.add_argument("--frames", type=_positive_int, default=10)
    benchmark_parser.add_argument("--warmup", type=_nonnegative_int, default=3)
    benchmark_parser.add_argument("--output", type=Path, help="Optional JSON result path")

    cosmos_parser = subparsers.add_parser(
        "cosmos", help="Turn simulated renders into photoreal video with Cosmos3-Super"
    )
    cosmos_actions = cosmos_parser.add_subparsers(dest="cosmos_action", required=True)

    cosmos_status = cosmos_actions.add_parser("status", help="Probe the remote runtime and server")
    cosmos_status.add_argument("--host", default=DEFAULT_RENDER_HOST)
    cosmos_status.add_argument("--port", type=_positive_int, default=8091)

    cosmos_setup = cosmos_actions.add_parser("setup", help="Install the pinned remote uv runtime")
    cosmos_setup.add_argument("--host", default=DEFAULT_RENDER_HOST)
    cosmos_setup.add_argument("--force", action="store_true")

    cosmos_start = cosmos_actions.add_parser(
        "start", help="Start the persistent 64B BF16 service on the remote GPU host"
    )
    cosmos_start.add_argument("--host", default=DEFAULT_RENDER_HOST)
    cosmos_start.add_argument("--port", type=_positive_int, default=8091)
    cosmos_start.add_argument("--no-wait", action="store_true")

    cosmos_stop = cosmos_actions.add_parser("stop", help="Stop the remote Cosmos service")
    cosmos_stop.add_argument("--host", default=DEFAULT_RENDER_HOST)

    cosmos_enhance = cosmos_actions.add_parser(
        "enhance", help="Render a source and generatively enhance it"
    )
    cosmos_enhance.add_argument("--backend", choices=backend_names(), default="blender")
    cosmos_enhance.add_argument(
        "--mode",
        choices=["i2v", "v2v", "depth", "seg", "depth-seg"],
        default="depth",
        help="Depth is the quality preset; I2V accepts a single rendered image",
    )
    cosmos_enhance.add_argument(
        "--input", type=Path, help="Use an existing rendered image/video instead of rendering"
    )
    cosmos_enhance.add_argument("--output", type=Path)
    cosmos_enhance.add_argument("--preview", type=Path)
    cosmos_enhance.add_argument("--host", default=DEFAULT_RENDER_HOST)
    cosmos_enhance.add_argument("--port", type=_positive_int, default=8091)
    cosmos_enhance.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    cosmos_enhance.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    cosmos_enhance.add_argument(
        "--frames",
        type=_positive_int,
        help="Default: 189 for I2V/V2V, 121 for transfer control",
    )
    cosmos_enhance.add_argument("--fps", type=_positive_int, default=24)
    cosmos_enhance.add_argument("--steps", type=_positive_int, default=50)
    cosmos_enhance.add_argument("--guidance", type=float)
    cosmos_enhance.add_argument("--flow-shift", type=float)
    cosmos_enhance.add_argument("--seed", type=int, default=17)
    cosmos_enhance.add_argument("--samples", type=_positive_int, default=256)
    cosmos_enhance.add_argument("--prompt")
    cosmos_enhance.add_argument("--negative-prompt")
    cosmos_enhance.add_argument("--no-start", action="store_true")
    cosmos_enhance.add_argument("--keep-bundle", action="store_true")
    return parser


def _backend_status(backend: str, host: str) -> dict[str, object]:
    try:
        module = _module(backend)
    except (ImportError, RuntimeError) as exc:
        return {
            "backend": backend,
            "available": False,
            "error": str(exc),
        }
    remote = backend_spec(backend)["remote"]
    return module.backend_status(host) if remote in (True, "optional") else module.backend_status()


def run_status(args: argparse.Namespace | None = None) -> dict[str, object]:
    host = args.host if args is not None else DEFAULT_RENDER_HOST
    requested = args.backend if args is not None else None
    backends = (requested,) if requested else backend_names()
    return {backend: _backend_status(backend, host) for backend in backends}


def run_scene(args: argparse.Namespace) -> dict[str, object]:
    from robotic_renderers.scene import export_mjcf_bundle, scene_summary

    result = scene_summary(args.width, args.height)
    if args.export:
        result["export"] = str(export_mjcf_bundle(args.export, args.width, args.height))
    return result


def run_setup(args: argparse.Namespace) -> dict[str, object]:
    if args.backend == "warp":
        module = _module("warp")
        if args.remote_warp:
            return module.setup_remote(args.host, force=args.force)
        return module.backend_status()
    if args.backend in ("ovrtx", "mitsuba"):
        return _module(args.backend).setup_remote(args.host, force=args.force)
    return _module(args.backend).backend_status()


def run_still(args: argparse.Namespace) -> dict[str, object]:
    output = args.output or _default_output(args.backend, "png")
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.render_still(
            output_path=output,
            width=args.width,
            height=args.height,
            time_s=args.time,
        )
    if args.backend == "warp":
        return module.render_still(
            output_path=output,
            width=args.width,
            height=args.height,
            time_s=args.time,
            supersample=args.supersample,
            host=args.host,
            local=not args.remote_warp,
        )
    if args.backend == "blender":
        return module.render_still(
            output_path=output,
            width=args.width,
            height=args.height,
            time_s=args.time,
            samples=args.samples,
        )
    if args.backend == "ovrtx":
        return module.render_still(
            output_path=output,
            width=args.width,
            height=args.height,
            time_s=args.time,
            samples=args.samples,
            host=args.host,
            render_mode=args.render_mode,
            aovs=tuple(args.aovs),
            aov_output_dir=args.aov_output_dir,
        )
    if args.backend == "mitsuba":
        return module.render_still(
            output_path=output,
            width=args.width,
            height=args.height,
            time_s=args.time,
            samples=args.samples,
            host=args.host,
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def run_live(args: argparse.Namespace) -> dict[str, object]:
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.render_live(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
        )
    if args.backend == "warp":
        return module.render_live(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            supersample=args.supersample,
            host=args.host,
            local=not args.remote_warp,
        )
    if args.backend == "blender":
        return module.render_live(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            samples=args.samples,
        )
    if args.backend == "ovrtx":
        return module.render_live(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            samples=args.samples,
            host=args.host,
            render_mode=args.render_mode,
            aovs=tuple(args.aovs),
            aov_output_dir=args.aov_output_dir,
        )
    if args.backend == "mitsuba":
        return module.render_live(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            samples=args.samples,
            host=args.host,
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def run_record(args: argparse.Namespace) -> dict[str, object]:
    output = args.output or _default_output(args.backend, "mp4")
    if args.backend == "all":
        from robotic_renderers.workflows.tiled import record_all_backends

        return record_all_backends(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            supersample=args.supersample,
            samples=args.samples,
            host=args.host,
            remote_warp=args.remote_warp,
            keep_components=args.keep_components,
            render_mode=args.render_mode,
        )
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.record_video(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    if args.backend == "warp":
        return module.record_video(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            supersample=args.supersample,
            host=args.host,
            local=not args.remote_warp,
        )
    if args.backend == "blender":
        return module.record_video(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            samples=args.samples,
            keep_frames=args.keep_frames,
        )
    if args.backend == "ovrtx":
        return module.record_video(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            samples=args.samples,
            host=args.host,
            render_mode=args.render_mode,
            aovs=tuple(args.aovs),
            aov_output_dir=args.aov_output_dir,
        )
    if args.backend == "mitsuba":
        return module.record_video(
            output_path=output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            samples=args.samples,
            host=args.host,
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def _benchmark_one(backend: str, args: argparse.Namespace) -> dict[str, object]:
    module = _module(backend)
    if backend == "opengl":
        return module.benchmark(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
        )
    if backend == "warp":
        return module.benchmark(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
            supersample=args.supersample,
            host=args.host,
            local=not args.remote_warp,
        )
    if backend == "blender":
        return module.benchmark(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
            samples=args.samples,
        )
    if backend == "ovrtx":
        return module.benchmark(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
            samples=args.samples,
            host=args.host,
            render_mode=args.render_mode,
            aovs=tuple(args.aovs),
            aov_output_dir=args.aov_output_dir,
        )
    if backend == "mitsuba":
        return module.benchmark(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
            samples=args.samples,
            host=args.host,
        )
    raise AssertionError(f"Unhandled backend: {backend}")


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    backends = list(backend_names()) if args.backend == "all" else [args.backend]
    results = {}
    for backend in backends:
        try:
            results[backend] = _benchmark_one(backend, args)
        except Exception as exc:
            results[backend] = {"backend": backend, "error": str(exc)}
    failed_backends = [
        backend for backend, backend_result in results.items() if "error" in backend_result
    ]
    successful_backends = [backend for backend in backends if backend not in failed_backends]
    result = {
        "scene": "fr3_renderer_evaluation",
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "results": results,
        "success": bool(successful_backends),
        "partial": bool(successful_backends and failed_backends),
        "failed_backends": failed_backends,
    }
    if args.output:
        from robotic_renderers.media import atomic_output_path

        with atomic_output_path(args.output) as (temporary, output):
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["output"] = str(output)
    return result


def run_cosmos(args: argparse.Namespace) -> dict[str, object]:
    from robotic_renderers.workflows import cosmos3

    if args.cosmos_action == "status":
        return cosmos3.backend_status(args.host, args.port)
    if args.cosmos_action == "setup":
        return cosmos3.setup_remote(args.host, args.force)
    if args.cosmos_action == "start":
        return cosmos3.start_server(args.host, args.port, cosmos3.DEFAULT_MODEL, not args.no_wait)
    if args.cosmos_action == "stop":
        return cosmos3.stop_server(args.host)
    if args.cosmos_action == "enhance":
        outputs = output_directory()
        output = args.output or (outputs / f"cosmos3_super_{args.mode}.mp4")
        preview = args.preview or (outputs / f"cosmos3_super_{args.mode}.png")
        kwargs = {}
        if args.prompt is not None:
            kwargs["prompt"] = args.prompt
        if args.negative_prompt is not None:
            kwargs["negative_prompt"] = args.negative_prompt
        return cosmos3.enhance_render(
            output,
            backend=args.backend,
            mode=args.mode,
            input_path=args.input,
            host=args.host,
            port=args.port,
            width=args.width,
            height=args.height,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            guidance=args.guidance,
            flow_shift=args.flow_shift,
            seed=args.seed,
            samples=args.samples,
            preview_path=preview,
            start_if_needed=not args.no_start,
            keep_bundle=args.keep_bundle,
            **kwargs,
        )
    raise AssertionError(f"Unhandled Cosmos action: {args.cosmos_action}")


def _explicit_options(argv: list[str]) -> set[str]:
    return {token.partition("=")[0] for token in argv if token.startswith("--") and token != "--"}


def _validate_backend_options(args: argparse.Namespace, argv: list[str]) -> None:
    if args.command not in {"setup", "still", "live", "record", "benchmark"}:
        return
    backend = args.backend
    options = _explicit_options(argv)

    allowed_backends = {
        "--host": {"warp", "ovrtx", "mitsuba", "all"},
        "--remote-warp": {"warp", "all"},
        "--supersample": {"warp", "all"},
        "--samples": {"blender", "ovrtx", "mitsuba", "all"},
        "--render-mode": {"ovrtx", "all"},
        "--aov": {"ovrtx"},
        "--aov-output-dir": {"ovrtx"},
        "--keep-frames": {"blender"},
        "--keep-components": {"all"},
    }
    for option, supported in allowed_backends.items():
        if option in options and backend not in supported:
            choices = ", ".join(sorted(supported))
            raise ValueError(f"{option} is supported only by: {choices}")

    if backend == "all":
        if args.command not in {"record", "benchmark"}:
            raise ValueError("--backend all is supported only for record and benchmark")
        if args.command == "record" and ({"--aov", "--aov-output-dir"} & options):
            raise ValueError("AOV output is not part of the tiled comparison video")
    if "--aov-output-dir" in options and "--aov" not in options:
        raise ValueError("--aov-output-dir requires at least one --aov")
    if backend == "warp" and "--host" in options and not args.remote_warp:
        raise ValueError("--host affects Warp only when --remote-warp is enabled")
    if args.command == "setup" and "--force" in options:
        remote_setup = backend in {"ovrtx", "mitsuba"} or (backend == "warp" and args.remote_warp)
        if not remote_setup:
            raise ValueError("--force applies only to remote backend setup")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    runners = {
        "status": lambda: run_status(args),
        "scene": lambda: run_scene(args),
        "setup": lambda: run_setup(args),
        "still": lambda: run_still(args),
        "live": lambda: run_live(args),
        "record": lambda: run_record(args),
        "benchmark": lambda: run_benchmark(args),
        "cosmos": lambda: run_cosmos(args),
    }
    try:
        _validate_backend_options(args, raw_argv)
        result = runners[args.command]()
        _print_result(result)
        if args.command == "benchmark" and not result["success"]:
            return 1
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
