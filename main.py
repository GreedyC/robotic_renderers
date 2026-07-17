"""Command-line app for the FR3 renderer evaluation suite."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

BACKEND_MODULES = {
    "opengl": "render_mujoco_opengl",
    "warp": "render_mujoco_warp",
    "blender": "render_blender",
    "ovrtx": "render_ovrtx",
}


def _module(backend: str):
    return importlib.import_module(BACKEND_MODULES[backend])


def _print_result(result) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _default_output(backend: str, extension: str) -> Path:
    output = Path(__file__).resolve().parent / "outputs" / f"{backend}.{extension}"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _add_backend(parser: argparse.ArgumentParser, allow_all: bool = False) -> None:
    choices = list(BACKEND_MODULES)
    if allow_all:
        choices.append("all")
    parser.add_argument("--backend", choices=choices, required=True)


def _add_dimensions(parser: argparse.ArgumentParser, live: bool = False) -> None:
    parser.add_argument("--width", type=int, default=960 if live else 1280)
    parser.add_argument("--height", type=int, default=540 if live else 720)


def _add_shared_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="l4", help="GPU SSH host for Warp or ovrtx")
    parser.add_argument(
        "--remote-warp",
        action="store_true",
        help="Use Warp remotely over SSH instead of the local macOS CPU device",
    )
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--samples", type=int, default=256, help="Blender/ovrtx path-tracing samples"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one canonical Franka FR3 scene with four pipelines."
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Probe all renderer installations and devices")

    scene_parser = subparsers.add_parser("scene", help="Inspect or export the canonical MjSpec")
    scene_parser.add_argument("--export", type=Path, help="Write a portable MJCF bundle")
    _add_dimensions(scene_parser)

    setup_parser = subparsers.add_parser("setup", help="Prepare an external renderer")
    _add_backend(setup_parser)
    _add_dimensions(setup_parser)
    _add_shared_backend_options(setup_parser)
    setup_parser.add_argument("--force", action="store_true")

    still_parser = subparsers.add_parser("still", help="Render one PNG")
    _add_backend(still_parser)
    _add_dimensions(still_parser)
    _add_shared_backend_options(still_parser)
    still_parser.add_argument("--time", type=float, default=1.0)
    still_parser.add_argument("--output", type=Path)

    live_parser = subparsers.add_parser("live", help="Open a live renderer view")
    _add_backend(live_parser)
    _add_dimensions(live_parser, live=True)
    _add_shared_backend_options(live_parser)
    live_parser.set_defaults(supersample=1, samples=64)
    live_parser.add_argument("--fps", type=int, default=30)
    live_parser.add_argument("--duration", type=float)

    record_parser = subparsers.add_parser("record", help="Record an MP4")
    _add_backend(record_parser, allow_all=True)
    _add_dimensions(record_parser)
    _add_shared_backend_options(record_parser)
    record_parser.add_argument("--fps", type=int, default=30)
    record_parser.add_argument("--duration", type=float, default=5.0)
    record_parser.add_argument("--output", type=Path)
    record_parser.add_argument("--keep-frames", action="store_true")
    record_parser.add_argument(
        "--keep-components",
        action="store_true",
        help="Keep the four source MP4s next to an --backend all output",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Measure renderer throughput (encoding excluded)"
    )
    _add_backend(benchmark_parser, allow_all=True)
    _add_dimensions(benchmark_parser)
    _add_shared_backend_options(benchmark_parser)
    benchmark_parser.set_defaults(samples=32, supersample=1)
    benchmark_parser.add_argument("--fps", type=int, default=30)
    benchmark_parser.add_argument("--frames", type=int, default=10)
    benchmark_parser.add_argument("--warmup", type=int, default=3)
    benchmark_parser.add_argument("--output", type=Path, help="Optional JSON result path")

    cosmos_parser = subparsers.add_parser(
        "cosmos", help="Turn simulated renders into photoreal video with Cosmos3-Super"
    )
    cosmos_actions = cosmos_parser.add_subparsers(dest="cosmos_action", required=True)

    cosmos_status = cosmos_actions.add_parser("status", help="Probe the g4 runtime and server")
    cosmos_status.add_argument("--host", default="g4")
    cosmos_status.add_argument("--port", type=int, default=8091)

    cosmos_setup = cosmos_actions.add_parser("setup", help="Install the pinned uv runtime on g4")
    cosmos_setup.add_argument("--host", default="g4")
    cosmos_setup.add_argument("--force", action="store_true")

    cosmos_start = cosmos_actions.add_parser(
        "start", help="Start the persistent 64B BF16 service on g4's local GPU pair"
    )
    cosmos_start.add_argument("--host", default="g4")
    cosmos_start.add_argument("--port", type=int, default=8091)
    cosmos_start.add_argument("--no-wait", action="store_true")

    cosmos_stop = cosmos_actions.add_parser("stop", help="Stop the remote Cosmos service")
    cosmos_stop.add_argument("--host", default="g4")

    cosmos_enhance = cosmos_actions.add_parser(
        "enhance", help="Render a source and generatively enhance it"
    )
    cosmos_enhance.add_argument("--backend", choices=list(BACKEND_MODULES), default="blender")
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
    cosmos_enhance.add_argument("--host", default="g4")
    cosmos_enhance.add_argument("--port", type=int, default=8091)
    cosmos_enhance.add_argument("--width", type=int, default=1280)
    cosmos_enhance.add_argument("--height", type=int, default=720)
    cosmos_enhance.add_argument(
        "--frames", type=int, help="Default: 189 for I2V/V2V, 121 for transfer control"
    )
    cosmos_enhance.add_argument("--fps", type=int, default=24)
    cosmos_enhance.add_argument("--steps", type=int, default=50)
    cosmos_enhance.add_argument("--guidance", type=float)
    cosmos_enhance.add_argument("--flow-shift", type=float)
    cosmos_enhance.add_argument("--seed", type=int, default=17)
    cosmos_enhance.add_argument("--samples", type=int, default=256)
    cosmos_enhance.add_argument("--prompt")
    cosmos_enhance.add_argument("--negative-prompt")
    cosmos_enhance.add_argument("--no-start", action="store_true")
    cosmos_enhance.add_argument("--keep-bundle", action="store_true")
    return parser


def run_status() -> dict[str, object]:
    return {backend: _module(backend).backend_status() for backend in BACKEND_MODULES}


def run_scene(args: argparse.Namespace) -> dict[str, object]:
    from scene import export_mjcf_bundle, scene_summary

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
    if args.backend == "ovrtx":
        return _module("ovrtx").setup_remote(args.host, force=args.force)
    return _module(args.backend).backend_status()


def run_still(args: argparse.Namespace) -> dict[str, object]:
    output = args.output or _default_output(args.backend, "png")
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.render_still(output, args.width, args.height, args.time)
    if args.backend == "warp":
        return module.render_still(
            output,
            args.width,
            args.height,
            args.time,
            args.supersample,
            args.host,
            not args.remote_warp,
        )
    if args.backend == "blender":
        return module.render_still(output, args.width, args.height, args.time, args.samples)
    if args.backend == "ovrtx":
        return module.render_still(
            output, args.width, args.height, args.time, args.samples, args.host
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def run_live(args: argparse.Namespace) -> dict[str, object]:
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.render_live(args.width, args.height, args.fps, args.duration)
    if args.backend == "warp":
        return module.render_live(
            args.width,
            args.height,
            args.fps,
            args.duration,
            args.supersample,
            args.host,
            not args.remote_warp,
        )
    if args.backend == "blender":
        return module.render_live(args.width, args.height, args.fps, args.duration, args.samples)
    if args.backend == "ovrtx":
        return module.render_live(
            args.width, args.height, args.fps, args.duration, args.samples, args.host
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def run_record(args: argparse.Namespace) -> dict[str, object]:
    output = args.output or _default_output(args.backend, "mp4")
    if args.backend == "all":
        from render_tiled import record_all_backends

        return record_all_backends(
            output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.supersample,
            args.samples,
            args.host,
            args.remote_warp,
            args.keep_components,
        )
    module = _module(args.backend)
    if args.backend == "opengl":
        return module.record_video(output, args.duration, args.fps, args.width, args.height)
    if args.backend == "warp":
        return module.record_video(
            output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.supersample,
            args.host,
            not args.remote_warp,
        )
    if args.backend == "blender":
        return module.record_video(
            output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.samples,
            args.keep_frames,
        )
    if args.backend == "ovrtx":
        return module.record_video(
            output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.samples,
            args.host,
        )
    raise AssertionError(f"Unhandled backend: {args.backend}")


def _benchmark_one(backend: str, args: argparse.Namespace) -> dict[str, object]:
    module = _module(backend)
    if backend == "opengl":
        return module.benchmark(args.frames, args.width, args.height, args.fps, args.warmup)
    if backend == "warp":
        return module.benchmark(
            args.frames,
            args.width,
            args.height,
            args.fps,
            args.warmup,
            args.supersample,
            args.host,
            not args.remote_warp,
        )
    if backend == "blender":
        return module.benchmark(args.frames, args.width, args.height, args.fps, args.samples)
    if backend == "ovrtx":
        return module.benchmark(
            args.frames,
            args.width,
            args.height,
            args.fps,
            args.warmup,
            args.samples,
            args.host,
        )
    raise AssertionError(f"Unhandled backend: {backend}")


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    backends = list(BACKEND_MODULES) if args.backend == "all" else [args.backend]
    results = {}
    for backend in backends:
        try:
            results[backend] = _benchmark_one(backend, args)
        except Exception as exc:
            results[backend] = {"backend": backend, "error": str(exc)}
    result = {
        "scene": "fr3_renderer_evaluation",
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "results": results,
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["output"] = str(output)
    return result


def run_cosmos(args: argparse.Namespace) -> dict[str, object]:
    import render_cosmos3

    if args.cosmos_action == "status":
        return render_cosmos3.backend_status(args.host, args.port)
    if args.cosmos_action == "setup":
        return render_cosmos3.setup_remote(args.host, args.force)
    if args.cosmos_action == "start":
        return render_cosmos3.start_server(
            args.host, args.port, render_cosmos3.DEFAULT_MODEL, not args.no_wait
        )
    if args.cosmos_action == "stop":
        return render_cosmos3.stop_server(args.host)
    if args.cosmos_action == "enhance":
        output = args.output or (
            Path(__file__).resolve().parent / "videos" / f"cosmos3_super_{args.mode}.mp4"
        )
        preview = args.preview or (
            Path(__file__).resolve().parent / "outputs" / f"cosmos3_super_{args.mode}.png"
        )
        kwargs = {}
        if args.prompt is not None:
            kwargs["prompt"] = args.prompt
        if args.negative_prompt is not None:
            kwargs["negative_prompt"] = args.negative_prompt
        return render_cosmos3.enhance_render(
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runners = {
        "status": lambda: run_status(),
        "scene": lambda: run_scene(args),
        "setup": lambda: run_setup(args),
        "still": lambda: run_still(args),
        "live": lambda: run_live(args),
        "record": lambda: run_record(args),
        "benchmark": lambda: run_benchmark(args),
        "cosmos": lambda: run_cosmos(args),
    }
    try:
        _print_result(runners[args.command]())
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
