import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

import main
import render_cosmos3


def test_cosmos_cli_defaults_to_high_quality_depth_transfer():
    args = main.build_parser().parse_args(["cosmos", "enhance"])

    assert args.backend == "blender"
    assert args.mode == "depth"
    assert (args.width, args.height) == (1280, 720)
    assert args.steps == 50
    assert args.host == "g4"


def test_cosmos_server_uses_super_bf16_on_one_numa_node():
    command = render_cosmos3._remote_server_command(8091, render_cosmos3.DEFAULT_MODEL)

    assert "nvidia/Cosmos3-Super" in command
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command
    assert "--cfg-parallel-size" not in command
    assert "--ulysses-degree 4" in command
    assert "--hsdp-shard-size 4" in command
    assert "--no-guardrails" in command
    assert "VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200" in command
    assert "fp8" not in command.lower()


def test_transfer_request_uses_depth_and_first_rgb_frame():
    extra = render_cosmos3._transfer_extra("depth", "/remote/job", 121, 24)

    assert extra["depth"] == {"control_path": "/remote/job/depth.mp4"}
    assert extra["num_first_chunk_conditional_frames"] == 1
    assert extra["num_video_frames_per_chunk"] == 121
    assert extra["resolution"] == "720"
    assert extra["guidance_scale"] == 3.0
    assert extra["control_guidance"] == 1.5


def test_request_contains_full_quality_controls():
    command = render_cosmos3._request_command(
        "/remote/job",
        "input.mp4",
        "depth",
        8091,
        render_cosmos3.DEFAULT_MODEL,
        1280,
        720,
        121,
        24,
        50,
        3.0,
        10.0,
        17,
        "prompt",
        "negative",
    )

    assert "size=1280x720" in command
    assert "num_inference_steps=50" in command
    assert "max_sequence_length=4096" in command
    assert '"depth":{"control_path":"/remote/job/depth.mp4"}' in command


def test_small_conditioning_bundle_has_depth_and_segmentation(tmp_path: Path):
    bundle = render_cosmos3.prepare_conditioning_bundle(
        tmp_path, width=160, height=90, frames=2, fps=2
    )

    manifest = json.loads((tmp_path / "conditioning.json").read_text())
    assert Path(bundle["controls"]["depth"]).is_file()
    assert Path(bundle["controls"]["segmentation"]).is_file()
    assert manifest["frames"] == 2
    assert manifest["controls"].keys() == {"depth", "segmentation"}


def test_transfer_image_is_sent_as_one_rgb_keyframe(tmp_path: Path):
    source = tmp_path / "render.png"
    Image.fromarray(np.full((90, 160, 3), 127, dtype=np.uint8)).save(source)

    video = render_cosmos3._copy_input(source, tmp_path / "job", "depth", 24)
    reader = render_cosmos3.imageio.get_reader(video)
    try:
        assert reader.get_data(0).shape == (90, 160, 3)
        try:
            reader.get_data(1)
        except IndexError:
            pass
        else:
            raise AssertionError("transfer keyframe video unexpectedly has a second frame")
    finally:
        reader.close()


def test_depth_and_segmentation_frames_are_valid_controls():
    context = render_cosmos3.create_renderer(160, 90)
    try:
        depth = render_cosmos3._depth_control_frame(context, 0.0)
        palette = render_cosmos3._segmentation_palette(context["model"])
        segmentation = render_cosmos3._segmentation_control_frame(context, palette, 0.0)
    finally:
        render_cosmos3.close_renderer(context)

    for frame in (depth, segmentation):
        assert frame.shape == (90, 160, 3)
        assert frame.dtype == np.uint8
        assert frame.std() > 5.0


def test_cosmos_source_defines_no_application_classes():
    source = Path(render_cosmos3.__file__).read_text()
    tree = ast.parse(source)

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
