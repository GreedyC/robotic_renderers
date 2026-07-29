import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robotic_renderers import cli as main
from robotic_renderers.workflows import cosmos3 as render_cosmos3


def test_cosmos_cli_defaults_to_high_quality_depth_transfer():
    args = main.build_parser().parse_args(["cosmos", "enhance"])

    assert args.backend == "blender"
    assert args.mode == "depth"
    assert (args.width, args.height) == (1280, 720)
    assert args.steps == 50
    assert args.host == "render-host"


def test_cosmos_cli_defaults_write_to_user_output_directory(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    monkeypatch.setenv("ROBOTIC_RENDERERS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        render_cosmos3,
        "enhance_render",
        lambda output, **kwargs: calls.append((output, kwargs)) or {"output": str(output)},
    )
    args = main.build_parser().parse_args(["cosmos", "enhance"])

    main.run_cosmos(args)

    assert calls[0][0] == tmp_path / "cosmos3_super_depth.mp4"
    assert calls[0][1]["preview_path"] == tmp_path / "cosmos3_super_depth.png"


def test_cosmos_server_uses_configured_gpus_and_super_bf16():
    command = render_cosmos3._remote_server_command(8091, render_cosmos3.DEFAULT_MODEL)

    assert "nvidia/Cosmos3-Super" in command
    assert render_cosmos3.DEFAULT_MODEL_REVISION in command
    assert "--revision" in command
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command
    assert '"$state/gpus"' in command
    assert '"$state/gpus.tmp"' in command
    assert "--cfg-parallel-size" not in command
    assert "--ulysses-degree 4" in command
    assert "--hsdp-shard-size 4" in command
    assert "--no-guardrails" in command
    assert "VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200" in command
    assert 'export VLLM_API_KEY="$api_key"' in command
    assert "--api-key" not in command
    assert "umask 077" in command
    assert "flock -w 60" in command
    assert "fp8" not in command.lower()


def test_cosmos_gpu_id_parser_requires_unique_nonnegative_four_gpu_selection():
    assert render_cosmos3._parse_gpu_ids("4, 5,6,7") == (4, 5, 6, 7)
    for value in ("0,1,2", "0,1,2,3,4", "0,1,2,2", "0,1,2,-1", "0,1,two,3"):
        with pytest.raises(ValueError):
            render_cosmos3._parse_gpu_ids(value)


def test_cosmos_pid_lifecycle_checks_process_identity_before_every_signal():
    start = render_cosmos3._remote_server_command(8091, render_cosmos3.DEFAULT_MODEL)
    stop = render_cosmos3._remote_stop_command()

    for command in (start, stop):
        assert "/proc/$candidate/stat" in command
        assert "/proc/$candidate/cmdline" in command
        assert 'actual_start=$(cosmos_proc_start "$candidate")' in command
        assert 'cosmos_pid_matches "$candidate" "$expected_start" || return 0' in command
    assert 'kill -KILL -- "-$candidate"' in stop
    assert (
        "kill -KILL"
        not in stop.split('cosmos_pid_matches "$candidate" "$expected_start" || return 0')[0]
    )


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
    )

    assert "size=1280x720" in command
    assert "num_inference_steps=50" in command
    assert "max_sequence_length=4096" in command
    assert '"depth":{"control_path":"/remote/job/depth.mp4"}' in command
    assert "prompt=</remote/job/prompt.txt" in command
    assert "negative_prompt=</remote/job/negative_prompt.txt" in command
    assert "@code/robot_rendering_cosmos3/.server/auth.header" in command


def test_cosmos_request_text_is_private_and_not_in_curl_arguments(tmp_path: Path):
    prompt = "private positive prompt"
    negative_prompt = "private negative prompt"
    render_cosmos3._write_request_text_files(tmp_path, prompt, negative_prompt)
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
    )

    assert prompt not in command
    assert negative_prompt not in command
    assert (tmp_path / "prompt.txt").read_text() == prompt
    assert (tmp_path / "negative_prompt.txt").read_text() == negative_prompt
    assert (tmp_path / "prompt.txt").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "negative_prompt.txt").stat().st_mode & 0o777 == 0o600


def test_cosmos_cleans_local_job_when_remote_home_lookup_fails(tmp_path: Path, monkeypatch):
    job = tmp_path / "conditioning-job"
    job.mkdir()
    source = job / "input.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        render_cosmos3,
        "_server_probe",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        render_cosmos3,
        "_prepare_job",
        lambda *_args, **_kwargs: (job, source, {}),
    )
    monkeypatch.setattr(
        render_cosmos3,
        "_remote_home",
        lambda _host: (_ for _ in ()).throw(RuntimeError("SSH unavailable")),
    )

    with pytest.raises(RuntimeError, match="SSH unavailable"):
        render_cosmos3.enhance_render(
            tmp_path / "result.mp4",
            mode="i2v",
            frames=1,
            start_if_needed=False,
        )

    assert not job.exists()


@pytest.mark.integration
def test_small_conditioning_bundle_has_depth_and_segmentation(tmp_path: Path):
    bundle = render_cosmos3.prepare_conditioning_bundle(
        tmp_path, width=160, height=90, frames=2, fps=2
    )

    manifest = json.loads((tmp_path / "conditioning.json").read_text())
    assert Path(bundle["controls"]["depth"]).is_file()
    assert Path(bundle["controls"]["segmentation"]).is_file()
    assert manifest["frames"] == 2
    assert manifest["controls"] == {
        "depth": "depth.mp4",
        "segmentation": "segmentation.mp4",
    }


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


@pytest.mark.integration
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
