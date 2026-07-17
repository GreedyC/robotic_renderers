from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import main
from render_common import write_video
from render_tiled import BACKEND_ORDER, compose_tiled_video


def test_record_parser_accepts_all_backends():
    args = main.build_parser().parse_args(["record", "--backend", "all"])

    assert args.backend == "all"
    assert args.keep_components is False


def test_tiled_compositor_labels_four_synchronized_videos(tmp_path: Path):
    colors = {
        "opengl": (210, 30, 30),
        "warp": (30, 210, 30),
        "blender": (30, 30, 210),
        "ovrtx": (180, 30, 210),
    }
    paths = {}
    for backend in BACKEND_ORDER:
        path = tmp_path / f"{backend}.mp4"
        frame = np.full((36, 64, 3), colors[backend], dtype=np.uint8)
        write_video((frame.copy() for _ in range(2)), path, fps=2)
        paths[backend] = path

    result = compose_tiled_video(paths, tmp_path / "tiled.mp4", 1.0, 2, 64, 36)
    reader = imageio.get_reader(result["output"])
    try:
        first = reader.get_data(0)
        second = reader.get_data(1)
    finally:
        reader.close()

    assert first.shape == (72, 128, 3)
    assert second.shape == first.shape
    assert result["frames"] == 2
    assert result["width"] == 128
    assert result["height"] == 72
    assert result["columns"] == 2
    assert result["rows"] == 2
