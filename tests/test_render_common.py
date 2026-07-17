from pathlib import Path

import numpy as np
from PIL import Image

from render_common import ensure_rgb8, resize_rgb, write_png, write_video


def test_image_helpers(tmp_path: Path):
    float_image = np.linspace(0.0, 1.0, 16 * 12 * 3).reshape(12, 16, 3)
    rgb = ensure_rgb8(float_image)
    resized = resize_rgb(rgb, 8, 6)
    output = write_png(resized, tmp_path / "frame.png")

    assert rgb.dtype == np.uint8
    assert resized.shape == (6, 8, 3)
    assert Image.open(output).size == (8, 6)


def test_mp4_writer(tmp_path: Path):
    frames = [
        np.full((16, 16, 3), fill_value=value, dtype=np.uint8)
        for value in (0, 64, 128)
    ]
    output, count, seconds = write_video(frames, tmp_path / "tiny.mp4", fps=3)

    assert output.exists()
    assert output.stat().st_size > 0
    assert count == 3
    assert seconds >= 0.0
