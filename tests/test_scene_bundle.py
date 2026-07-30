import json
from pathlib import Path

import numpy as np
from PIL import Image

from robotic_renderers import geometry, scene_bundle
from robotic_renderers.backends import blender as render_blender


def test_scene_bundle_contains_shared_scene_inputs(tmp_path: Path):
    manifest_path = scene_bundle.prepare_scene_bundle(
        tmp_path,
        times=[0.0, 1.0 / 30.0],
        width=160,
        height=90,
        fps=30,
    )
    manifest = json.loads(manifest_path.read_text())
    source_geometry = np.load(tmp_path / manifest["geometry"])

    assert manifest["width"] == 160
    assert manifest["height"] == 90
    assert len(manifest["frames"]) == 2
    assert {light["name"] for light in manifest["lights"]} == {
        "sun_key",
        "soft_fill",
        "warm_rim",
    }
    assert manifest["environment"]["floor"]["tile_size"] == 1.0 / 3.0
    assert manifest["environment"]["headlight_active"] is False
    assert {"studio_sky", "floor_checker"} <= set(manifest["environment"]["textures"])
    assert any(name.startswith("vertices_") for name in source_geometry.files)


def test_blender_bundle_uses_the_canonical_scene_bundle(tmp_path: Path):
    manifest_path = render_blender.prepare_blender_bundle(
        tmp_path,
        times=[0.0],
        width=160,
        height=90,
        fps=30,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["lights"][0]["name"] == "sun_key"
    assert manifest["environment"]["floor"]["tile_size"] == 1.0 / 3.0


def test_shared_geometry_helpers_preserve_mesh_winding_and_normals():
    vertices, faces = geometry.box_mesh([1.0, 2.0, 3.0])
    normals = geometry.surface_normals(vertices, faces)

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_array_equal(vertices.min(axis=0), [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(vertices.max(axis=0), [1.0, 2.0, 3.0])


def test_shared_gradient_sky_honors_backend_resolution(tmp_path: Path):
    path = geometry.write_gradient_sky(tmp_path, width=64, height=32)

    image = np.asarray(Image.open(path))
    assert image.shape == (32, 64, 3)
    assert np.unique(image, axis=1).shape[1] == 1
    assert image[0].mean() > image[-1].mean()
