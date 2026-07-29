"""Renderer-neutral triangle-mesh and environment-image helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return normalized, area-weighted vertex normals for triangular faces."""
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals.astype(np.float32)


def box_mesh(size: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a MuJoCo box whose size values are half-extents."""
    x, y, z = map(float, size)
    vertices = np.asarray(
        [
            (-x, -y, -z),
            (x, -y, -z),
            (x, y, -z),
            (-x, y, -z),
            (-x, -y, z),
            (x, -y, z),
            (x, y, z),
            (-x, y, z),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    return vertices, faces


def sphere_mesh(
    radii: tuple[float, float, float],
    segments: int = 96,
    rings: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a sphere or ellipsoid with the requested axis radii."""
    vertices = []
    for ring in range(rings + 1):
        latitude = -math.pi / 2.0 + math.pi * ring / rings
        radial = math.cos(latitude)
        for segment in range(segments):
            longitude = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    radii[0] * radial * math.cos(longitude),
                    radii[1] * radial * math.sin(longitude),
                    radii[2] * math.sin(latitude),
                )
            )
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            lower = ring * segments + segment
            lower_next = ring * segments + next_segment
            upper = (ring + 1) * segments + segment
            upper_next = (ring + 1) * segments + next_segment
            if ring > 0:
                faces.append((lower, lower_next, upper_next))
            if ring < rings - 1:
                faces.append((lower, upper_next, upper))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def cylinder_mesh(
    radius: float,
    half_height: float,
    segments: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a capped MuJoCo cylinder."""
    vertices = []
    for z in (-half_height, half_height):
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, half_height))
    faces = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.extend(
            [
                (segment, next_segment, segments + next_segment),
                (segment, segments + next_segment, segments + segment),
                (bottom_center, next_segment, segment),
                (top_center, segments + segment, segments + next_segment),
            ]
        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def geom_mesh(
    description: dict[str, object],
    source_geometry,
    meshes: dict[str, object],
    renderer_name: str = "renderer",
) -> tuple[np.ndarray, np.ndarray]:
    """Translate one scene-bundle geom into a triangle mesh."""
    geom_type = description["type"]
    size = description["size"]
    if geom_type == "box":
        return box_mesh(size)
    if geom_type == "sphere":
        return sphere_mesh((size[0], size[0], size[0]))
    if geom_type == "ellipsoid":
        return sphere_mesh((size[0], size[1], size[2]))
    if geom_type == "capsule":
        # Preserve the established renderer translation: a capsule is represented
        # by an ellipsoid with the same total axial extent.
        return sphere_mesh((size[0], size[0], size[0] + size[1]))
    if geom_type == "cylinder":
        return cylinder_mesh(size[0], size[1])
    if geom_type == "mesh":
        mesh = meshes[str(description["mesh_id"])]
        return (
            np.asarray(source_geometry[mesh["vertices"]], dtype=np.float32),
            np.asarray(source_geometry[mesh["faces"]], dtype=np.int32),
        )
    raise ValueError(f"Unsupported {renderer_name} geom type: {geom_type}")


def write_gradient_sky(
    destination: str | Path,
    *,
    width: int,
    height: int,
    filename: str = "studio_gradient.png",
    horizon: tuple[float, float, float] = (0.003, 0.004, 0.009),
    zenith: tuple[float, float, float] = (0.48, 0.60, 0.83),
) -> Path:
    """Write the shared linear-light studio gradient as an sRGB image."""
    from PIL import Image

    latitude = np.pi / 2.0 - np.pi * (np.arange(height) + 0.5) / height
    factor = np.maximum(0.0, np.sin(latitude))[:, np.newaxis]
    horizon_color = np.asarray(horizon, dtype=np.float64)
    zenith_color = np.asarray(zenith, dtype=np.float64)
    linear = horizon_color + factor * (zenith_color - horizon_color)
    linear = np.broadcast_to(linear[:, np.newaxis, :], (height, width, 3))
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    output = Path(destination) / filename
    Image.fromarray(np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output)
    return output
