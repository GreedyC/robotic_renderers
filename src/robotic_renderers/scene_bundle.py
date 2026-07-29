"""Portable extraction of the canonical MuJoCo scene for renderer adapters."""

from __future__ import annotations

import json
from pathlib import Path


def _name(model, object_type, object_id: int, fallback: str) -> str:
    import mujoco

    return mujoco.mj_id2name(model, object_type, object_id) or fallback


def material_for_geom(model, geom_id: int) -> dict[str, object]:
    """Describe the material visible on one compiled MuJoCo geom."""
    import mujoco

    material_id = int(model.geom_matid[geom_id])
    if material_id < 0:
        return {
            "name": f"geom_material_{geom_id}",
            "rgba": model.geom_rgba[geom_id].astype(float).tolist(),
            "metallic": 0.0,
            "roughness": 0.42,
            "specular": 0.45,
            "reflectance": 0.0,
            "textures": [],
            "texture_repeat": [1.0, 1.0],
            "texture_uniform": False,
        }

    texture_names = []
    for texture_id in model.mat_texid[material_id]:
        texture_id = int(texture_id)
        texture_names.append(
            _name(
                model,
                mujoco.mjtObj.mjOBJ_TEXTURE,
                texture_id,
                f"texture_{texture_id}",
            )
            if texture_id >= 0
            else None
        )
    return {
        "name": _name(
            model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            material_id,
            f"material_{material_id}",
        ),
        "rgba": model.mat_rgba[material_id].astype(float).tolist(),
        "metallic": float(model.mat_metallic[material_id]),
        "roughness": float(model.mat_roughness[material_id]),
        "specular": float(model.mat_specular[material_id]),
        "reflectance": float(model.mat_reflectance[material_id]),
        "textures": texture_names,
        "texture_repeat": model.mat_texrepeat[material_id].astype(float).tolist(),
        "texture_uniform": bool(model.mat_texuniform[material_id]),
    }


def geom_type_name(geom_type: int) -> str:
    """Return the portable name used for a supported MuJoCo geom type."""
    import mujoco

    names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }
    if geom_type not in names:
        raise ValueError(f"Scene bundle does not support MuJoCo geom type {geom_type}")
    return names[geom_type]


def _visible_geom_ids(model) -> list[int]:
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) in (0, 1, 2)
        and float(model.geom_rgba[geom_id, 3]) > 0.001
    ]


def _extract_meshes(model, geom_ids: list[int]) -> tuple[dict[str, object], dict[str, object]]:
    import mujoco
    import numpy as np

    mesh_ids = sorted(
        {
            int(model.geom_dataid[geom_id])
            for geom_id in geom_ids
            if geom_type_name(int(model.geom_type[geom_id])) == "mesh"
        }
    )
    arrays = {}
    metadata: dict[str, object] = {}
    for mesh_id in mesh_ids:
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertex_key = f"vertices_{mesh_id}"
        face_key = f"faces_{mesh_id}"
        arrays[vertex_key] = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count],
            dtype=np.float32,
        )
        arrays[face_key] = np.asarray(
            model.mesh_face[face_address : face_address + face_count],
            dtype=np.int32,
        )
        metadata[str(mesh_id)] = {
            "name": _name(
                model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id,
                f"mesh_{mesh_id}",
            ),
            "vertices": vertex_key,
            "faces": face_key,
        }
    return arrays, metadata


def _extract_geoms(model, geom_ids: list[int]) -> list[dict[str, object]]:
    import mujoco

    geoms = []
    for geom_id in geom_ids:
        geom_type = geom_type_name(int(model.geom_type[geom_id]))
        geoms.append(
            {
                "id": geom_id,
                "name": _name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                    f"geom_{geom_id}",
                ),
                "type": geom_type,
                "size": model.geom_size[geom_id].astype(float).tolist(),
                "mesh_id": int(model.geom_dataid[geom_id]) if geom_type == "mesh" else None,
                "material": material_for_geom(model, geom_id),
            }
        )
    return geoms


def _extract_frames(model, data, camera_id: int, geom_ids: list[int], times: list[float]):
    from robotic_renderers.scene import set_scene_time

    frames = []
    for frame_index, time_s in enumerate(times, start=1):
        set_scene_time(model, data, time_s)
        frames.append(
            {
                "frame": frame_index,
                "time": time_s,
                "transforms": [
                    {
                        "position": data.geom_xpos[geom_id].astype(float).tolist(),
                        "rotation": data.geom_xmat[geom_id].astype(float).reshape(3, 3).tolist(),
                    }
                    for geom_id in geom_ids
                ],
                "camera": {
                    "position": data.cam_xpos[camera_id].astype(float).tolist(),
                    "rotation": data.cam_xmat[camera_id].astype(float).reshape(3, 3).tolist(),
                },
            }
        )
    return frames


def _extract_lights(model, data) -> list[dict[str, object]]:
    import mujoco

    type_names = {
        int(mujoco.mjtLightType.mjLIGHT_SPOT): "spot",
        int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL): "directional",
        int(mujoco.mjtLightType.mjLIGHT_POINT): "point",
    }
    return [
        {
            "name": _name(
                model,
                mujoco.mjtObj.mjOBJ_LIGHT,
                light_id,
                f"light_{light_id}",
            ),
            "type": type_names[int(model.light_type[light_id])],
            "position": data.light_xpos[light_id].astype(float).tolist(),
            "direction": data.light_xdir[light_id].astype(float).tolist(),
            "ambient": model.light_ambient[light_id].astype(float).tolist(),
            "diffuse": model.light_diffuse[light_id].astype(float).tolist(),
            "specular": model.light_specular[light_id].astype(float).tolist(),
            "intensity": float(model.light_intensity[light_id]),
            "range": float(model.light_range[light_id]),
            "bulb_radius": float(model.light_bulbradius[light_id]),
            "cutoff": float(model.light_cutoff[light_id]),
            "exponent": float(model.light_exponent[light_id]),
            "cast_shadow": bool(model.light_castshadow[light_id]),
        }
        for light_id in range(model.nlight)
    ]


def _texture_description(model, texture_id: int) -> dict[str, object]:
    import mujoco
    import numpy as np

    address = int(model.tex_adr[texture_id])
    width = int(model.tex_width[texture_id])
    height = int(model.tex_height[texture_id])
    channels = int(model.tex_nchannel[texture_id])
    pixels = np.asarray(
        model.tex_data[address : address + width * height * channels],
        dtype=np.uint8,
    ).reshape(height, width, channels)
    return {
        "name": _name(
            model,
            mujoco.mjtObj.mjOBJ_TEXTURE,
            texture_id,
            f"texture_{texture_id}",
        ),
        "width": width,
        "height": height,
        "channels": channels,
        "rgb_min": pixels.min(axis=(0, 1)).astype(float).tolist(),
        "rgb_max": pixels.max(axis=(0, 1)).astype(float).tolist(),
    }


def _extract_environment(model) -> dict[str, object]:
    import mujoco

    textures = {
        description["name"]: description
        for description in (
            _texture_description(model, texture_id) for texture_id in range(model.ntex)
        )
    }
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_material = material_for_geom(model, floor_id) if floor_id >= 0 else None
    repeat = floor_material["texture_repeat"] if floor_material is not None else [1.0, 1.0]
    return {
        "textures": textures,
        "floor": {
            "geom": "floor",
            "material": floor_material["name"] if floor_material is not None else None,
            "texture": "floor_checker",
            "texture_repeat": repeat,
            "tile_size": 1.0 / float(repeat[0]),
        },
        "camera_clip": {
            "near": float(model.vis.map.znear),
            "far": float(model.vis.map.zfar),
        },
        "fog": {
            "start": float(model.vis.map.fogstart),
            "end": float(model.vis.map.fogend),
            "haze": [float(component) for component in model.vis.rgba.haze],
        },
        "headlight_active": bool(model.vis.headlight.active),
    }


def prepare_scene_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Extract geometry, animation, camera, lighting, and environment metadata."""
    import mujoco
    import numpy as np

    from robotic_renderers.scene import CAMERA_NAME, create_model_and_data, set_scene_time

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model, data = create_model_and_data(width=width, height=height)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} is absent from the compiled scene")

    geom_ids = _visible_geom_ids(model)
    geometry_arrays, mesh_metadata = _extract_meshes(model, geom_ids)
    geometry_path = destination / "geometry.npz"
    np.savez_compressed(geometry_path, **geometry_arrays)
    geoms = _extract_geoms(model, geom_ids)
    frames = _extract_frames(model, data, camera_id, geom_ids, times)

    set_scene_time(model, data, times[0] if times else 0.0)
    manifest = {
        "width": width,
        "height": height,
        "fps": fps,
        "geometry": geometry_path.name,
        "meshes": mesh_metadata,
        "geoms": geoms,
        "frames": frames,
        "camera": {
            "name": CAMERA_NAME,
            "position": data.cam_xpos[camera_id].astype(float).tolist(),
            "rotation": data.cam_xmat[camera_id].astype(float).reshape(3, 3).tolist(),
            "fovy": float(model.cam_fovy[camera_id]),
        },
        "lights": _extract_lights(model, data),
        "environment": _extract_environment(model),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest_path
