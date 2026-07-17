"""Blender Cycles backend driven from the canonical MuJoCo scene.

The uv process exports compiled meshes and exact MuJoCo geom transforms. Blender's
bundled Python then builds and renders that interchange bundle in a separate process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND_NAME = "blender"
ROOT = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def find_blender() -> Path | None:
    candidates = [
        Path(os.environ["BLENDER_BIN"]) if "BLENDER_BIN" in os.environ else None,
        DEFAULT_BLENDER,
        Path(shutil.which("blender")) if shutil.which("blender") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def backend_status() -> dict[str, object]:
    blender = find_blender()
    version = None
    error = None
    if blender:
        try:
            completed = subprocess.run(
                [str(blender), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = completed.stdout.splitlines()[0].removeprefix("Blender ")
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc)
    return {
        "backend": BACKEND_NAME,
        "available": blender is not None and error is None,
        "version": version,
        "device": "Cycles/Metal (Apple Silicon)",
        "binary": str(blender) if blender else None,
        "live": True,
        "record": True,
        "notes": (
            "Cycles path tracing uses Blender's version-matched bundled bpy in a separate "
            "process; PNG frames are encoded to MP4 by the uv environment."
        ),
        "error": error,
    }


def _material_for_geom(model, geom_id: int) -> dict[str, object]:
    import mujoco

    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
        return {
            "name": name or f"material_{material_id}",
            "rgba": model.mat_rgba[material_id].astype(float).tolist(),
            "metallic": float(model.mat_metallic[material_id]),
            "roughness": float(model.mat_roughness[material_id]),
            "specular": float(model.mat_specular[material_id]),
            "reflectance": float(model.mat_reflectance[material_id]),
        }
    return {
        "name": f"geom_material_{geom_id}",
        "rgba": model.geom_rgba[geom_id].astype(float).tolist(),
        "metallic": 0.0,
        "roughness": 0.42,
        "specular": 0.45,
        "reflectance": 0.0,
    }


def _geom_type_name(geom_type: int) -> str:
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
        raise ValueError(f"Blender adapter does not support MuJoCo geom type {geom_type}")
    return names[geom_type]


def prepare_blender_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
) -> Path:
    import mujoco
    import numpy as np

    from scene import CAMERA_NAME, create_model_and_data, set_scene_time

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model, data = create_model_and_data(width=width, height=height)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} is absent from the compiled scene")

    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) in (0, 1, 2)
        and float(model.geom_rgba[geom_id, 3]) > 0.001
    ]
    mesh_ids = sorted(
        {
            int(model.geom_dataid[geom_id])
            for geom_id in geom_ids
            if _geom_type_name(int(model.geom_type[geom_id])) == "mesh"
        }
    )

    geometry_arrays = {}
    mesh_metadata: dict[str, object] = {}
    for mesh_id in mesh_ids:
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertex_key = f"vertices_{mesh_id}"
        face_key = f"faces_{mesh_id}"
        geometry_arrays[vertex_key] = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count], dtype=np.float32
        )
        geometry_arrays[face_key] = np.asarray(
            model.mesh_face[face_address : face_address + face_count], dtype=np.int32
        )
        mesh_metadata[str(mesh_id)] = {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            or f"mesh_{mesh_id}",
            "vertices": vertex_key,
            "faces": face_key,
        }
    geometry_path = destination / "geometry.npz"
    np.savez_compressed(geometry_path, **geometry_arrays)

    geoms = []
    for geom_id in geom_ids:
        geom_type = _geom_type_name(int(model.geom_type[geom_id]))
        geoms.append(
            {
                "id": geom_id,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom_{geom_id}",
                "type": geom_type,
                "size": model.geom_size[geom_id].astype(float).tolist(),
                "mesh_id": int(model.geom_dataid[geom_id]) if geom_type == "mesh" else None,
                "material": _material_for_geom(model, geom_id),
            }
        )

    frames = []
    for frame_index, time_s in enumerate(times, start=1):
        set_scene_time(model, data, time_s)
        transforms = []
        for geom_id in geom_ids:
            transforms.append(
                {
                    "position": data.geom_xpos[geom_id].astype(float).tolist(),
                    "rotation": data.geom_xmat[geom_id].astype(float).reshape(3, 3).tolist(),
                }
            )
        frames.append(
            {
                "frame": frame_index,
                "time": time_s,
                "transforms": transforms,
                "camera": {
                    "position": data.cam_xpos[camera_id].astype(float).tolist(),
                    "rotation": data.cam_xmat[camera_id].astype(float).reshape(3, 3).tolist(),
                },
            }
        )

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
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest_path


def _new_job(prefix: str) -> Path:
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=build))


def _blender_command(mode: str, manifest: Path, output: Path, samples: int) -> list[str]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError(
            "Blender was not found. Set BLENDER_BIN or install /Applications/Blender.app."
        )
    command = [str(blender)]
    if mode != "live":
        command.extend(["--background", "--factory-startup", "--python-exit-code", "1"])
    else:
        command.append("--factory-startup")
    command.extend(
        [
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--blender-worker",
            mode,
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--samples",
            str(samples),
        ]
    )
    return command


def _run_blender(mode: str, manifest: Path, output: Path, samples: int) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        _blender_command(mode, manifest, output, samples),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr)[-12000:]
        raise RuntimeError(f"Blender {mode} failed with exit {completed.returncode}:\n{details}")
    return time.perf_counter() - started


def render_still(
    output_path: str | Path,
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
    keep_job: bool = False,
) -> dict[str, object]:
    import numpy as np
    from PIL import Image

    from render_common import image_digest

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    job = _new_job("blender-still-")
    try:
        manifest = prepare_blender_bundle(job, [time_s], width, height, fps=1)
        render_seconds = _run_blender("still", manifest, output, samples)
        image = np.asarray(Image.open(output).convert("RGB"))
        return {
            "backend": BACKEND_NAME,
            "output": str(output),
            "seconds": render_seconds,
            "sha256": image_digest(image),
            "width": width,
            "height": height,
            "samples": samples,
            "device": "Cycles/Metal",
        }
    finally:
        if not keep_job:
            shutil.rmtree(job, ignore_errors=True)


def render_frame(
    width: int = 1280,
    height: int = 720,
    time_s: float = 1.0,
    samples: int = 256,
):
    import numpy as np
    from PIL import Image

    job = _new_job("blender-frame-")
    try:
        output = job / "frame.png"
        render_still(output, width, height, time_s, samples)
        return np.asarray(Image.open(output).convert("RGB"))
    finally:
        shutil.rmtree(job, ignore_errors=True)


def record_video(
    output_path: str | Path,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    samples: int = 256,
    keep_frames: bool = False,
) -> dict[str, object]:
    from PIL import Image

    from render_common import write_video
    from scene import trajectory_times

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    job = _new_job("blender-video-")
    frame_directory = job / "frames"
    frame_directory.mkdir()
    times = list(trajectory_times(duration, fps))
    try:
        manifest = prepare_blender_bundle(job, times, width, height, fps)
        render_seconds = _run_blender("frames", manifest, frame_directory, samples)
        frame_paths = sorted(frame_directory.glob("*.png"))
        if len(frame_paths) != len(times):
            raise RuntimeError(f"Blender produced {len(frame_paths)} frames; expected {len(times)}")

        def frames():
            for frame_path in frame_paths:
                with Image.open(frame_path) as image:
                    yield image.convert("RGB").copy()

        _, frame_count, encode_seconds = write_video(frames(), output, fps=fps)
        if keep_frames:
            saved_frames = output.with_suffix("").with_name(output.stem + "_frames")
            if saved_frames.exists():
                shutil.rmtree(saved_frames)
            shutil.copytree(frame_directory, saved_frames)
        return {
            "backend": BACKEND_NAME,
            "output": str(output),
            "frames": frame_count,
            "render_seconds": render_seconds,
            "encode_seconds": encode_seconds,
            "seconds": render_seconds + encode_seconds,
            "fps": frame_count / max(render_seconds, 1e-9),
            "width": width,
            "height": height,
            "samples": samples,
            "device": "Cycles/Metal",
        }
    finally:
        shutil.rmtree(job, ignore_errors=True)


def render_live(
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    duration: float | None = None,
    samples: int = 64,
) -> dict[str, object]:
    from scene import trajectory_times

    loop_duration = duration if duration is not None else 10.0
    job = _new_job("blender-live-")
    manifest = prepare_blender_bundle(
        job, list(trajectory_times(loop_duration, fps)), width, height, fps
    )
    blend_file = job / "interactive.blend"
    process = subprocess.Popen(_blender_command("live", manifest, blend_file, samples))
    if duration is not None:
        time.sleep(max(duration, 0.0))
        process.terminate()
    return {
        "backend": BACKEND_NAME,
        "pid": process.pid,
        "blend_file": str(blend_file),
        "notes": "Blender stays open for interactive rendered-viewport playback.",
    }


def benchmark(
    frame_count: int = 3,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    samples: int = 32,
) -> dict[str, object]:
    job = _new_job("blender-benchmark-")
    frame_directory = job / "frames"
    frame_directory.mkdir()
    try:
        times = [index / fps for index in range(frame_count)]
        manifest = prepare_blender_bundle(job, times, width, height, fps)
        seconds = _run_blender("frames", manifest, frame_directory, samples)
        return {
            "backend": BACKEND_NAME,
            "device": "Cycles/Metal",
            "width": width,
            "height": height,
            "samples": samples,
            "frames": frame_count,
            "render_seconds": seconds,
            "render_fps": frame_count / max(seconds, 1e-9),
        }
    finally:
        shutil.rmtree(job, ignore_errors=True)


# Everything below this line runs inside Blender's bundled Python.


def _worker_matrix(position, rotation):
    from mathutils import Matrix

    return Matrix(
        (
            (*rotation[0], position[0]),
            (*rotation[1], position[1]),
            (*rotation[2], position[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _worker_set_input(node, names: tuple[str, ...], value) -> None:
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def _worker_environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _worker_make_material(description):
    import bpy

    name = description["name"]
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    rgba = tuple(description["rgba"])
    _worker_set_input(principled, ("Base Color",), rgba)
    _worker_set_input(principled, ("Metallic",), description["metallic"])
    _worker_set_input(principled, ("Roughness",), max(0.035, description["roughness"]))
    _worker_set_input(
        principled,
        ("Specular IOR Level", "Specular"),
        min(1.0, 0.25 + 0.5 * description["specular"]),
    )
    _worker_set_input(principled, ("Alpha",), rgba[3])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    if name == "floor_material":
        texcoord = nodes.new("ShaderNodeTexCoord")
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Color1"].default_value = (0.094, 0.113, 0.133, 1.0)
        checker.inputs["Color2"].default_value = (0.022, 0.027, 0.033, 1.0)
        checker.inputs["Scale"].default_value = 600.0
        links.new(texcoord.outputs["Generated"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], principled.inputs["Base Color"])
    return material


def _worker_apply_bevel(obj, width: float) -> None:
    modifier = obj.modifiers.new(name="Micro bevel", type="BEVEL")
    modifier.width = width
    modifier.segments = 4
    modifier.limit_method = "ANGLE"


def _worker_create_geom(description, geometry, meshes):
    import bpy

    geom_type = description["type"]
    size = description["size"]
    name = description["name"]
    if geom_type == "plane":
        bpy.ops.mesh.primitive_plane_add(size=200.0)
        obj = bpy.context.object
    elif geom_type == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.object
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _worker_apply_bevel(obj, max(0.0015, min(size) * 0.055))
    elif geom_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=64, radius=size[0])
        obj = bpy.context.object
    elif geom_type == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=64, radius=1.0)
        obj = bpy.context.object
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif geom_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=size[0], depth=2.0 * size[1])
        obj = bpy.context.object
        _worker_apply_bevel(obj, max(0.001, min(size[0], size[1]) * 0.035))
    elif geom_type == "capsule":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=64, radius=1.0)
        obj = bpy.context.object
        obj.scale = (size[0], size[0], size[0] + size[1])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif geom_type == "mesh":
        mesh_description = meshes[str(description["mesh_id"])]
        vertices = geometry[mesh_description["vertices"]]
        faces = geometry[mesh_description["faces"]]
        mesh = bpy.data.meshes.new(name=f"{name}_mesh")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(name=name, object_data=mesh)
        bpy.context.collection.objects.link(obj)
        smooth = mesh_description["name"] != "mesh_gem"
        for polygon in mesh.polygons:
            polygon.use_smooth = smooth
    else:
        raise RuntimeError(f"Unsupported geom type {geom_type}")
    obj.name = name
    material = _worker_make_material(description["material"])
    obj.data.materials.append(material)
    return obj


def _worker_point_light(light, direction):
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _worker_add_lighting():
    import bpy
    from mathutils import Vector

    key_data = bpy.data.lights.new(name="sun_key", type="SUN")
    key_data.energy = 2.8
    key_data.angle = math.radians(3.0)
    key_data.color = (0.95, 0.91, 0.84)
    key = bpy.data.objects.new(name="sun_key", object_data=key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-1.5, -1.0, 3.0)
    _worker_point_light(key, Vector((0.45, 0.2, -1.0)))

    fill_data = bpy.data.lights.new(name="soft_fill", type="POINT")
    fill_data.energy = 90.0
    fill_data.shadow_soft_size = 0.25
    fill_data.color = (0.34, 0.47, 0.7)
    fill = bpy.data.objects.new(name="soft_fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (0.2, -1.5, 1.45)

    rim_data = bpy.data.lights.new(name="warm_rim", type="SPOT")
    rim_data.energy = 100.0
    rim_data.spot_size = math.radians(55.0)
    rim_data.spot_blend = 0.35
    rim_data.shadow_soft_size = 0.08
    rim_data.color = (0.8, 0.46, 0.25)
    rim = bpy.data.objects.new(name="warm_rim", object_data=rim_data)
    bpy.context.collection.objects.link(rim)
    rim.location = (-0.65, 0.9, 1.8)
    _worker_point_light(rim, Vector((0.5, -0.6, -1.0)))


def _worker_add_world():
    import bpy

    world = bpy.data.worlds.new("Studio World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.35
    coordinates = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    gradient = nodes.new("ShaderNodeValToRGB")
    gradient.color_ramp.elements[0].position = 0.0
    gradient.color_ramp.elements[0].color = (0.003, 0.004, 0.009, 1.0)
    gradient.color_ramp.elements[1].position = 1.0
    gradient.color_ramp.elements[1].color = (0.48, 0.60, 0.83, 1.0)
    links.new(coordinates.outputs["Normal"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], gradient.inputs["Fac"])
    links.new(gradient.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def _worker_configure_render(manifest, samples: int, live: bool):
    import bpy

    scene = bpy.context.scene
    scene.render.resolution_x = manifest["width"]
    scene.render.resolution_y = manifest["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = manifest["fps"]
    scene.render.engine = "BLENDER_EEVEE" if live else "CYCLES"
    if not live:
        scene.render.use_persistent_data = _worker_environment_flag(
            "BLENDER_PERSISTENT_DATA", True
        )
        requested_threads = os.environ.get("BLENDER_CPU_THREADS")
        if requested_threads:
            scene.render.threads_mode = "FIXED"
            scene.render.threads = max(1, int(requested_threads))
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        preferences = bpy.context.preferences.addons["cycles"].preferences
        requested = os.environ.get("BLENDER_CYCLES_DEVICE")
        candidates = (
            (requested,)
            if requested
            else (("METAL",) if sys.platform == "darwin" else ("OPTIX", "CUDA"))
        )
        selected = None
        selected_devices = []
        for backend in candidates:
            try:
                preferences.compute_device_type = backend
                preferences.refresh_devices()
                available = [device for device in preferences.devices if device.type == backend]
                if not available:
                    continue
                for device in preferences.devices:
                    device.use = device.type == backend
                selected_devices = [device.name for device in available]
                selected = backend
                break
            except Exception as exc:
                print(f"Cycles {backend} unavailable: {exc}")
        scene.cycles.device = "GPU" if selected else "CPU"
        scene["robot_rendering_cycles_device"] = selected or "CPU"
        scene["robot_rendering_cycles_devices"] = selected_devices or ["CPU"]
        print(
            f"Cycles render device: {selected or 'CPU'} "
            f"({', '.join(selected_devices) if selected_devices else 'CPU'})"
        )
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = float(
            os.environ.get("BLENDER_ADAPTIVE_THRESHOLD", "0.001")
        )
        requested_denoiser = os.environ.get("BLENDER_DENOISER", "AUTO").upper()
        scene.cycles.use_denoising = requested_denoiser != "NONE"
        if scene.cycles.use_denoising:
            denoiser = (
                "OPTIX"
                if requested_denoiser == "AUTO" and selected == "OPTIX"
                else (
                    "OPENIMAGEDENOISE"
                    if requested_denoiser == "AUTO"
                    else requested_denoiser
                )
            )
            scene.cycles.denoiser = denoiser
        scene.cycles.max_bounces = int(os.environ.get("BLENDER_MAX_BOUNCES", "12"))
        scene.cycles.diffuse_bounces = int(
            os.environ.get("BLENDER_DIFFUSE_BOUNCES", "6")
        )
        scene.cycles.glossy_bounces = int(
            os.environ.get("BLENDER_GLOSSY_BOUNCES", "8")
        )
        scene.cycles.transmission_bounces = int(
            os.environ.get("BLENDER_TRANSMISSION_BOUNCES", "8")
        )
        scene.cycles.transparent_max_bounces = int(
            os.environ.get("BLENDER_TRANSPARENT_BOUNCES", "8")
        )
        print(
            "Cycles throughput settings: "
            f"persistent_data={scene.render.use_persistent_data}, "
            f"threads={scene.render.threads_mode}/{scene.render.threads}, "
            f"denoiser={scene.cycles.denoiser if scene.cycles.use_denoising else 'NONE'}, "
            f"samples={scene.cycles.samples}, max_bounces={scene.cycles.max_bounces}"
        )
    with contextlib.suppress(TypeError):
        scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.0


def _worker_build_scene(manifest_path: Path, samples: int, live: bool):
    import bpy
    import numpy as np

    manifest = json.loads(manifest_path.read_text())
    geometry = np.load(manifest_path.parent / manifest["geometry"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)

    _worker_configure_render(manifest, samples=samples, live=live)
    _worker_add_world()
    _worker_add_lighting()
    objects = [
        _worker_create_geom(description, geometry, manifest["meshes"])
        for description in manifest["geoms"]
    ]

    for frame in manifest["frames"]:
        for obj, transform in zip(objects, frame["transforms"], strict=True):
            obj.matrix_world = _worker_matrix(transform["position"], transform["rotation"])
            obj.rotation_mode = "QUATERNION"
            obj.keyframe_insert(data_path="location", frame=frame["frame"])
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame["frame"])
    camera_description = manifest["camera"]
    camera_data = bpy.data.cameras.new(camera_description["name"])
    camera = bpy.data.objects.new(camera_description["name"], camera_data)
    bpy.context.collection.objects.link(camera)
    camera.matrix_world = _worker_matrix(
        camera_description["position"], camera_description["rotation"]
    )
    camera.rotation_mode = "QUATERNION"
    for frame in manifest["frames"]:
        camera.matrix_world = _worker_matrix(
            frame["camera"]["position"], frame["camera"]["rotation"]
        )
        camera.keyframe_insert(data_path="location", frame=frame["frame"])
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame["frame"])
    camera_data.sensor_fit = "VERTICAL"
    camera_data.sensor_height = 32.0
    camera_data.lens = 16.0 / math.tan(math.radians(camera_description["fovy"]) / 2.0)
    camera_data.dof.use_dof = False
    bpy.context.scene.camera = camera
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(1, len(manifest["frames"]))
    bpy.context.scene.frame_set(1)
    return manifest


def _worker_render(args: argparse.Namespace) -> None:
    import bpy

    manifest_path = Path(args.manifest).resolve()
    live = args.blender_worker == "live"
    manifest = _worker_build_scene(manifest_path, args.samples, live=live)
    output = Path(args.output).resolve()
    if args.blender_worker == "still":
        output.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(output)
        bpy.context.scene.frame_set(1)
        bpy.ops.render.render(write_still=True)
    elif args.blender_worker == "frames":
        output.mkdir(parents=True, exist_ok=True)
        for frame in manifest["frames"]:
            frame_number = frame["frame"]
            bpy.context.scene.frame_set(frame_number)
            bpy.context.scene.render.filepath = str(output / f"frame_{frame_number:06d}.png")
            bpy.ops.render.render(write_still=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output))
        screen = bpy.context.screen
        if screen is not None:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.type = "RENDERED"
        with contextlib.suppress(RuntimeError):
            bpy.ops.screen.animation_play()


def _worker_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-worker", choices=["still", "frames", "live"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    separator = sys.argv.index("--") if "--" in sys.argv else -1
    worker_arguments = sys.argv[separator + 1 :] if separator >= 0 else []
    if "--blender-worker" in worker_arguments:
        _worker_render(_worker_arguments(worker_arguments))
