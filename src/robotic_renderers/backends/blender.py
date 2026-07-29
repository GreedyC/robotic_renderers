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
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND_NAME = "blender"
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

# MuJoCo and Blender describe light output and apparent source size in different
# units. These are the only Blender-specific light compensations: all placement,
# direction, color, cone, range, and shadow intent comes from the scene manifest.
_BLENDER_LIGHT_ENERGY_SCALE = {
    "directional": 1.0,
    "point": 10.0,
    "spot": 20.0,
}
_BLENDER_LIGHT_RADIUS_SCALE = {
    "point": 1.0,
    "spot": 4.0,
}
_BLENDER_SUN_ANGLE_RADIANS = math.radians(3.0)


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
        "device": _cycles_device_description(),
        "binary": str(blender) if blender else None,
        "live": True,
        "record": True,
        "notes": (
            "Cycles path tracing uses Blender's version-matched bundled bpy in a separate "
            "process; PNG frames are encoded to MP4 by the uv environment."
        ),
        "error": error,
    }


def _cycles_device_description() -> str:
    if sys.platform == "darwin":
        return f"Cycles/Metal ({platform.machine()})"
    if sys.platform.startswith("linux"):
        return "Cycles/OptiX or CUDA"
    return f"Cycles ({platform.system()} {platform.machine()})"


def prepare_blender_bundle(
    destination: str | Path,
    times: list[float],
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Compatibility wrapper for the renderer-neutral scene bundle."""
    from robotic_renderers.scene_bundle import prepare_scene_bundle

    return prepare_scene_bundle(destination, times, width, height, fps)


def _new_job(prefix: str) -> Path:
    from robotic_renderers.defaults import cache_directory

    jobs = cache_directory() / "jobs" / BACKEND_NAME
    jobs.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=jobs))


def _blender_command(
    mode: str,
    manifest: Path,
    output: Path,
    samples: int,
    warmup: int = 0,
) -> list[str]:
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
            "--warmup",
            str(warmup),
        ]
    )
    return command


def _start_blender(
    mode: str,
    manifest: Path,
    output: Path,
    samples: int,
    warmup: int = 0,
    capture_output: bool = True,
) -> subprocess.Popen:
    options: dict[str, object] = {
        "start_new_session": os.name == "posix",
    }
    if capture_output:
        options.update(
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            }
        )
    return subprocess.Popen(
        _blender_command(mode, manifest, output, samples, warmup=warmup),
        **options,
    )


def _run_blender(
    mode: str,
    manifest: Path,
    output: Path,
    samples: int,
    warmup: int = 0,
) -> float:
    from robotic_renderers.processes import terminate_and_reap

    started = time.perf_counter()
    process = _start_blender(mode, manifest, output, samples, warmup=warmup)
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        terminate_and_reap({BACKEND_NAME: process})
        raise
    if process.returncode != 0:
        details = ((stdout or "") + "\n" + (stderr or ""))[-12000:]
        raise RuntimeError(f"Blender {mode} failed with exit {process.returncode}:\n{details}")
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

    from robotic_renderers.media import atomic_output_path, image_digest

    output = Path(output_path).expanduser().resolve()
    job = _new_job("blender-still-")
    try:
        manifest = prepare_blender_bundle(job, [time_s], width, height, fps=1)
        with atomic_output_path(output) as (staged_output, final_output):
            render_seconds = _run_blender("still", manifest, staged_output, samples)
            with Image.open(staged_output) as rendered:
                if rendered.size != (width, height):
                    raise RuntimeError(
                        f"Blender still has size {rendered.size}; expected {(width, height)}"
                    )
                image = np.asarray(rendered.convert("RGB"))
        result = {
            "backend": BACKEND_NAME,
            "output": str(final_output),
            "seconds": render_seconds,
            "sha256": image_digest(image),
            "width": width,
            "height": height,
            "samples": samples,
            "device": _cycles_device_description(),
        }
        if keep_job:
            result["job"] = str(job)
        return result
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
        manifest = prepare_blender_bundle(job, [time_s], width, height, fps=1)
        _run_blender("still", manifest, output, samples)
        with Image.open(output) as rendered:
            if rendered.size != (width, height):
                raise RuntimeError(
                    f"Blender frame has size {rendered.size}; expected {(width, height)}"
                )
            return np.asarray(rendered.convert("RGB"))
    finally:
        shutil.rmtree(job, ignore_errors=True)


def record_video(
    output_path: str | Path,
    duration: float = 30.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    samples: int = 256,
    keep_frames: bool = False,
) -> dict[str, object]:
    from PIL import Image

    from robotic_renderers.media import write_video
    from robotic_renderers.scene import trajectory_times

    output = Path(output_path).expanduser().resolve()
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
                    if image.size != (width, height):
                        raise RuntimeError(
                            f"Blender frame {frame_path.name} has size {image.size}; "
                            f"expected {(width, height)}"
                        )
                    yield image.convert("RGB").copy()

        encoded_output, frame_count, encode_seconds = write_video(frames(), output, fps=fps)
        saved_frames = None
        if keep_frames:
            saved_frames = output.with_suffix("").with_name(output.stem + "_frames")
            if saved_frames.exists():
                shutil.rmtree(saved_frames)
            shutil.copytree(frame_directory, saved_frames)
        result = {
            "backend": BACKEND_NAME,
            "output": str(encoded_output),
            "frames": frame_count,
            "render_seconds": render_seconds,
            "encode_seconds": encode_seconds,
            "seconds": render_seconds + encode_seconds,
            "video_fps": fps,
            "render_fps": frame_count / max(render_seconds, 1e-9),
            "width": width,
            "height": height,
            "samples": samples,
            "device": _cycles_device_description(),
        }
        if saved_frames is not None:
            result["frame_directory"] = str(saved_frames)
        return result
    finally:
        shutil.rmtree(job, ignore_errors=True)


def render_live(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float | None = None,
    samples: int = 64,
) -> dict[str, object]:
    from robotic_renderers.processes import terminate_and_reap
    from robotic_renderers.scene import CAMERA_SEQUENCE_DURATION, trajectory_times

    loop_duration = duration if duration is not None else CAMERA_SEQUENCE_DURATION
    job = _new_job("blender-live-")
    process = None
    started = time.perf_counter()
    stopped_at_deadline = False
    try:
        manifest = prepare_blender_bundle(
            job, list(trajectory_times(loop_duration, fps)), width, height, fps
        )
        blend_file = job / "interactive.blend"
        process = _start_blender(
            "live",
            manifest,
            blend_file,
            samples,
            capture_output=False,
        )
        if duration is None:
            return_code = process.wait()
        else:
            try:
                return_code = process.wait(timeout=max(0.0, duration))
            except subprocess.TimeoutExpired:
                stopped_at_deadline = True
                return_code = terminate_and_reap({BACKEND_NAME: process})[BACKEND_NAME]
        if return_code != 0 and not stopped_at_deadline:
            raise RuntimeError(f"Blender live view exited with status {return_code}")
        return {
            "backend": BACKEND_NAME,
            "pid": process.pid,
            "seconds": time.perf_counter() - started,
            "return_code": return_code,
            "stopped_at_deadline": stopped_at_deadline,
            "notes": "The Blender process was reaped and its temporary scene was removed.",
        }
    finally:
        if process is not None and process.poll() is None:
            terminate_and_reap({BACKEND_NAME: process})
        shutil.rmtree(job, ignore_errors=True)


def benchmark(
    frame_count: int = 3,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    samples: int = 32,
) -> dict[str, object]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    job = _new_job("blender-benchmark-")
    frame_directory = job / "frames"
    frame_directory.mkdir()
    try:
        times = [index / fps for index in range(warmup + frame_count)]
        manifest = prepare_blender_bundle(job, times, width, height, fps)
        process_seconds = _run_blender(
            "benchmark",
            manifest,
            frame_directory,
            samples,
            warmup=warmup,
        )
        result_path = frame_directory / "benchmark.json"
        if not result_path.is_file():
            raise RuntimeError("Blender benchmark returned no timing result")
        worker_result = json.loads(result_path.read_text())
        measured_frames = int(worker_result["frames"])
        render_seconds = float(worker_result["render_seconds"])
        if measured_frames != frame_count:
            raise RuntimeError(
                f"Blender benchmark measured {measured_frames} frames; expected {frame_count}"
            )
        return {
            "backend": BACKEND_NAME,
            "device": worker_result.get("device", _cycles_device_description()),
            "width": width,
            "height": height,
            "samples": samples,
            "frames": frame_count,
            "warmup": warmup,
            "init_seconds": float(worker_result["init_seconds"]),
            "render_seconds": render_seconds,
            "process_seconds": process_seconds,
            "render_fps": frame_count / max(render_seconds, 1e-9),
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


def _worker_light_settings(description):
    light_type = description["type"]
    blender_types = {
        "directional": "SUN",
        "point": "POINT",
        "spot": "SPOT",
    }
    try:
        blender_type = blender_types[light_type]
        energy_scale = _BLENDER_LIGHT_ENERGY_SCALE[light_type]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported canonical light type {light_type!r}") from exc

    settings = {
        "name": description["name"],
        "type": blender_type,
        "position": tuple(description["position"]),
        "direction": tuple(description["direction"]),
        "color": tuple(description["diffuse"]),
        "energy": float(description["intensity"]) * energy_scale,
        "cast_shadow": bool(description["cast_shadow"]),
        "range": float(description["range"]),
    }
    if light_type == "directional":
        settings["angle"] = _BLENDER_SUN_ANGLE_RADIANS
    else:
        settings["radius"] = (
            float(description["bulb_radius"]) * _BLENDER_LIGHT_RADIUS_SCALE[light_type]
        )
    if light_type == "spot":
        exponent = max(0.0, float(description["exponent"]))
        settings["spot_size"] = math.radians(float(description["cutoff"]))
        settings["spot_blend"] = 1.0 / (1.0 + exponent)
    return settings


def _worker_add_lighting(descriptions):
    import bpy
    from mathutils import Vector

    for description in descriptions:
        settings = _worker_light_settings(description)
        light_data = bpy.data.lights.new(name=settings["name"], type=settings["type"])
        light_data.energy = settings["energy"]
        light_data.color = settings["color"]
        light_data.use_shadow = settings["cast_shadow"]
        if settings["type"] == "SUN":
            light_data.angle = settings["angle"]
        else:
            light_data.shadow_soft_size = settings["radius"]
            light_data.use_custom_distance = True
            light_data.cutoff_distance = settings["range"]
        if settings["type"] == "SPOT":
            light_data.spot_size = settings["spot_size"]
            light_data.spot_blend = settings["spot_blend"]

        light = bpy.data.objects.new(name=settings["name"], object_data=light_data)
        bpy.context.collection.objects.link(light)
        light.location = settings["position"]
        _worker_point_light(light, Vector(settings["direction"]))


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
        scene.render.use_persistent_data = _worker_environment_flag("BLENDER_PERSISTENT_DATA", True)
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
                else ("OPENIMAGEDENOISE" if requested_denoiser == "AUTO" else requested_denoiser)
            )
            scene.cycles.denoiser = denoiser
        scene.cycles.max_bounces = int(os.environ.get("BLENDER_MAX_BOUNCES", "12"))
        scene.cycles.diffuse_bounces = int(os.environ.get("BLENDER_DIFFUSE_BOUNCES", "6"))
        scene.cycles.glossy_bounces = int(os.environ.get("BLENDER_GLOSSY_BOUNCES", "8"))
        scene.cycles.transmission_bounces = int(os.environ.get("BLENDER_TRANSMISSION_BOUNCES", "8"))
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
    _worker_add_lighting(manifest["lights"])
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
    init_started = time.perf_counter()
    manifest = _worker_build_scene(manifest_path, args.samples, live=live)
    init_seconds = time.perf_counter() - init_started
    output = Path(args.output).resolve()
    if args.blender_worker == "still":
        output.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(output)
        bpy.context.scene.frame_set(1)
        bpy.ops.render.render(write_still=True)
    elif args.blender_worker in {"frames", "benchmark"}:
        output.mkdir(parents=True, exist_ok=True)
        if args.blender_worker == "benchmark" and not 0 <= args.warmup < len(manifest["frames"]):
            raise ValueError(
                f"Benchmark warmup {args.warmup} leaves no measured frames "
                f"out of {len(manifest['frames'])}"
            )
        render_started = None
        for index, frame in enumerate(manifest["frames"]):
            frame_number = frame["frame"]
            bpy.context.scene.frame_set(frame_number)
            bpy.context.scene.render.filepath = str(output / f"frame_{frame_number:06d}.png")
            if args.blender_worker == "benchmark" and index == args.warmup:
                render_started = time.perf_counter()
            bpy.ops.render.render(write_still=args.blender_worker != "benchmark")
        if args.blender_worker == "benchmark":
            render_seconds = time.perf_counter() - render_started
            scene = bpy.context.scene
            device = scene.get("robot_rendering_cycles_device", "CPU")
            devices = list(scene.get("robot_rendering_cycles_devices", ["CPU"]))
            (output / "benchmark.json").write_text(
                json.dumps(
                    {
                        "device": f"Cycles/{device} ({', '.join(devices)})",
                        "frames": len(manifest["frames"]) - args.warmup,
                        "warmup": args.warmup,
                        "init_seconds": init_seconds,
                        "render_seconds": render_seconds,
                    },
                    separators=(",", ":"),
                )
            )
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
    parser.add_argument(
        "--blender-worker",
        choices=["still", "frames", "benchmark", "live"],
        required=True,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    separator = sys.argv.index("--") if "--" in sys.argv else -1
    worker_arguments = sys.argv[separator + 1 :] if separator >= 0 else []
    if "--blender-worker" in worker_arguments:
        _worker_render(_worker_arguments(worker_arguments))
