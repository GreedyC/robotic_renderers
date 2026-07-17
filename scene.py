"""Canonical FR3 evaluation scene built with MuJoCo's procedural MjSpec API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
FR3_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_fr3"
FR3_XML = FR3_DIR / "fr3.xml"
CAMERA_NAME = "evaluation_camera"
CAMERA_TARGET = np.array([0.0, 0.0, 0.48], dtype=float)
CAMERA_RADIUS = 2.35
CAMERA_HEIGHT = 1.45
CAMERA_START_ANGLE = np.deg2rad(-58.0)
CAMERA_ORBIT_PERIOD = 8.0
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def _texture_slots(rgb_texture: str) -> list[str]:
    slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = rgb_texture
    return slots


def _tune_robot_materials(spec: mujoco.MjSpec) -> None:
    settings = {
        "black": (0.04, 0.26, 0.0),
        "white": (0.2, 0.3, 0.0),
        "red": (0.12, 0.28, 0.05),
        "gray": (0.15, 0.34, 0.08),
        "button_green": (0.2, 0.3, 0.05),
        "button_red": (0.2, 0.3, 0.05),
        "button_blue": (0.2, 0.3, 0.05),
    }
    for name, (specular, roughness, metallic) in settings.items():
        material = spec.material(name)
        if material is not None:
            material.specular = specular
            material.roughness = roughness
            material.metallic = metallic


def _add_studio_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="studio_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.72, 0.8, 0.92],
        rgb2=[0.035, 0.055, 0.09],
        width=1024,
        height=4096,
    )
    spec.add_texture(
        name="floor_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.34, 0.37, 0.4],
        rgb2=[0.16, 0.18, 0.2],
        markrgb=[0.08, 0.09, 0.1],
        width=1024,
        height=1024,
    )
    spec.add_material(
        name="floor_material",
        textures=_texture_slots("floor_checker"),
        texuniform=1,
        texrepeat=[3.0, 3.0],
        rgba=[0.8, 0.82, 0.84, 1.0],
        specular=0.25,
        reflectance=0.16,
        roughness=0.5,
        metallic=0.0,
    )
    spec.add_material(
        name="prop_red",
        rgba=[0.72, 0.055, 0.035, 1.0],
        specular=0.55,
        roughness=0.22,
        metallic=0.05,
    )
    spec.add_material(
        name="prop_blue",
        rgba=[0.025, 0.2, 0.72, 1.0],
        specular=0.65,
        roughness=0.17,
        metallic=0.12,
    )
    spec.add_material(
        name="prop_gold",
        rgba=[0.83, 0.43, 0.055, 1.0],
        specular=0.9,
        roughness=0.16,
        metallic=0.82,
    )
    spec.add_material(
        name="prop_green",
        rgba=[0.04, 0.48, 0.22, 1.0],
        specular=0.72,
        roughness=0.2,
        metallic=0.35,
    )

    torus = spec.add_mesh(
        name="mesh_torus",
        scale=[0.115, 0.115, 0.115],
        smoothnormal=1,
    )
    torus.make_supertorus(48, 0.34, 1.0, 1.0)

    gem = spec.add_mesh(
        name="mesh_gem",
        scale=[0.09, 0.09, 0.12],
        smoothnormal=0,
    )
    gem.make_supersphere(12, 0.35, 0.35)


def _add_scene_geometry(spec: mujoco.MjSpec, width: int, height: int) -> None:
    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, -0.012],
        size=[0.0, 0.0, 0.05],
        material="floor_material",
        friction=[0.95, 0.01, 0.001],
        group=0,
    )

    primitive_box = world.add_body(name="primitive_box", pos=[0.56, 0.29, 0.085])
    primitive_box.add_geom(
        name="red_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.085, 0.085, 0.085],
        material="prop_red",
        group=0,
    )

    primitive_sphere = world.add_body(name="primitive_sphere", pos=[0.78, 0.29, 0.09])
    primitive_sphere.add_geom(
        name="blue_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.09, 0.09, 0.09],
        material="prop_blue",
        group=0,
    )

    primitive_cylinder = world.add_body(name="primitive_cylinder", pos=[0.48, -0.3, 0.12])
    primitive_cylinder.add_geom(
        name="gold_cylinder",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.073, 0.12, 0.073],
        material="prop_gold",
        group=0,
    )

    torus = world.add_body(name="mesh_torus_object", pos=[0.75, 0.0, 0.16])
    torus.add_geom(
        name="green_torus_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="mesh_torus",
        material="prop_green",
        group=0,
    )

    gem = world.add_body(
        name="mesh_gem_object",
        pos=[0.73, -0.29, 0.105],
        axisangle=[0.0, 0.0, 1.0, 0.35],
    )
    gem.add_geom(
        name="gold_gem_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="mesh_gem",
        material="prop_gold",
        group=0,
    )

    target = world.add_body(name="camera_target", pos=CAMERA_TARGET.tolist())
    target.add_site(
        name="camera_target_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.001, 0.001, 0.001],
        rgba=[0.0, 0.0, 0.0, 0.0],
        group=5,
    )
    world.add_camera(
        name=CAMERA_NAME,
        pos=camera_position_at_time(0.0).tolist(),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="camera_target",
        fovy=44.0,
        resolution=[width, height],
        output=int(mujoco.mjtCamOutBit.mjCAMOUT_RGB),
    )

    world.add_light(
        name="sun_key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-1.5, -1.0, 3.0],
        dir=[0.45, 0.2, -1.0],
        castshadow=1,
        intensity=2.8,
        ambient=[0.035, 0.035, 0.04],
        diffuse=[0.95, 0.91, 0.84],
        specular=[1.0, 0.95, 0.88],
    )
    world.add_light(
        name="soft_fill",
        type=mujoco.mjtLightType.mjLIGHT_POINT,
        pos=[0.2, -1.5, 1.45],
        castshadow=0,
        intensity=9.0,
        range=5.0,
        bulbradius=0.25,
        ambient=[0.02, 0.025, 0.035],
        diffuse=[0.34, 0.47, 0.7],
        specular=[0.28, 0.38, 0.6],
    )
    world.add_light(
        name="warm_rim",
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[-0.65, 0.9, 1.8],
        dir=[0.5, -0.6, -1.0],
        castshadow=1,
        intensity=5.0,
        range=4.0,
        cutoff=55.0,
        exponent=2.0,
        diffuse=[0.8, 0.46, 0.25],
        specular=[0.75, 0.42, 0.2],
    )


def _tune_visuals(spec: mujoco.MjSpec, width: int, height: int) -> None:
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.global_.azimuth = 135.0
    spec.visual.global_.elevation = -22.0
    spec.visual.quality.shadowsize = 8192
    spec.visual.quality.offsamples = 8
    spec.visual.quality.numslices = 64
    spec.visual.quality.numstacks = 32
    spec.visual.quality.numquads = 16
    spec.visual.map.znear = 0.01
    spec.visual.map.zfar = 50.0
    spec.visual.map.shadowclip = 4.0
    spec.visual.map.shadowscale = 0.9
    spec.visual.map.fogstart = 5.0
    spec.visual.map.fogend = 12.0
    # Keep the camera headlight disabled: it has no equivalent in the translated
    # backends and would make OpenGL camera-facing surfaces artificially bright.
    spec.visual.headlight.active = 0
    spec.visual.rgba.haze = [0.18, 0.22, 0.3, 1.0]


def create_scene_spec(width: int = 1280, height: int = 720) -> mujoco.MjSpec:
    """Create the canonical scene. All render backends start from this MjSpec."""
    if not FR3_XML.exists():
        raise FileNotFoundError(
            f"Missing FR3 model at {FR3_XML}. Restore third_party/mujoco_menagerie."
        )
    spec = mujoco.MjSpec.from_file(str(FR3_XML))
    spec.modelname = "fr3_renderer_evaluation"
    spec.option.timestep = 0.002
    spec.compiler.usethread = 1
    _tune_robot_materials(spec)
    _add_studio_assets(spec)
    _add_scene_geometry(spec, width, height)
    _tune_visuals(spec, width, height)
    return spec


def create_model_and_data(width: int = 1280, height: int = 720):
    spec = create_scene_spec(width=width, height=height)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_data(model, data)
    return model, data


def reset_data(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    set_scene_time(model, data, 0.0)


def robot_pose_at_time(time_s: float) -> np.ndarray:
    """Return the deterministic pose used by every renderer for fair comparisons."""
    t = float(time_s)
    offsets = np.array(
        [
            0.36 * np.sin(0.55 * t),
            0.22 * np.sin(0.7 * t + 0.4),
            0.28 * np.sin(0.48 * t + 1.2),
            0.2 * np.sin(0.62 * t + 0.8),
            0.34 * np.sin(0.83 * t + 1.7),
            0.28 * np.sin(0.58 * t + 0.3),
            0.42 * np.sin(0.72 * t + 2.0),
        ]
    )
    return HOME_QPOS + offsets


def camera_position_at_time(time_s: float) -> np.ndarray:
    """Return the shared close camera orbit around the robot origin."""
    angle = CAMERA_START_ANGLE + 2.0 * np.pi * float(time_s) / CAMERA_ORBIT_PERIOD
    return np.array(
        [
            CAMERA_TARGET[0] + CAMERA_RADIUS * np.cos(angle),
            CAMERA_TARGET[1] + CAMERA_RADIUS * np.sin(angle),
            CAMERA_HEIGHT,
        ],
        dtype=float,
    )


def set_scene_time(model: mujoco.MjModel, data: mujoco.MjData, time_s: float) -> None:
    pose = robot_pose_at_time(time_s)
    data.qpos[: len(pose)] = pose
    if model.nu:
        data.ctrl[: min(model.nu, len(pose))] = pose[: model.nu]
    data.qvel[:] = 0.0
    data.time = float(time_s)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id >= 0:
        model.cam_pos[camera_id] = camera_position_at_time(time_s)
    mujoco.mj_forward(model, data)


def trajectory_times(duration: float, fps: int):
    frame_count = max(1, int(round(float(duration) * int(fps))))
    for frame_index in range(frame_count):
        yield frame_index / float(fps)


def export_mjcf_bundle(destination: str | Path, width: int = 1280, height: int = 720) -> Path:
    """Write a portable, flattened MJCF and its FR3 assets for external engines."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    assets_destination = destination / "assets"
    if assets_destination.exists():
        shutil.rmtree(assets_destination)
    shutil.copytree(FR3_DIR / "assets", assets_destination)

    spec = create_scene_spec(width=width, height=height)
    spec.meshdir = "assets"
    spec.texturedir = "assets"
    xml_path = destination / "scene.xml"
    spec.compile()
    spec.to_file(str(xml_path))

    # Fail here, close to export, rather than later inside an external renderer.
    mujoco.MjModel.from_xml_path(str(xml_path))
    metadata = {
        "camera": CAMERA_NAME,
        "width": width,
        "height": height,
        "source": str(FR3_XML),
        "menagerie_commit": "71f066ad0be9cd271f7ed58c030243ef157af9f4",
    }
    (destination / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return xml_path


def scene_summary(width: int = 1280, height: int = 720) -> dict[str, object]:
    model, _ = create_model_and_data(width=width, height=height)
    return {
        "model": model.names.decode(errors="replace").split("\x00", 1)[0],
        "bodies": model.nbody,
        "geoms": model.ngeom,
        "meshes": model.nmesh,
        "materials": model.nmat,
        "lights": model.nlight,
        "cameras": model.ncam,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "camera": CAMERA_NAME,
    }
