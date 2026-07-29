from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotic_renderers import scene


def test_canonical_scene_compiles_with_expected_content():
    model, data = scene.create_model_and_data(320, 180)

    assert model.nq == 7
    assert model.nu == 7
    assert model.ncam == 1
    assert model.nlight == 3
    assert model.nmesh >= 2
    floor_material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_material")
    np.testing.assert_allclose(model.mat_texrepeat[floor_material], [3.0, 3.0])
    for name in (
        "red_box",
        "blue_sphere",
        "gold_cylinder",
        "green_torus_mesh",
        "gold_gem_mesh",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, scene.CAMERA_NAME)
    assert camera_id >= 0
    assert np.isfinite(data.cam_xpos[camera_id]).all()


def test_trajectory_is_deterministic_and_in_joint_ranges():
    model, data = scene.create_model_and_data()
    first = scene.robot_pose_at_time(1.25)
    second = scene.robot_pose_at_time(1.25)

    np.testing.assert_array_equal(first, second)
    scene.set_scene_time(model, data, 1.25)
    np.testing.assert_allclose(data.qpos, first)
    assert np.all(data.qpos >= model.jnt_range[:, 0])
    assert np.all(data.qpos <= model.jnt_range[:, 1])


def test_camera_follows_smooth_periodic_multi_view_orbit():
    times = np.linspace(0.0, scene.CAMERA_SEQUENCE_DURATION, 721)
    positions = np.asarray([scene.camera_position_at_time(time_s) for time_s in times])
    targets = np.asarray([scene.camera_target_at_time(time_s) for time_s in times])
    angles = np.asarray([scene.camera_orbit_angle_at_time(time_s) for time_s in times])
    radii = np.linalg.norm(positions[:, :2] - targets[:, :2], axis=1)

    np.testing.assert_allclose(positions[0], positions[-1], atol=1e-12)
    np.testing.assert_allclose(targets[0], targets[-1], atol=1e-12)
    assert np.isfinite(positions).all()
    assert radii.min() < 1.80
    assert radii.max() > 2.90
    assert positions[:, 2].min() < 1.07
    assert positions[:, 2].max() > 1.69
    assert np.all(np.ptp(targets, axis=0) > [0.09, 0.07, 0.17])
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() < 0.065
    assert np.all(np.diff(angles) > 0.0)
    assert np.isclose(
        scene.camera_orbit_angle_at_time(scene.CAMERA_SEQUENCE_DURATION)
        - scene.camera_orbit_angle_at_time(0.0),
        4.0 * np.pi,
    )
    epsilon = 1e-4
    start_velocity = (
        scene.camera_position_at_time(epsilon) - scene.camera_position_at_time(-epsilon)
    ) / (2.0 * epsilon)
    end_velocity = (
        scene.camera_position_at_time(scene.CAMERA_SEQUENCE_DURATION + epsilon)
        - scene.camera_position_at_time(scene.CAMERA_SEQUENCE_DURATION - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(start_velocity, end_velocity, atol=1e-8)


def test_compiled_camera_matches_analytical_position_and_target():
    model, data = scene.create_model_and_data(320, 180)
    time_s = 12.5
    scene.set_scene_time(model, data, time_s)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, scene.CAMERA_NAME)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_target")
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    expected_forward = scene.camera_target_at_time(time_s) - data.cam_xpos[camera_id]
    expected_forward /= np.linalg.norm(expected_forward)

    np.testing.assert_allclose(data.cam_xpos[camera_id], scene.camera_position_at_time(time_s))
    np.testing.assert_allclose(model.body_pos[target_id], scene.camera_target_at_time(time_s))
    np.testing.assert_allclose(-rotation[:, 2], expected_forward, atol=1e-7)


def test_thirty_second_trajectory_has_720_unique_frame_times():
    times = list(scene.trajectory_times(scene.CAMERA_SEQUENCE_DURATION, 24))

    assert len(times) == 720
    assert times[0] == 0.0
    assert times[-1] == 719 / 24


@pytest.mark.parametrize(("duration", "fps"), ((0.0, 24), (-1.0, 24), (1.0, 0)))
def test_trajectory_rejects_nonpositive_timing(duration, fps):
    with pytest.raises(ValueError):
        list(scene.trajectory_times(duration, fps))


def test_portable_mjcf_bundle_round_trips(tmp_path: Path):
    xml_path = scene.export_mjcf_bundle(tmp_path, 320, 180)
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    assert xml_path.exists()
    assert (tmp_path / "assets" / "link0_0.obj").exists()
    assert model.nq == 7
    assert model.ncam == 1


def test_fr3_assets_resolve_in_source_and_installed_layouts(tmp_path: Path):
    package_root = tmp_path / "site-packages" / "robotic_renderers"
    installed_assets = package_root / "data" / "franka_fr3"
    installed_assets.mkdir(parents=True)
    (installed_assets / "fr3.xml").write_text("<mujoco/>")

    source_root = tmp_path / "checkout"
    source_assets = source_root / "third_party" / "mujoco_menagerie" / "franka_fr3"
    source_assets.mkdir(parents=True)
    (source_assets / "fr3.xml").write_text("<mujoco/>")

    assert scene._resolve_fr3_directory(package_root, source_root) == installed_assets
    assert scene._resolve_fr3_directory(tmp_path / "empty-package", source_root) == source_assets
