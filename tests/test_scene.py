from pathlib import Path

import mujoco
import numpy as np

import scene


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


def test_camera_follows_constant_radius_orbit():
    start = scene.camera_position_at_time(0.0)
    quarter_turn = scene.camera_position_at_time(scene.CAMERA_ORBIT_PERIOD / 4.0)

    assert not np.allclose(start, quarter_turn)
    assert np.isclose(np.linalg.norm(start[:2] - scene.CAMERA_TARGET[:2]), scene.CAMERA_RADIUS)
    assert np.isclose(
        np.linalg.norm(quarter_turn[:2] - scene.CAMERA_TARGET[:2]),
        scene.CAMERA_RADIUS,
    )


def test_portable_mjcf_bundle_round_trips(tmp_path: Path):
    xml_path = scene.export_mjcf_bundle(tmp_path, 320, 180)
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    assert xml_path.exists()
    assert (tmp_path / "assets" / "link0_0.obj").exists()
    assert model.nq == 7
    assert model.ncam == 1
