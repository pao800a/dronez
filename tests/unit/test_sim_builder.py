"""MjSpec builder: structure, inertial, gear sign derivation, sensors, error handling."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from dronez.cad.artifacts import MissingModelError, ModelArtifacts, load_artifacts
from dronez.cad.schema import DroneConfig
from dronez.sim import SimConfig, build_model, build_model_from_artifacts, motor_gear
from dronez.sim.builder import BODY_NAME, build_spec

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def built(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> tuple[mujoco.MjModel, object]:
    return build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig())


def test_dimensions_and_sensors(built: tuple[mujoco.MjModel, object]) -> None:
    model, index = built
    assert (model.nq, model.nv, model.nu) == (7, 6, 4)
    sl = index.sensor_slices  # type: ignore[attr-defined]
    for name, dim in [("gyro", 3), ("accelerometer", 3), ("framequat", 4), ("framepos", 3),
                      ("framelinvel", 3)]:  # fmt: skip
        assert sl[name].stop - sl[name].start == dim
    assert model.opt.timestep == pytest.approx(0.001)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert model.opt.gravity[2] == pytest.approx(-9.81)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground") >= 0


def test_inertial_matches_mass_props(
    built: tuple[mujoco.MjModel, object], sim_art: ModelArtifacts
) -> None:
    model, index = built
    b = index.body_id  # type: ignore[attr-defined]
    mp = sim_art.mass_props
    assert model.body_mass[b] == pytest.approx(mp.mass_kg)
    np.testing.assert_allclose(model.body_ipos[b], 0.0)  # body frame origin at the CoM
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[b])
    tensor = rot.reshape(3, 3) @ np.diag(model.body_inertia[b]) @ rot.reshape(3, 3).T
    np.testing.assert_allclose(tensor, np.array(mp.inertia_com_kg_m2), atol=2e-8)
    nom = index.nominal  # type: ignore[attr-defined]
    assert nom.mass == pytest.approx(mp.mass_kg)
    np.testing.assert_allclose(nom.inertia, model.body_inertia[b])


def test_gear_sign_derived_from_spin(
    built: tuple[mujoco.MjModel, object], sim_cfg_drone: DroneConfig
) -> None:
    model, index = built
    ratio = sim_cfg_drone.propulsion.k_m / sim_cfg_drone.propulsion.k_f
    spin = np.array(sim_cfg_drone.spin_signs, dtype=float)
    np.testing.assert_array_equal(spin, [1, 1, -1, -1])  # M0 FR ccw, M1 RL ccw, M2 FL cw, M3 RR cw
    gear = model.actuator_gear[index.actuator_ids]  # type: ignore[attr-defined]
    np.testing.assert_allclose(gear[:, :5], np.tile([0, 0, 1, 0, 0], (4, 1)))
    np.testing.assert_allclose(gear[:, 5], -spin * ratio)
    # the helper is general: any spin vector, any ratio
    g = motor_gear(np.array([1.0, -1.0]), 0.02)
    np.testing.assert_allclose(g[:, 5], [-0.02, 0.02])


def test_actuator_applies_force_and_reaction(built: tuple[mujoco.MjModel, object]) -> None:
    """ctrl is thrust in N: M0 (CCW, front-right) at 1 N gives Fz=1, tau_x<0, tau_y<0, tau_z<0."""
    model, _ = built
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    mujoco.mj_forward(model, data)
    fz, tx, ty, tz = data.qfrc_actuator[[2, 3, 4, 5]]
    assert fz == pytest.approx(1.0)
    assert tx < 0.0  # right motor up -> roll left (rotation about -x)
    assert ty < 0.0  # front motor up -> nose up (rotation about -y)
    assert tz == pytest.approx(-model.actuator_gear[0, 5].__abs__())  # CCW rotor -> body torque -z


def test_sites_and_defaults(
    built: tuple[mujoco.MjModel, object], sim_art: ModelArtifacts, sim_cfg_drone: DroneConfig
) -> None:
    model, index = built
    sites = model.site_pos[index.motor_site_ids]  # type: ignore[attr-defined]
    np.testing.assert_allclose(sites, np.array(sim_art.mass_props.motor_hubs_b_m))
    com = np.array(sim_art.mass_props.com_c_m)
    imu = model.site_pos[index.imu_site_id]  # type: ignore[attr-defined]
    np.testing.assert_allclose(imu, np.array(sim_cfg_drone.geometry.imu_position_m) - com)
    assert not model.actuator_ctrllimited.any()
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1.0)
    np.testing.assert_allclose(model.actuator_biasprm, 0.0)
    assert (model.actuator_dyntype == 0).all()
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1) == BODY_NAME


def test_prop_disk_toggle(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> None:
    off, _ = build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig())
    on, _ = build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig(prop_disk_collision=True))
    assert on.ngeom == off.ngeom + 4
    no_ground, _ = build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig(ground_plane=False))
    assert no_ground.ngeom == off.ngeom - 1


@pytest.mark.parametrize("name", ["euler", "implicit", "implicitfast", "rk4"])
def test_integrator_option(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts, name: str) -> None:
    spec = build_spec(sim_cfg_drone, sim_art, SimConfig(integrator=name))  # type: ignore[arg-type]
    model = spec.compile()
    assert model.opt.integrator == getattr(mujoco.mjtIntegrator, f"mjINT_{name.upper()}")


def test_missing_cache_is_a_clear_error(
    sim_cfg_drone: DroneConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRONEZ_DATA_DIR", str(tmp_path))
    with pytest.raises(MissingModelError, match="dronez-build-model"):
        build_model(sim_cfg_drone, SimConfig())


def _read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    tri = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(raw[84 : 84 + 50 * n], dtype=tri)["v"].reshape(-1, 3).astype(np.float64)


def test_real_cache_meshes_are_visual_only(sim_cfg_drone: DroneConfig) -> None:
    """With the built CAD cache: mesh geoms have no contacts and render where the CAD put them."""
    try:
        art = load_artifacts(sim_cfg_drone)
    except MissingModelError:
        pytest.skip("CAD cache not built (uv run dronez-build-model)")
    model, index = build_model_from_artifacts(sim_cfg_drone, art, SimConfig())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = np.array(art.mass_props.com_c_m)
    for part in art.mass_props.parts:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vis_{part.name}")
        assert g >= 0
        assert model.geom_contype[g] == 0
        assert model.geom_conaffinity[g] == 0
        # MuJoCo recentres meshes; compare the world-frame bounding box with the STL (frame C - CoM)
        mid = model.geom_dataid[g]
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[adr : adr + num]
        world = (
            verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - data.xpos[index.body_id]
        )  # type: ignore[attr-defined]
        stl = _read_binary_stl(art.meshes[part.name]) - com
        np.testing.assert_allclose(world.min(axis=0), stl.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(world.max(axis=0), stl.max(axis=0), atol=1e-6)
