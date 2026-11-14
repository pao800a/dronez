"""DroneSim: hover/trim, signs, set_params from the nominal snapshot, latency, integrator."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.sim import (
    DroneSim,
    SimConfig,
    build_model_from_artifacts,
    inertia_triangle_ok,
    project_inertia_triangle,
)

pytestmark = pytest.mark.fast


def make_sim(cfg: DroneConfig, art: ModelArtifacts, **sim_kw: object) -> DroneSim:
    sim_cfg = SimConfig(**sim_kw)  # type: ignore[arg-type]
    model, index = build_model_from_artifacts(cfg, art, sim_cfg)
    return DroneSim(cfg, sim_cfg, model=model, index=index)


@pytest.fixture
def sim(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> DroneSim:
    return make_sim(sim_cfg_drone, sim_art)


def run(sim: DroneSim, u: np.ndarray, n: int) -> None:
    for _ in range(n):
        sim.substep(u)


def test_trim_hover_holds_altitude_for_2s(sim: DroneSim) -> None:
    thrust = sim.trim_thrusts()
    assert thrust.sum() == pytest.approx(sim.weight_n())
    u = sim.trim_command()
    sim.reset(sim.default_qpos(1.0), np.zeros(6), sim.motor.command_to_omega(u))
    run(sim, u, 2000)
    s = sim.true_state()
    assert abs(s.pos_w[2] - 1.0) < 1e-6
    np.testing.assert_allclose(s.angvel_b, 0.0, atol=1e-9)
    np.testing.assert_allclose(s.accel_b, [0, 0, 9.81], atol=1e-6)  # specific force = +g in hover


def test_equal_split_hover_command_is_close(sim: DroneSim) -> None:
    """The literal hover_command leaves only the ~0.1 mm CAD CoM offset as pitch torque."""
    u = sim.hover_command()
    sim.reset(sim.default_qpos(1.0), np.zeros(6), sim.motor.command_to_omega(u))
    run(sim, u, 500)
    assert abs(sim.true_state().pos_w[2] - 1.0) < 0.02


def test_free_fall_and_accelerometer(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.motor.omega_min = np.zeros(4)  # allow zero thrust
    sim.set_params(p)
    sim.reset(sim.default_qpos(10.0), np.zeros(6), np.zeros(4))
    run(sim, np.zeros(4), 1000)
    s = sim.true_state()
    assert s.pos_w[2] == pytest.approx(10.0 - 0.5 * 9.81 * 1.0**2, abs=1e-3)
    assert s.linvel_w[2] == pytest.approx(-9.81, abs=2e-3)
    np.testing.assert_allclose(s.accel_b, 0.0, atol=1e-6)  # specific force vanishes in free fall


def test_accelerometer_reads_plus_g_at_rest_on_ground(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.motor.omega_min = np.zeros(4)
    sim.set_params(p)
    sim.reset(sim.default_qpos(0.2), np.zeros(6), np.zeros(4))
    run(sim, np.zeros(4), 1500)
    s = sim.true_state()
    np.testing.assert_allclose(s.accel_b, [0, 0, 9.81], atol=1e-3)
    assert 0.0 < s.pos_w[2] < 0.1  # resting on the collision geoms


@pytest.mark.parametrize(
    ("motors", "axis", "sign"),
    [((0, 3), 0, -1), ((0, 2), 1, -1), ((2, 3), 2, +1)],
    ids=["roll(right up)", "pitch(front up)", "yaw(CW pair up)"],
)
def test_response_signs(sim: DroneSim, motors: tuple[int, int], axis: int, sign: int) -> None:
    u = sim.trim_command()
    u[list(motors)] += 0.1
    sim.reset(sim.default_qpos(5.0), np.zeros(6), sim.motor.command_to_omega(sim.trim_command()))
    run(sim, u, 100)
    w = sim.true_state().angvel_b
    assert sign * w[axis] > 0.0
    others = np.delete(np.abs(w), axis)
    assert abs(w[axis]) > 5 * others.max()


def test_latency_delays_command_by_n_physics_steps(sim: DroneSim) -> None:
    n = 7
    p = sim.nominal_params()
    p.latency_steps = n
    sim.set_params(p)
    u0 = sim.trim_command()
    sim.reset(sim.default_qpos(5.0), np.zeros(6), sim.motor.command_to_omega(u0))
    run(sim, u0, 20)
    w_before = sim.motor.omega.copy()
    u1 = u0 + 0.2
    for k in range(n + 1):
        sim.substep(u1)
        if k < n:  # the step command has not reached the motors yet
            np.testing.assert_allclose(sim.motor.omega, w_before)
    assert np.all(sim.motor.omega > w_before)


def test_set_params_writes_from_nominal_and_matches_fresh_compile(
    sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    """Mass/inertia edits + mj_setConst reproduce a fresh compile bit-for-bit; repeated calls with
    the nominal values restore the model exactly (no drift)."""
    mass_new = 0.62
    inertia_new = np.array([2.0e-3, 2.1e-3, 3.6e-3])
    mp = sim_art.mass_props
    art2 = dataclasses.replace(
        sim_art,
        mass_props=mp.model_copy(
            update={"mass_kg": mass_new, "principal_moments_kg_m2": tuple(inertia_new)}
        ),
    )
    a = make_sim(sim_cfg_drone, sim_art)
    ref = make_sim(sim_cfg_drone, art2)
    nominal_snapshot = (
        a.model.body_mass.copy(),
        a.model.body_inertia.copy(),
        a.model.body_ipos.copy(),
    )
    p = a.nominal_params()
    p.mass_kg = mass_new
    p.inertia_diag_kg_m2 = inertia_new
    a.set_params(p)
    u = np.array([0.5, 0.4, 0.6, 0.45])
    for s in (a, ref):
        s.reset(s.default_qpos(2.0), np.array([0, 0, 0, 1.0, 2.0, 3.0]), np.full(4, 700.0))
        run(s, u, 300)
    sa, sr = a.true_state(), ref.true_state()
    np.testing.assert_allclose(sa.pos_w, sr.pos_w, atol=1e-12)
    np.testing.assert_allclose(sa.quat_wxyz, sr.quat_wxyz, atol=1e-12)
    np.testing.assert_allclose(sa.angvel_b, sr.angvel_b, atol=1e-12)
    # stale-constant sanity: without mj_setConst the result would differ (documented in builder)
    for _ in range(3):
        a.set_params(p)
        a.set_params(a.nominal_params())
    np.testing.assert_array_equal(a.model.body_mass, nominal_snapshot[0])
    np.testing.assert_array_equal(a.model.body_inertia, nominal_snapshot[1])
    np.testing.assert_array_equal(a.model.body_ipos, nominal_snapshot[2])


def test_set_params_com_offset_and_gear(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.com_offset_b_m = np.array([0.004, -0.002, 0.001])
    p.motor.k_m = p.motor.k_m * np.array([1.0, 1.1, 0.9, 1.0])
    sim.set_params(p)
    np.testing.assert_allclose(sim.model.body_ipos[sim.index.body_id], p.com_offset_b_m)
    np.testing.assert_allclose(
        sim.model.actuator_gear[sim.index.actuator_ids, 5],
        -sim.index.spin * p.motor.k_m / p.motor.k_f,
    )
    # trim follows the new CoM: zero net torque about it
    u = sim.trim_command()
    sim.reset(sim.default_qpos(3.0), np.zeros(6), sim.motor.command_to_omega(u))
    run(sim, u, 500)
    s = sim.true_state()
    assert np.abs(s.angvel_b).max() < 1e-6
    assert abs(s.pos_w[2] - (3.0 + p.com_offset_b_m[2])) < 1e-6


def test_battery_scale_lowers_max_thrust(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.battery_scale = 0.9
    sim.set_params(p)
    np.testing.assert_allclose(sim.motor.params.omega_max, 0.9 * p.motor.omega_max)
    np.testing.assert_allclose(p.motor.omega_max, sim.nominal_motor.omega_max)  # input untouched


def test_inertia_triangle_inequality() -> None:
    assert inertia_triangle_ok([1.7e-3, 1.9e-3, 3.2e-3])
    assert not inertia_triangle_ok([1.0e-3, 1.0e-3, 2.5e-3])
    assert not inertia_triangle_ok([-1e-3, 2e-3, 2e-3])
    fixed = project_inertia_triangle([1.0e-3, 1.0e-3, 2.5e-3])
    assert inertia_triangle_ok(fixed)
    assert fixed[2] == pytest.approx(2.5e-3)


def test_set_params_rejects_invalid_inertia(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.inertia_diag_kg_m2 = np.array([1.0e-3, 1.0e-3, 2.5e-3])
    with pytest.raises(ValueError, match="triangle"):
        sim.set_params(p)


def test_deterministic_replay(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> None:
    def rollout() -> np.ndarray:
        s = make_sim(sim_cfg_drone, sim_art)
        rng = np.random.default_rng(0)
        s.reset(s.default_qpos(2.0), np.zeros(6), np.full(4, 600.0))
        out = []
        for _ in range(200):
            s.set_wrench(rng.normal(size=3) * 0.1, np.zeros(3))
            s.substep(rng.uniform(0.2, 0.6, 4))
            out.append(s.data.qpos.copy())
        return np.array(out)

    np.testing.assert_array_equal(rollout(), rollout())


def test_default_integrator_beats_euler_on_a_tumble(
    sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    """Torque-free tumble near the intermediate axis at 10 rad/s: relative drift of the rotational
    energy and the world-frame angular momentum over 10 s (evidence for the integrator choice)."""
    assert SimConfig().integrator == "rk4"

    def drift(integrator: str) -> float:
        s = make_sim(sim_cfg_drone, sim_art, integrator=integrator, ground_plane=False)
        m, d, b = s.model, s.data, s.index.body_id
        m.opt.gravity[:] = 0.0
        s.reset(s.default_qpos(10.0), np.array([0, 0, 0, 0.8, 10.0, 0.4]), np.zeros(4))
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[b])
        tensor = rot.reshape(3, 3) @ np.diag(m.body_inertia[b]) @ rot.reshape(3, 3).T

        def energy_and_l() -> tuple[float, np.ndarray]:
            w = d.qvel[3:6]
            return 0.5 * w @ tensor @ w, d.xmat[b].reshape(3, 3) @ (tensor @ w)

        mujoco.mj_forward(m, d)
        e0, l0 = energy_and_l()
        worst = 0.0
        for k in range(10000):
            d.ctrl[:] = 0.0
            mujoco.mj_step(m, d)
            if k % 100 == 0:
                e, ll = energy_and_l()
                worst = max(worst, abs(e - e0) / e0, np.linalg.norm(ll - l0) / np.linalg.norm(l0))
        return worst

    rk4, euler = drift("rk4"), drift("euler")
    assert rk4 < 1e-6
    assert euler > 1e-3
    assert euler > 1e3 * rk4


def test_optional_rotor_drag_and_ground_effect(
    sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    # rotor drag opposes horizontal motion
    drag = make_sim(sim_cfg_drone, sim_art, rotor_drag=True)
    plain = make_sim(sim_cfg_drone, sim_art)
    for s in (drag, plain):
        u = s.trim_command()
        s.reset(s.default_qpos(5.0), np.array([5.0, 0, 0, 0, 0, 0]), s.motor.command_to_omega(u))
        run(s, u, 200)
    assert drag.true_state().linvel_w[0] < plain.true_state().linvel_w[0] - 0.01
    # ground effect: more thrust close to the ground at the same command
    ge = make_sim(sim_cfg_drone, sim_art, ground_effect=True)
    ref = make_sim(sim_cfg_drone, sim_art)
    for s in (ge, ref):
        u = s.trim_command()
        s.reset(s.default_qpos(0.15), np.zeros(6), s.motor.command_to_omega(u))
        run(s, u, 50)
    assert ge.true_state().thrust.sum() > ref.true_state().thrust.sum()


def test_reset_prefills_delay_with_omega0_command(sim: DroneSim) -> None:
    """The first N commands after reset are the one that holds omega0, not the fresh action."""
    n = 5
    p = sim.nominal_params()
    p.latency_steps = n
    sim.set_params(p)
    u0 = sim.trim_command()
    omega0 = sim.motor.command_to_omega(u0)
    sim.reset(sim.default_qpos(5.0), np.zeros(6), omega0)
    for k in range(n + 2):
        sim.substep(np.ones(4))
        if k < n:  # steps 0..N-1 still see the omega0 command: motors do not move
            np.testing.assert_allclose(sim.motor.omega, omega0, rtol=1e-12)
        else:
            assert np.all(sim.motor.omega > omega0)


def test_battery_scale_must_keep_omega_max_above_min(sim: DroneSim) -> None:
    p = sim.nominal_params()
    p.battery_scale = 0.05  # omega_max * 0.05 < omega_min
    with pytest.raises(ValueError, match="omega_max"):
        sim.set_params(p)
    with pytest.raises(ValueError, match="positive"):
        sim.nominal_motor.with_battery_scale(0.0)


def test_rotor_inertia_flag_default_on_and_composes_with_wrench(
    sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    """The rotor torque is added to ``set_wrench``, never overwriting it (symmetric spin-up has
    zero net rotor torque, so on/off must agree and keep the wrench-driven roll rate)."""
    assert SimConfig().rotor_inertia is True
    rates = []
    for flag in (True, False):
        s = make_sim(sim_cfg_drone, sim_art, rotor_inertia=flag)
        s.reset(s.default_qpos(5.0), np.zeros(6), np.full(4, 700.0))
        s.set_wrench(np.zeros(3), np.array([0.01, 0.0, 0.0]))
        run(s, np.full(4, 0.5), 20)
        rates.append(s.true_state().angvel_b)
    assert rates[0][0] > 0.05
    np.testing.assert_allclose(rates[0], rates[1], atol=1e-9)
