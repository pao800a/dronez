"""Rotor reaction/gyroscopic torque, wrench frames, CoM force, inverted thrust, optional effects.

Expectations come from the YAML/CAD constants (rig), never from the sim.

Sign of the rotor reaction (s = +1 CCW seen from above, H = sum s_i I_r w_i along +z_b): the
airframe gets ``-dH/dt z_b - Omega x (H z_b)``. A CCW rotor that speeds up is accelerated by the
motor with ``+I_r dw``, so the airframe feels ``-I_r dw`` (clockwise, negative yaw): raising the CCW
pair (or lowering the CW pair) gives an *additional* negative yaw rate ``-I_r sum(s dw) / Izz``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.physics

CCW, CW = [0, 1], [2, 3]


def _i_r(rig) -> float:
    return float(rig.cfg.propulsion.rotor_inertia_kg_m2 or 0.0)


def _run(plant, u, n):
    for _ in range(n):
        plant.substep(u)
    return plant.state()


def test_rotor_reaction_yaw_transient_sign_and_size(rig):
    """CCW pair up, CW pair down (no roll/pitch torque): on - off yaw rate = -I_r sum(s dw)/Izz."""
    assert _i_r(rig) > 0.0
    trim = rig.trim_thrust()
    n = rig.steps(0.02)
    out = {}
    for flag in (True, False):
        p = rig.new_plant(rotor_inertia=flag)
        p.reset_hover(thrust=trim)
        u = p.u_for_thrust(trim)
        u[CCW] += 0.08
        u[CW] -= 0.08
        out[flag] = _run(p, u, n)
    w0 = np.sqrt(trim / rig.k_f)
    d_omega = np.asarray(out[True].omega) - w0
    assert np.sum(rig.spin * d_omega) > 0.0  # CCW pair net faster
    predicted = np.linalg.inv(rig.inertia) @ np.array(
        [0.0, 0.0, -_i_r(rig) * np.sum(rig.spin * d_omega)]
    )
    diff = out[True].angvel_b - out[False].angvel_b
    assert diff[2] < 0.0, "rotor reaction must push yaw clockwise when the CCW pair speeds up"
    assert diff[2] == pytest.approx(predicted[2], rel=0.03)
    assert np.abs(diff[:2]).max() < 0.05 * abs(diff[2])


def test_rotor_inertia_off_matches_pure_aero_yaw(rig):
    """With the flag off the yaw rate is the old analytic k_m result only."""
    trim = rig.trim_thrust()
    n = rig.steps(0.02)
    p = rig.new_plant(rotor_inertia=False)
    p.reset_hover(thrust=trim)
    u = p.u_for_thrust(trim)
    u[CCW] += 0.08
    u[CW] -= 0.08
    s = _run(p, u, n)
    w_cmd = rig.omega_min + u * (rig.omega_max - rig.omega_min)
    w0 = np.sqrt(trim / rig.k_f)
    rate = np.zeros(3)
    inv_i = np.linalg.inv(rig.inertia)
    for k in range(1, n + 1):
        w = w_cmd + (w0 - w_cmd) * np.exp(-k * rig.dt / rig.tau_m)
        tau_z = -np.sum(rig.spin * rig.k_m * w**2)
        rate += inv_i @ np.array([0.0, 0.0, tau_z]) * rig.dt
    assert s.angvel_b[2] == pytest.approx(rate[2], rel=0.01)


def test_gyroscopic_torque_precession(rig):
    """Net rotor momentum H and roll rate Omega_x give a pitch torque +H Omega_x."""
    omega0 = np.array([900.0, 900.0, 500.0, 500.0])
    h = _i_r(rig) * float(np.sum(rig.spin * omega0))
    n = rig.steps(0.01)
    res = {}
    for flag in (True, False):
        p = rig.new_plant(rotor_inertia=flag)
        p.reset_level(z=5.0, omega0=omega0, qvel=np.array([0, 0, 0, 2.0, 0, 0]))
        u = (omega0 - rig.omega_min) / (rig.omega_max - rig.omega_min)
        res[flag] = _run(p, u, n)
    expected = np.linalg.inv(rig.inertia) @ np.array([0.0, h * 2.0, 0.0]) * n * rig.dt
    diff = res[True].angvel_b - res[False].angvel_b
    assert h > 0.0
    assert diff[1] > 0.0
    assert diff[1] == pytest.approx(expected[1], rel=0.1)


def test_set_wrench_torque_is_world_frame_under_yaw(rig):
    """At yaw 90 deg body +y points to world -x, so a world +x torque is a body -y torque."""
    trim = rig.trim_thrust()
    p = rig.new_plant()
    p.reset_level(z=5.0, omega0=np.sqrt(trim / rig.k_f), yaw=np.pi / 2)
    tau = 0.02
    p.sim.set_wrench(np.zeros(3), np.array([tau, 0.0, 0.0]))
    n = rig.steps(0.05)
    s = _run(p, p.u_for_thrust(trim), n)
    expected = np.linalg.inv(rig.inertia) @ np.array([0.0, -tau, 0.0]) * n * rig.dt
    assert s.angvel_b[1] == pytest.approx(expected[1], rel=0.03)
    assert abs(s.angvel_b[0]) < 0.05 * abs(expected[1])
    assert abs(s.angvel_b[2]) < 0.05 * abs(expected[1])


def test_force_at_com_produces_no_rotation_with_com_offset(rig):
    p = rig.new_plant()
    p.set_params(p.replace(com_offset_b_m=np.array([0.01, -0.006, 0.008])))
    u = p.sim.trim_command()
    omega0 = p.sim.motor.command_to_omega(u)
    res = {}
    for force in (np.zeros(3), np.array([1.5, -0.7, 0.9])):
        p.sim.reset(p.sim.default_qpos(5.0), np.zeros(6), omega0)
        p.sim.set_wrench(force, np.zeros(3))
        res[bool(force.any())] = _run(p, u, rig.steps(0.1))
    np.testing.assert_allclose(res[True].angvel_b, res[False].angvel_b, atol=1e-9)
    assert np.linalg.norm(res[True].pos_w - res[False].pos_w) > 1e-3  # the force did act


def test_inverted_thrust_accelerates_downward(rig):
    """Upside down at hover thrust: a = -(T/m + g) = -2 g."""
    trim = rig.trim_thrust()
    p = rig.new_plant()
    p.sim.reset(np.array([0, 0, 5.0, 0, 1.0, 0, 0]), np.zeros(6), np.sqrt(trim / rig.k_f))
    s = _run(p, p.u_for_thrust(trim), rig.steps(0.1))
    assert s.linvel_w[2] == pytest.approx(-2.0 * rig.g * 0.1, rel=0.01)


@pytest.mark.parametrize("flags", [{"rotor_drag": True, "ground_effect": True}])
def test_trajectory_independent_of_true_state_calls(rig, flags):
    """Optional effects read fresh kinematics: interleaved true_state() must not change anything."""
    omega0 = np.array([900.0, 800.0, 850.0, 700.0])
    u = (omega0 - rig.omega_min) / (rig.omega_max - rig.omega_min)
    qpos = []
    for interleave in (False, True):
        p = rig.new_plant(**flags)
        p.sim.reset(
            p.sim.default_qpos(0.12),
            np.array([2.0, 1.0, 0.0, 0.5, -0.3, 0.2]),
            omega0,
        )
        for _ in range(200):
            p.substep(u)
            if interleave:
                p.state()
        qpos.append(p.sim.data.qpos.copy())
    np.testing.assert_array_equal(qpos[0], qpos[1])


def test_ground_effect_scales_thrust_not_yaw_torque(rig):
    omega0 = np.array([1000.0, 1000.0, 800.0, 800.0])
    u = (omega0 - rig.omega_min) / (rig.omega_max - rig.omega_min)
    p = rig.new_plant(ground_effect=True, rotor_inertia=False)
    p.reset_level(z=0.07, omega0=omega0)
    n = rig.steps(0.02)
    s = _run(p, u, n)
    assert s.thrust.sum() > rig.k_f * np.sum(omega0**2) * 1.03  # the gain is active
    tau_z = -np.sum(rig.spin * rig.k_m * omega0**2)
    expected = np.linalg.inv(rig.inertia) @ np.array([0.0, 0.0, tau_z]) * n * rig.dt
    assert s.angvel_b[2] == pytest.approx(expected[2], rel=0.005)
