"""Hover and free fall: the plant integrates Newton's second law with the right gravity."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.utils import rotations

pytestmark = pytest.mark.physics


def test_hover_holds_position(rig_rate):
    """Trimmed hover from rest at 1 m (rotors pre-spun to the analytic speeds).

    Per-motor thrust solves ``A T = [m g, 0, 0, 0]`` with the CAD hub positions (the CoM is
    ~0.1 mm off the hub centroid, so the exact-hover split is not exactly equal). Expectation:
    position drift <= 2 cm (max over the whole run) during 2 s, tilt < 0.5 deg, speed < 2 cm/s.
    Run at the default rate and at 500 Hz.
    """
    plant = rig_rate.new_plant()
    thrust = rig_rate.trim_thrust()
    plant.reset_hover(z=1.0, thrust=thrust)
    u = plant.u_for_thrust(thrust)
    p0 = plant.state().pos_w.copy()
    drift = tilt = speed = 0.0
    for _ in range(rig_rate.steps(2.0)):
        plant.substep(u)
        s = plant.state()
        drift = max(drift, float(np.linalg.norm(s.pos_w - p0)))
        tilt = max(tilt, float(rotations.tilt_angle(s.quat_wxyz)))
        speed = max(speed, float(np.linalg.norm(s.linvel_w)))
    assert drift <= 0.02, f"drift {drift * 100:.2f} cm over 2 s (limit 2 cm)"
    assert np.rad2deg(tilt) < 0.5, f"tilt {np.rad2deg(tilt):.3f} deg (limit 0.5)"
    assert speed < 0.02, f"speed {speed:.4f} m/s (limit 0.02)"


def test_hover_command_holds_short_term(rig_rate):
    """The literal ``hover_command`` (equal thrust split) for 0.5 s: drift <= 2 cm.

    Equal split is only approximately hover (see ``Rig.trim_thrust``): the CAD CoM-to-hub
    asymmetry gives a small residual pitch torque, so only a short horizon is asserted.
    """
    plant = rig_rate.new_plant()
    plant.reset_hover(z=1.0)
    u = plant.hover_u()
    p0 = plant.state().pos_w.copy()
    drift = 0.0
    for _ in range(rig_rate.steps(0.5)):
        plant.substep(u)
        drift = max(drift, float(np.linalg.norm(plant.state().pos_w - p0)))
    assert drift <= 0.02, f"drift {drift * 100:.2f} cm over 0.5 s (limit 2 cm)"


def test_hover_command_gives_weight_thrust(rig):
    """``hover_command`` maps through the motor model to k_f*w^2 = m*g/4 per motor (rel 1e-6)."""
    plant = rig.new_plant()
    u = plant.hover_u()
    assert u.shape == (4,)
    assert np.all((u > 0.0) & (u < 1.0)), f"hover command {u} not strictly inside (0, 1)"
    omega = np.asarray(plant.motor.command_to_omega(u))
    thrust = rig.k_f * omega**2
    np.testing.assert_allclose(thrust, rig.mass_kg * rig.g / 4.0, rtol=1e-6)


def test_hover_throttle_is_physical(rig):
    """Sanity: hover is well above idle and below max, T/W > 1.5 for a 5-inch quad."""
    t_max = rig.k_f * rig.omega_max**2 * 4.0
    tw = t_max / (rig.mass_kg * rig.g)
    assert tw > 1.5, f"thrust-to-weight {tw:.2f} < 1.5"
    w_h = np.sqrt(rig.mass_kg * rig.g / (4.0 * rig.k_f))
    assert rig.omega_min < w_h < rig.omega_max, f"hover speed {w_h:.0f} rad/s outside motor range"


def test_free_fall_is_gravity(rig):
    """Zero thrust (omega_min = 0, rotors stopped): a = -g (rel 1e-3), no lateral drift.

    The accelerometer measures specific force, so it reads ~0 (|a| < 1e-3 m/s^2) in free fall.
    Displacement follows -g*t^2/2 within 1 % (the symplectic Euler offset is O(dt)).
    """
    plant = rig.new_plant()
    plant.set_params(plant.with_motor(omega_min=0.0))
    plant.reset_level(z=2.0, omega0=0.0)
    u = np.zeros(4)
    n = rig.steps(0.3)
    v_prev = plant.state().linvel_w.copy()
    acc_err = 0.0
    spec_force = 0.0
    for _ in range(n):
        plant.substep(u)
        s = plant.state()
        a_z = (s.linvel_w[2] - v_prev[2]) / rig.dt
        acc_err = max(acc_err, abs(a_z + rig.g) / rig.g)
        spec_force = max(spec_force, float(np.linalg.norm(plant.accel_b())))
        v_prev = s.linvel_w.copy()
    s = plant.state()
    t = n * rig.dt
    assert acc_err < 1e-3, f"|a_z + g|/g = {acc_err:.2e} (limit 1e-3)"
    assert spec_force < 1e-3, f"free-fall accelerometer {spec_force:.2e} m/s^2 (limit 1e-3)"
    assert np.linalg.norm(s.linvel_w[:2]) < 1e-6
    assert np.linalg.norm(s.angvel_b) < 1e-6
    assert float(np.sum(s.thrust)) < 1e-9, "thrust is not zero with stopped rotors"
    fall = 2.0 - s.pos_w[2]
    np.testing.assert_allclose(fall, 0.5 * rig.g * t**2, rtol=1e-2)


def test_at_rest_on_ground_accelerometer_reads_plus_g(rig):
    """Drone resting on the ground plane (no thrust): specific force = +g along body z.

    Dropped from 0.3 m and given 2 s to settle; the mean over the last 0.5 s must be
    (0, 0, g) within 0.5 m/s^2 laterally and 3 % vertically, with speed < 5 cm/s.
    """
    plant = rig.new_plant()
    plant.set_params(plant.with_motor(omega_min=0.0))
    plant.reset_level(z=0.3, omega0=0.0)
    u = np.zeros(4)
    for _ in range(rig.steps(1.5)):
        plant.substep(u)
    acc = []
    for _ in range(rig.steps(0.5)):
        plant.substep(u)
        acc.append(plant.accel_b())
    mean = np.mean(acc, axis=0)
    s = plant.state()
    assert np.linalg.norm(s.linvel_w) < 0.05, f"still moving: {s.linvel_w}"
    assert s.pos_w[2] < 0.3, "drone did not come down onto the ground"
    np.testing.assert_allclose(mean[:2], 0.0, atol=0.5)
    np.testing.assert_allclose(mean[2], rig.g, rtol=0.03)
