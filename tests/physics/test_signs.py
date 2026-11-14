"""Torque signs and magnitudes in FLU (docs/conventions.md "Sign sanity").

Motor identities are hard-coded from the convention table (M0 FR CCW, M1 RL CCW, M2 FL CW,
M3 RR CW); the expected rates are computed from the YAML/CAD constants only (hub positions and
inertia tensor from ``mass_props.json``, k_f, k_m, tau_m from the drone YAML), never from the sim.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.physics

RIGHT, LEFT = [0, 3], [1, 2]
FRONT, REAR = [0, 2], [1, 3]
CW, CCW = [2, 3], [0, 1]

# (plus motors, minus motors, axis index of the expected rotation, expected sign,
#  differential d with T = T_h (1 +/- d), magnitude tolerance vs the analytic rate)
SIGN_CASES = [
    pytest.param((RIGHT, LEFT, 0, -1, 0.1, 0.08), id="right-up-rolls-left"),
    pytest.param((FRONT, REAR, 1, -1, 0.1, 0.08), id="front-up-nose-up"),
    pytest.param((CW, CCW, 2, +1, 0.3, 0.40), id="cw-up-yaws-left"),
]


def _expected_rates(rig, thrust_cmd: np.ndarray, n: int) -> np.ndarray:
    """Body rates after n steps from the YAML constants: exact first-order lag, Euler steps.

    Torque about the CoM: tau_x = sum y_i T_i, tau_y = -sum x_i T_i, tau_z = -sum s_i k_m w_i^2.
    Starts at hover speed; the rate then follows I^-1 * integral of the torque.
    """
    t_h = rig.mass_kg * rig.g / 4.0
    w_h = np.sqrt(t_h / rig.k_f)
    w_cmd = np.sqrt(thrust_cmd / rig.k_f)
    inv_i = np.linalg.inv(rig.inertia)
    rate = np.zeros(3)
    for k in range(1, n + 1):
        w = w_cmd + (w_h - w_cmd) * np.exp(-k * rig.dt / rig.tau_m)
        thrust = rig.k_f * w**2
        tau = np.array(
            [
                np.sum(rig.hubs[:, 1] * thrust),
                -np.sum(rig.hubs[:, 0] * thrust),
                -np.sum(rig.spin * rig.k_m * w**2),
            ]
        )
        rate += inv_i @ tau * rig.dt
    return rate


@pytest.mark.parametrize("case", SIGN_CASES)
def test_differential_sign(rig, case):
    """Small differential from level hover, 100 ms: the body rate has the table's sign.

    Expectation: omega[axis] * sign > 0, the other two axes stay below 10 % of the main one
    (pure mixer pattern; only CAD inertia products couple), and the main-axis rate matches the
    analytic value from the YAML constants within ``tol`` (8 % roll/pitch; 40 % yaw, which is
    also affected by rotor-spin-up reaction torque if the sim models it).
    """
    plus, minus, axis, sign, d, tol = case
    plant = rig.new_plant()
    plant.reset_hover(z=1.0)
    t_h = rig.mass_kg * rig.g / 4.0
    thrust = np.full(4, t_h)
    thrust[plus] *= 1.0 + d
    thrust[minus] *= 1.0 - d
    u = plant.u_for_thrust(thrust)
    n = rig.steps(0.1)
    for _ in range(n):
        plant.substep(u)
    w = plant.state().angvel_b
    main = w[axis]
    others = np.delete(w, axis)
    assert main * sign > 0.0, f"wrong sign: omega_b = {w} rad/s, expected sign {sign:+d} on {axis}"
    assert np.all(np.abs(others) < 0.1 * abs(main)), f"cross-axis leakage: omega_b = {w} rad/s"
    expected = _expected_rates(rig, thrust, n)[axis]
    assert abs(main - expected) <= tol * abs(expected), (
        f"omega_b[{axis}] = {main:.4f} rad/s, analytic {expected:.4f} rad/s (tol {tol:.0%})"
    )


def test_steady_wrench_matches_mixer_convention(rig):
    """Constant, unequal rotor speeds (no lag motion): body accel = I^-1 tau, vertical = F/m.

    tau_x = sum y_i T_i, tau_y = -sum x_i T_i, tau_z = -sum s_i k_m w_i^2 (reaction torque
    opposite to the rotor spin), T_i = k_f w_i^2. After 20 steps: rates within 2 % of
    I^-1 tau * t (atol 2 % of the largest component), vertical speed within 2 % of
    (sum T / m - g) * t (the tilt is < 1e-3 rad so the thrust axis is vertical).
    """
    plant = rig.new_plant()
    w_h = np.sqrt(rig.mass_kg * rig.g / (4.0 * rig.k_f))
    omega = w_h * (1.0 + np.array([0.10, -0.05, 0.08, -0.12]))
    thrust = rig.k_f * omega**2
    plant.reset_level(z=1.0, omega0=omega)
    u = plant.u_for_thrust(thrust)
    n = 20
    for _ in range(n):
        plant.substep(u)
    s = plant.state()
    t = n * rig.dt
    tau = np.array(
        [
            np.sum(rig.hubs[:, 1] * thrust),
            -np.sum(rig.hubs[:, 0] * thrust),
            -np.sum(rig.spin * rig.k_m * omega**2),
        ]
    )
    expected_rate = np.linalg.solve(rig.inertia, tau) * t
    np.testing.assert_allclose(
        s.angvel_b, expected_rate, rtol=0.02, atol=0.02 * np.max(np.abs(expected_rate))
    )
    expected_vz = (np.sum(thrust) / rig.mass_kg - rig.g) * t
    np.testing.assert_allclose(s.linvel_w[2], expected_vz, rtol=0.02, atol=1e-4)
    np.testing.assert_allclose(np.asarray(s.omega).ravel(), omega, rtol=1e-6)
