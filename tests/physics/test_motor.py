"""Motor model: first-order lag with time constant tau_m, exact discretisation."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.physics

TARGET = 1.0 - np.exp(-1.0)  # 0.6321


@pytest.mark.parametrize("dt", [0.0005, 0.001, 0.002, 0.005])
def test_motor_step_response_at_tau(rig, dt):
    """Pure MotorModel: after t = tau_m a step has reached 63.2 % of its size (+-5 % relative).

    Also exact for any dt that divides tau_m (the discretisation is 1 - exp(-dt/tau), so the
    result must not depend on the step size: tolerance 1e-6).
    """
    plant = rig.new_plant()
    m = plant.motor
    w0, w1 = 600.0, 1400.0  # inside [omega_min, omega_max]
    m.reset(np.full(4, w0))
    n = round(rig.tau_m / dt)
    thrust = None
    for _ in range(n):
        thrust = m.step(np.full(4, w1), dt)
    omega = np.sqrt(np.asarray(thrust) / rig.k_f)
    frac = (omega - w0) / (w1 - w0)
    assert np.all(np.abs(frac - TARGET) <= 0.05 * TARGET), f"fraction {frac} vs {TARGET:.4f}"
    np.testing.assert_allclose(frac, TARGET, atol=1e-6)


def test_motor_step_response_in_plant(rig):
    """Through DroneSim at the sim rate: a command step 0.6 -> 1.4 of hover thrust is at
    63.2 % (+-5 % relative) of the rotor-speed step after t = tau_m, and settles (< 1 % of the
    step left) after 5 tau_m."""
    plant = rig.new_plant()
    t_h = rig.mass_kg * rig.g / 4.0
    u1 = plant.u_for_thrust(np.full(4, 1.4 * t_h))
    w0 = plant.omega_for_thrust(np.full(4, 0.6 * t_h))
    w1 = plant.omega_for_thrust(np.full(4, 1.4 * t_h))
    plant.reset_level(z=5.0, omega0=w0)
    n_tau = rig.steps(rig.tau_m)
    for _ in range(n_tau):
        plant.substep(u1)
    omega = np.asarray(plant.state().omega).ravel()
    frac = (omega - w0) / (w1 - w0)
    assert np.all(np.abs(frac - TARGET) <= 0.05 * TARGET), f"fraction at tau_m: {frac}"
    for _ in range(rig.steps(4 * rig.tau_m)):
        plant.substep(u1)
    omega = np.asarray(plant.state().omega).ravel()
    assert np.all(np.abs(omega - w1) < 0.01 * (w1 - w0)), "not settled after 5 tau_m"


def test_motor_saturates(rig):
    """Output stays within [k_f w_min^2, k_f w_max^2] for out-of-range commands (rel 1e-9)."""
    plant = rig.new_plant()
    m = plant.motor
    m.reset(np.full(4, 1000.0))
    for _ in range(500):
        hi = np.asarray(m.step(np.full(4, 10.0 * rig.omega_max), 0.001))
    np.testing.assert_allclose(hi, rig.k_f * rig.omega_max**2, rtol=1e-9)
    for _ in range(500):
        lo = np.asarray(m.step(np.zeros(4), 0.001))
    np.testing.assert_allclose(lo, rig.k_f * rig.omega_min**2, rtol=1e-9)
