"""Motor model unit tests (pure numpy, no MuJoCo model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.sim import MotorModel, MotorParams, hover_command, hover_thrust_per_motor
from dronez.sim.motor_model import ground_effect_gain, rotor_drag_force_b

pytestmark = pytest.mark.fast

TAU = 0.03
DT = 0.001


def _params(n_batch: int | None = None) -> MotorParams:
    shape = (4,) if n_batch is None else (n_batch, 4)
    return MotorParams(
        k_f=np.full(shape, 1.91e-6),
        k_m=np.full(shape, 3.06e-8),
        tau_m_s=np.full(shape, TAU),
        omega_min=np.full(shape, 150.0),
        omega_max=np.full(shape, 2170.0),
    )


def test_step_response_63_percent_at_tau() -> None:
    mm = MotorModel(_params())
    mm.reset(np.full(4, 400.0))
    cmd = np.full(4, 1400.0)
    for _ in range(round(TAU / DT)):
        mm.step(cmd, DT)
    frac = (mm.omega - 400.0) / (1400.0 - 400.0)
    np.testing.assert_allclose(frac, 1.0 - np.exp(-1.0), rtol=1e-12)


@pytest.mark.parametrize("dt", [1e-4, 1e-3, 5e-3, 0.03, 0.3, 30.0])
def test_exact_discretization_is_step_size_independent(dt: float) -> None:
    """One step of dt equals n steps of dt/n (exactness), and dt >> tau is stable (no overshoot)."""
    a, b = MotorModel(_params()), MotorModel(_params())
    a.reset(np.full(4, 300.0))
    b.reset(np.full(4, 300.0))
    cmd = np.full(4, 1500.0)
    a.step(cmd, dt)
    for _ in range(10):
        b.step(cmd, dt / 10)
    np.testing.assert_allclose(a.omega, b.omega, rtol=1e-12)
    assert np.all(a.omega <= 1500.0 + 1e-9)
    assert np.all(a.omega >= 300.0)


def test_saturation_and_thrust() -> None:
    mm = MotorModel(_params())
    mm.reset(np.full(4, 2000.0))
    t = mm.step(np.full(4, 1e5), 1.0)  # way above omega_max
    np.testing.assert_allclose(mm.omega, 2170.0)
    np.testing.assert_allclose(t, 1.91e-6 * 2170.0**2)
    mm.step(np.zeros(4), 100.0)  # way below omega_min
    np.testing.assert_allclose(mm.omega, 150.0)


def test_command_to_omega_linear_and_clipped() -> None:
    mm = MotorModel(_params())
    cmd = mm.command_to_omega(np.array([0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(cmd, [150.0, 1160.0, 2170.0, 2170.0])
    np.testing.assert_allclose(mm.command_to_omega(-1.0 * np.ones(4)), 150.0)


def test_thrust_to_command_is_inverse() -> None:
    mm = MotorModel(_params())
    u = np.array([0.0, 0.1, 0.4, 1.0])
    thrust = 1.91e-6 * mm.command_to_omega(u) ** 2
    np.testing.assert_allclose(mm.thrust_to_command(thrust), u, atol=1e-12)
    # out of range clips
    np.testing.assert_allclose(mm.thrust_to_command(np.array([0.0, 1e3, -1.0, 0.0])),
                               [0.0, 1.0, 0.0, 0.0])  # fmt: skip


def test_hover_command_gives_weight() -> None:
    p = _params()
    mass = 0.5143
    u = hover_command(p, mass)
    mm = MotorModel(p)
    omega = mm.command_to_omega(u)
    np.testing.assert_allclose(4 * 1.91e-6 * omega**2, mass * 9.81, rtol=1e-12)
    assert hover_thrust_per_motor(mass) == pytest.approx(mass * 9.81 / 4)
    assert np.all((u > 0.0) & (u < 1.0))


def test_batched_over_envs() -> None:
    p = _params(3)
    p.tau_m_s[1] = 0.1  # env 1 slower
    mm = MotorModel(p)
    mm.reset(np.full((3, 4), 300.0))
    t = mm.step(np.full((3, 4), 1500.0), 0.01)
    assert t.shape == (3, 4)
    assert np.all(mm.omega[1] < mm.omega[0])
    np.testing.assert_allclose(mm.omega[0], mm.omega[2])
    assert hover_command(p, 0.5).shape == (3, 4)


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="omega_min"):
        MotorParams(k_f=1e-6, k_m=1e-8, tau_m_s=0.03, omega_min=100.0, omega_max=50.0)
    with pytest.raises(ValueError, match="positive"):
        MotorParams(k_f=1e-6, k_m=1e-8, tau_m_s=0.0, omega_min=0.0, omega_max=50.0)


def test_optional_effects() -> None:
    f = rotor_drag_force_b(np.full(4, 1000.0), np.array([2.0, -1.0, 3.0]), 2e-5)
    np.testing.assert_allclose(f, [-0.16, 0.08, 0.0])
    assert ground_effect_gain(10.0, 0.065) == pytest.approx(1.0, abs=1e-3)
    g = ground_effect_gain(np.array([0.2, 0.1, 0.01]), 0.065, 1.5)
    assert np.all(np.diff(g) >= 0.0)
    assert g[-1] == pytest.approx(1.5)


def test_battery_scale_validated_after_scaling() -> None:
    params = _params()
    assert params.with_battery_scale(0.9).omega_max == pytest.approx(0.9 * params.omega_max)
    with pytest.raises(ValueError, match="omega_max"):
        params.with_battery_scale(100.0 / 2170.0)  # omega_max = 100 < omega_min
