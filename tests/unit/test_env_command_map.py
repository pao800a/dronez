"""T5 (ADR-0006 §3): the affine action -> command map, baseline action mapping, mixer limits."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.control import CascadedPID, CommandMap, Mixer, NominalVehicle
from dronez.control.base import thrusts_to_action
from dronez.sim import MotorModel, hover_command

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def vehicle(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> NominalVehicle:
    return NominalVehicle.from_drone(sim_cfg_drone, sim_art)


@pytest.fixture(scope="module")
def cmap(vehicle: NominalVehicle) -> CommandMap:
    return CommandMap(vehicle.motor, vehicle.mass_kg)


def _u_h(vehicle: NominalVehicle) -> float:
    return float(np.mean(hover_command(vehicle.motor, vehicle.mass_kg)))


def test_zero_action_is_exactly_the_hover_command(
    cmap: CommandMap, vehicle: NominalVehicle
) -> None:
    u_h = _u_h(vehicle)
    u = cmap.action_to_command(np.zeros(4))
    np.testing.assert_allclose(u, u_h, rtol=0, atol=1e-15)
    assert 0.25 < u_h < 0.32  # the ADR's 0.2864 (estimated propulsion constants)


@pytest.mark.parametrize("authority", [1.0, 0.5])
def test_extremes_are_hover_plus_minus_delta_u(vehicle: NominalVehicle, authority: float) -> None:
    cm = CommandMap(vehicle.motor, vehicle.mass_kg, authority=authority)
    u_h = _u_h(vehicle)
    delta_u = authority * min(u_h, 1.0 - u_h)
    np.testing.assert_allclose(cm.action_to_command(np.ones(4)), u_h + delta_u, atol=1e-15)
    np.testing.assert_allclose(cm.action_to_command(-np.ones(4)), u_h - delta_u, atol=1e-15)
    # the symmetric envelope: u in [0, 2 u_h] at authority 1 (u_h < 0.5), never the full range
    assert u_h + delta_u <= 2.0 * u_h + 1e-12
    assert u_h + delta_u < 1.0


def test_out_of_range_actions_are_clipped(cmap: CommandMap) -> None:
    np.testing.assert_array_equal(
        cmap.action_to_command(np.array([3.0, -7.0, 1.0, -1.0])),
        cmap.action_to_command(np.array([1.0, -1.0, 1.0, -1.0])),
    )


def test_command_to_action_inverts_action_to_command(cmap: CommandMap) -> None:
    a = np.random.default_rng(0).uniform(-1.0, 1.0, size=(1000, 4))
    np.testing.assert_allclose(cmap.command_to_action(cmap.action_to_command(a)), a, atol=1e-12)
    # and the other way, on the commands the envelope can produce
    u = cmap.action_to_command(a)
    np.testing.assert_allclose(cmap.action_to_command(cmap.command_to_action(u)), u, atol=1e-12)
    # commands outside the envelope saturate the action instead of leaving [-1, 1]
    far = cmap.command_to_action(np.array([0.0, 1.0, 0.9, 0.99]))
    assert np.all(np.abs(far) <= 1.0)


def test_thrust_limits_are_the_thrust_at_the_action_extremes(
    cmap: CommandMap, vehicle: NominalVehicle
) -> None:
    motor = MotorModel(vehicle.motor)
    lo, hi = cmap.thrust_limits(vehicle.motor)
    for t_lim, a in ((lo, -1.0), (hi, 1.0)):
        omega = motor.command_to_omega(cmap.action_to_command(np.full(4, a)))
        np.testing.assert_allclose(t_lim, vehicle.motor.k_f * omega**2, rtol=1e-12)
    assert np.all(lo < hi)
    # ADR §3: about 3.88 N at a = +1 (estimate). The ADR's 0.34 N at a = -1 is inconsistent with
    # its own formula (u = 0 gives omega_min = 150 rad/s, 0.033 N), so only the bound is checked.
    assert hi[0] == pytest.approx(3.88, rel=0.1)
    assert lo[0] < 0.1


def test_thrusts_to_action_stays_in_the_action_box(
    cmap: CommandMap, vehicle: NominalVehicle
) -> None:
    motor = MotorModel(vehicle.motor)
    lo, hi = cmap.thrust_limits(vehicle.motor)
    rng = np.random.default_rng(3)
    # inside the envelope, far above it and far below it
    for scale in (1.0, 5.0, 0.0):
        t = rng.uniform(lo, hi, size=(200, 4)) * scale
        a = np.array([thrusts_to_action(row, motor, cmap) for row in t])
        assert np.all(np.abs(a) <= 1.0)
    # a thrust inside the envelope maps back exactly (the inverse of action -> thrust)
    hover_thrust = 0.5 * (lo[0] + hi[0])
    t_in = np.full(4, hover_thrust)
    a_in = thrusts_to_action(t_in, motor, cmap)
    u = cmap.action_to_command(a_in)
    omega = motor.command_to_omega(u)
    np.testing.assert_allclose(vehicle.motor.k_f * omega**2, hover_thrust, rtol=1e-5)


def test_baseline_mixer_limits_are_the_envelope(vehicle: NominalVehicle, cmap: CommandMap) -> None:
    lo, hi = cmap.thrust_limits(vehicle.motor)
    mixer = Mixer.from_vehicle(vehicle, cmap)
    np.testing.assert_allclose(mixer.t_min, lo, rtol=1e-12)
    np.testing.assert_allclose(mixer.t_max, hi, rtol=1e-12)
    # and the default (no cmap) is that same nominal envelope, not the raw omega limits
    default = Mixer.from_vehicle(vehicle)
    np.testing.assert_allclose(default.t_max, hi, rtol=1e-12)
    assert np.all(hi < vehicle.thrust_limits_n[1])
    # a baseline controller exposes exactly that mixer and a map equal to the nominal one
    pid = CascadedPID(vehicle)
    np.testing.assert_allclose(pid.mixer.t_max, hi, rtol=1e-12)
    np.testing.assert_allclose(pid.cmap.u_hover, cmap.u_hover, atol=1e-15)


def test_meta_entry_matches_the_contract(cmap: CommandMap) -> None:
    m = cmap.meta()
    assert m["type"] == "affine_omega"
    assert set(m) >= {"u_hover", "delta_u", "authority"}
