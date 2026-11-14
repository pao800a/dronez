"""CascadedPID, RateController, GeometricSO3 on the CAD-free synthetic model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.control import (
    CascadedPID,
    Controller,
    GeometricSO3,
    NominalVehicle,
    RateController,
    Setpoint,
    load_geometric_config,
    load_pid_config,
)
from dronez.control.geometric import hat, vee
from dronez.control.pid import RateGains
from dronez.eval import metrics as M
from dronez.eval.runner import run_episode
from dronez.eval.scenarios import Scenario, make_plan
from dronez.sim import ImuNoiseConfig
from dronez.utils import rotations

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def vehicle(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> NominalVehicle:
    return NominalVehicle.from_drone(sim_cfg_drone, sim_art)


@pytest.fixture
def env(synthetic_env: Callable[..., Any]) -> Any:
    """``DronezAttitude-v0`` on the synthetic model, nominal plant, default sensor noise."""
    return synthetic_env()


def _make(kind: str, vehicle: NominalVehicle) -> CascadedPID | GeometricSO3:
    return CascadedPID(vehicle) if kind == "pid" else GeometricSO3(vehicle)


RECOVERY = Scenario(name="r", kind="recovery", n_episodes=1, duration_s=2.0, start_height_m=1.5,
                    tilt_deg=30.0)  # fmt: skip
HOVER = Scenario(name="h", kind="hover_hold", n_episodes=1, duration_s=4.0, hold_s=1.0)


def test_gain_files_load() -> None:
    assert load_pid_config().rate.kp[0] > 0
    assert load_geometric_config().attitude.wn_rad_s[0] > 0


@pytest.mark.parametrize("kind", ["pid", "geometric"])
def test_protocol_and_action_range(kind: str, vehicle: NominalVehicle) -> None:
    ctrl = _make(kind, vehicle)
    assert isinstance(ctrl, Controller)
    assert ctrl.action_mode == "motors"
    from dronez.sim import Sensed

    s = Sensed(rotations.identity_quat(), np.zeros(3), np.array([0, 0, 9.81]))
    a = ctrl.act(np.zeros(0, np.float32), s)
    assert a.shape == (4,)
    assert a.dtype == np.float32
    assert np.all((a >= -1) & (a <= 1))  # ADR-0006 §3: the action space is [-1, 1]^4
    # level, at rest, no position setpoint: the action maps to the CoM-trimmed hover command
    expected = ctrl.motor.thrust_to_command(ctrl.mixer.allocate(vehicle.weight_n, np.zeros(3)))
    np.testing.assert_allclose(ctrl.cmap.action_to_command(a), expected, atol=1e-6)
    assert np.max(np.abs(a)) < 0.01  # the nominal hover is a ~ 0


@pytest.mark.parametrize("kind", ["pid", "geometric"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_from_30deg_within_1s_ideal_sensors(
    kind: str, seed: int, synthetic_env: Callable[..., Any], vehicle: NominalVehicle
) -> None:
    env = synthetic_env(noise=ImuNoiseConfig.ideal())
    traj = run_episode(env, _make(kind, vehicle), make_plan(RECOVERY, seed))
    err = M.attitude_error(traj.quat_wxyz, traj.quat_target)
    assert err[0] == pytest.approx(np.deg2rad(30.0), abs=1e-9)
    assert not traj.crashed
    assert M.time_to_recover(err, traj.dt) <= 1.0
    assert traj.pos_w[:, 2].min() > 1.3  # altitude held while recovering


@pytest.mark.parametrize("kind", ["pid", "geometric"])
def test_hover_hold_with_noise(kind: str, env: Any, vehicle: NominalVehicle) -> None:
    traj = run_episode(env, _make(kind, vehicle), make_plan(HOVER, 5))
    assert M.position_error(traj.pos_w, traj.pos_target).max() < 0.05


@pytest.mark.parametrize("kind", ["pid", "geometric"])
def test_runs_are_deterministic_per_seed(kind: str, env: Any, vehicle: NominalVehicle) -> None:
    a = run_episode(env, _make(kind, vehicle), make_plan(RECOVERY, 11))
    b = run_episode(env, _make(kind, vehicle), make_plan(RECOVERY, 11))
    np.testing.assert_array_equal(a.u, b.u)
    np.testing.assert_array_equal(a.quat_wxyz, b.quat_wxyz)


def test_reset_clears_integrators(vehicle: NominalVehicle) -> None:
    c = CascadedPID(vehicle)
    c.rate.update(np.array([1.0, 0, 0]), np.zeros(3))
    assert np.any(c.rate.integral != 0)
    c.reset(0)
    assert np.all(c.rate.integral == 0)


def test_rate_controller_antiwindup_and_d_on_measurement(vehicle: NominalVehicle) -> None:
    gains = RateGains(kp=(10, 10, 10), ki=(50, 50, 50), kd=(0.5, 0.5, 0.5), i_limit=(2, 2, 2))
    rc = RateController(vehicle.inertia_kg_m2, gains, 0.01)
    for _ in range(1000):  # persistent error: the integrator clamps
        rc.update(np.array([1.0, 0, 0]), np.zeros(3))
    np.testing.assert_allclose(rc.integral, [2.0, 0, 0])
    held = rc.integral.copy()
    rc.update(np.array([1.0, 0, 0]), np.zeros(3), freeze_integral=True)
    np.testing.assert_array_equal(rc.integral, held)
    # derivative acts on the measurement only: a setpoint jump with a constant gyro adds no D kick
    rc.reset()
    g = RateGains(kp=(0, 0, 0), ki=(0, 0, 0), kd=(1, 1, 1))
    rc2 = RateController(vehicle.inertia_kg_m2, g, 0.01)
    rc2.update(np.zeros(3), np.zeros(3))
    tau = rc2.update(np.array([5.0, 0, 0]), np.zeros(3))
    np.testing.assert_allclose(tau, 0.0, atol=1e-12)


def test_rate_controller_gyroscopic_feedforward(vehicle: NominalVehicle) -> None:
    rc = RateController(vehicle.inertia_kg_m2, RateGains(kp=(0, 0, 0), ki=(0, 0, 0)), 0.01)
    w = np.array([1.0, 2.0, 3.0])
    tau = rc.update(w, w)
    np.testing.assert_allclose(tau, np.cross(w, vehicle.inertia_kg_m2 @ w))


def test_pid_attitude_error_is_geodesic_and_shortest_path(vehicle: NominalVehicle) -> None:
    c = CascadedPID(vehicle)
    q_d = rotations.quat_from_axis_angle([0, 0, 1], 0.3)
    q = rotations.quat_mul(q_d, rotations.quat_from_axis_angle([1, 2, 3], 0.7))
    e = c.attitude_error_rotvec(q, q_d)
    assert np.linalg.norm(e) == pytest.approx(float(rotations.geodesic_angle(q, q_d)))
    np.testing.assert_allclose(c.attitude_error_rotvec(-q, q_d), e, atol=1e-12)  # q ~ -q
    # body-frame error of a rotation about body x: the axis is body x
    q2 = rotations.quat_mul(q_d, rotations.quat_from_axis_angle([1, 0, 0], 0.2))
    np.testing.assert_allclose(c.attitude_error_rotvec(q2, q_d), [0.2, 0, 0], atol=1e-12)


def test_geometric_error_terms(vehicle: NominalVehicle) -> None:
    c = GeometricSO3(vehicle)
    th = 0.4
    q = rotations.quat_from_axis_angle([0, 1, 0], th)
    e_r, e_w = c.attitude_errors(q, np.array([0.1, 0.2, 0.3]), np.eye(3))
    np.testing.assert_allclose(e_r, [0, np.sin(th), 0], atol=1e-12)  # e_R = sin(theta) * axis
    np.testing.assert_allclose(e_w, [0.1, 0.2, 0.3])
    v = np.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(vee(hat(v)), v)
    # gains are inertia-scaled from (wn, zeta)
    wn = np.array(c.cfg.attitude.wn_rad_s)
    np.testing.assert_allclose(c.k_r, vehicle.inertia_kg_m2 @ np.diag(wn**2))


@pytest.mark.parametrize("kind", ["pid", "geometric"])
def test_torque_sign_corrects_a_roll_error(kind: str, vehicle: NominalVehicle) -> None:
    """Rolled left (+x rotation) at rest -> commanded body torque about x is negative."""
    from dronez.sim import Sensed

    ctrl = _make(kind, vehicle)
    q = rotations.quat_from_axis_angle([1, 0, 0], 0.2)
    ctrl.set_setpoint(Setpoint(quat_wxyz=rotations.identity_quat()))
    s = Sensed(q, np.zeros(3), np.array([0, 0, 9.81]))
    r_d, _ = ctrl._desired(q, s)
    tau = ctrl._torque(q, np.zeros(3), r_d)
    assert tau[0] < 0
    assert abs(tau[1]) < 1e-9
