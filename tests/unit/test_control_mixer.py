"""Mixer: matrix derived from the config, sign pattern, inversion, saturation priority."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.cad.artifacts import ModelArtifacts
from dronez.cad.schema import DroneConfig
from dronez.control import Mixer, NominalVehicle
from dronez.sim import DroneSim, SimConfig, build_model_from_artifacts, motor_gear

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def vehicle(sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts) -> NominalVehicle:
    return NominalVehicle.from_drone(sim_cfg_drone, sim_art)


@pytest.fixture(scope="module")
def mixer(vehicle: NominalVehicle) -> Mixer:
    return Mixer.from_vehicle(vehicle)


def test_allocation_matrix_signs_match_conventions_table(
    mixer: Mixer, vehicle: NominalVehicle
) -> None:
    a = mixer.A
    np.testing.assert_array_equal(a[0], 1.0)
    # Increase right motors (M0 FR, M3 RR; y < 0): tau_x < 0. Left motors (M1, M2): tau_x > 0.
    assert a[1, 0] < 0
    assert a[1, 3] < 0
    assert a[1, 1] > 0
    assert a[1, 2] > 0
    # Increase front motors (M0, M2; x > 0): tau_y < 0 (nose up). Rear motors (M1, M3): tau_y > 0.
    assert a[2, 0] < 0
    assert a[2, 2] < 0
    assert a[2, 1] > 0
    assert a[2, 3] > 0
    # Increase the CW pair (M2, M3): tau_z > 0 (yaw left). CCW pair (M0, M1): tau_z < 0.
    assert a[3, 2] > 0
    assert a[3, 3] > 0
    assert a[3, 0] < 0
    assert a[3, 1] < 0
    # magnitudes come from the config: |x|, |y| ~ a = 0.0884 m, yaw ratio k_m/k_f = 0.016 m
    np.testing.assert_allclose(np.abs(a[1:3]), 0.0884, atol=2e-4)
    np.testing.assert_allclose(np.abs(a[3]), vehicle.motor.k_m_over_k_f, rtol=1e-12)


def test_yaw_row_equals_the_mujoco_actuator_gear(
    mixer: Mixer, sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    """The mixer's yaw row and the model's site-actuator gear come from the same derivation."""
    model, index = build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig())
    np.testing.assert_allclose(mixer.A[3], model.actuator_gear[:, 5], rtol=1e-12)
    np.testing.assert_allclose(
        model.actuator_gear,
        motor_gear(index.spin, sim_cfg_drone.propulsion.k_m / sim_cfg_drone.propulsion.k_f),
    )


def test_uses_com_relative_hubs_and_reproduces_sim_trim(
    mixer: Mixer, vehicle: NominalVehicle, sim_cfg_drone: DroneConfig, sim_art: ModelArtifacts
) -> None:
    model, index = build_model_from_artifacts(sim_cfg_drone, sim_art, SimConfig())
    sim = DroneSim(sim_cfg_drone, SimConfig(), model=model, index=index)
    t = mixer.allocate(vehicle.weight_n, np.zeros(3))
    np.testing.assert_allclose(t, sim.trim_thrusts(), atol=1e-12)
    np.testing.assert_allclose(mixer.wrench(t), [vehicle.weight_n, 0, 0, 0], atol=1e-12)
    # not the equal split: the hubs are measured from the CoM (0.06 mm off the hub centroid)
    assert np.ptp(t) > 1e-6
    np.testing.assert_allclose(vehicle.hubs_b, index.motor_pos_b, atol=1e-12)


def test_allocate_then_wrench_is_identity_when_unsaturated(mixer: Mixer) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        t = rng.uniform(mixer.t_min + 0.2, mixer.t_max - 0.2)
        w = mixer.wrench(t)
        np.testing.assert_allclose(mixer.allocate(w[0], w[1:]), t, atol=1e-9)
        np.testing.assert_allclose(mixer.wrench(mixer.allocate(w[0], w[1:])), w, atol=1e-9)


def _hover(vehicle: NominalVehicle) -> float:
    return vehicle.weight_n


def test_yaw_is_dropped_first_roll_pitch_and_thrust_kept(
    mixer: Mixer, vehicle: NominalVehicle
) -> None:
    tau = np.array([0.05, -0.03, 5.0])  # absurd yaw demand
    t, info = mixer.allocate_info(_hover(vehicle), tau)
    assert info.saturated
    assert info.yaw_scale < 1.0
    assert info.rp_scale == 1.0
    w = mixer.wrench(t)
    np.testing.assert_allclose(w[:3], [_hover(vehicle), 0.05, -0.03], atol=1e-9)
    assert np.all(t >= mixer.t_min - 1e-12)
    assert np.all(t <= mixer.t_max + 1e-12)
    assert 0.0 <= w[3] < 5.0
    assert w[3] == pytest.approx(info.yaw_scale * 5.0, rel=1e-6)


def test_collective_is_sacrificed_before_roll_pitch(mixer: Mixer) -> None:
    """Near full thrust, a roll demand is kept exactly; the total thrust is what gives."""
    t_req = 4.0 * 0.97 * float(mixer.t_max[0])
    tau = np.array([0.3, 0.0, 0.0])
    t, info = mixer.allocate_info(t_req, tau)
    w = mixer.wrench(t)
    assert info.saturated
    assert info.rp_scale == 1.0
    assert w[1] == pytest.approx(0.3, abs=1e-9)
    assert w[2] == pytest.approx(0.0, abs=1e-9)
    assert w[0] < t_req
    assert info.collective_shift_n == pytest.approx(w[0] - t_req, abs=1e-9)
    assert np.all(t <= mixer.t_max + 1e-12)


def test_collective_shift_works_at_the_low_end_too(mixer: Mixer) -> None:
    t, info = mixer.allocate_info(0.2, np.array([0.05, 0.0, 0.0]))
    w = mixer.wrench(t)
    assert w[1] == pytest.approx(0.05, abs=1e-9)
    assert w[0] > 0.2
    assert info.collective_shift_n > 0
    assert np.all(t >= mixer.t_min - 1e-12)


def test_infeasible_roll_pitch_is_scaled_keeping_direction(
    mixer: Mixer, vehicle: NominalVehicle
) -> None:
    tau = np.array([40.0, -20.0, 0.0])
    t, info = mixer.allocate_info(_hover(vehicle), tau)
    w = mixer.wrench(t)
    assert info.rp_scale < 1.0
    np.testing.assert_allclose(w[1:3] / info.rp_scale, tau[:2], rtol=1e-6)
    assert np.all(t >= mixer.t_min - 1e-9)
    assert np.all(t <= mixer.t_max + 1e-9)


def test_random_requests_always_respect_limits_and_priority(mixer: Mixer) -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        t_req = rng.uniform(-2.0, 60.0)
        tau = rng.normal(scale=[0.4, 0.4, 0.15])
        t, info = mixer.allocate_info(t_req, tau)
        assert np.all(t >= mixer.t_min - 1e-9)
        assert np.all(t <= mixer.t_max + 1e-9)
        w = mixer.wrench(t)
        # achieved roll/pitch is exactly rp_scale * request (priority over thrust and yaw)
        np.testing.assert_allclose(w[1:3], info.rp_scale * tau[:2], atol=1e-7)
        if info.rp_scale == 1.0 and info.yaw_scale == 1.0 and not info.saturated:
            np.testing.assert_allclose(w, [t_req, *tau], atol=1e-7)
