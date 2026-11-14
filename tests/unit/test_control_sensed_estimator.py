"""Sensed gained an optional position/velocity estimate; the IMU channels must not change."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.sim import ImuNoiseConfig, ImuNoiseModel, Sensed, TrueState

pytestmark = pytest.mark.fast


def _true() -> TrueState:
    return TrueState(
        pos_w=np.array([1.0, -2.0, 3.0]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linvel_w=np.array([0.5, 0.0, -0.25]),
        angvel_b=np.array([0.1, -0.2, 0.3]),
        omega=np.full(4, 800.0),
        thrust=np.full(4, 1.2),
        accel_b=np.array([0.0, 0.0, 9.81]),
    )


def test_sensed_estimator_fields_default_to_none() -> None:
    s = Sensed(np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    assert s.pos_w is None
    assert s.vel_w is None


def test_ideal_noise_gives_exact_pos_vel() -> None:
    m = ImuNoiseModel(ImuNoiseConfig.ideal())
    rng = np.random.default_rng(0)
    m.reset(rng)
    s = m.apply(_true(), 0.01, rng)
    np.testing.assert_array_equal(s.pos_w, _true().pos_w)
    np.testing.assert_array_equal(s.vel_w, _true().linvel_w)


def test_pos_vel_noise_std_and_unchanged_imu_stream() -> None:
    cfg = ImuNoiseConfig(pos_noise_std=0.01, vel_noise_std=0.05)
    m = ImuNoiseModel(cfg)
    rng = np.random.default_rng(3)
    m.reset(rng)
    samples = [m.apply(_true(), 0.01, rng) for _ in range(4000)]
    dp = np.array([s.pos_w - _true().pos_w for s in samples])
    dv = np.array([s.vel_w - _true().linvel_w for s in samples])
    assert dp.std() == pytest.approx(0.01, rel=0.05)
    assert abs(dp.mean()) < 1e-3
    assert dv.std() == pytest.approx(0.05, rel=0.05)
    # the IMU channels of a seed do not depend on the pos/vel noise settings (drawn afterwards)
    ref = ImuNoiseModel(ImuNoiseConfig(pos_noise_std=0.0, vel_noise_std=0.0))
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    ref.reset(rng_a)
    m2 = ImuNoiseModel(cfg)
    m2.reset(rng_b)
    a, b = ref.apply(_true(), 0.01, rng_a), m2.apply(_true(), 0.01, rng_b)
    np.testing.assert_array_equal(a.gyro_b, b.gyro_b)
    np.testing.assert_array_equal(a.quat_wxyz, b.quat_wxyz)
