"""Pure eval metrics on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.eval import metrics as M
from dronez.utils import rotations

pytestmark = pytest.mark.fast

DT = 0.01


def test_attitude_error_is_the_geodesic_angle() -> None:
    q = rotations.quat_from_axis_angle([0, 1, 0], np.deg2rad(30.0))
    assert M.attitude_error(q, rotations.identity_quat()) == pytest.approx(np.deg2rad(30.0))
    assert M.attitude_error(-q, rotations.identity_quat()) == pytest.approx(np.deg2rad(30.0))


def test_time_to_recover_exponential_decay() -> None:
    tau = 0.2
    t = np.arange(0, 3.0, DT)
    err = np.deg2rad(30.0) * np.exp(-t / tau)
    expected = tau * np.log(30.0 / 5.0)
    assert M.time_to_recover(err, DT) == pytest.approx(expected, abs=2 * DT)


def test_time_to_recover_needs_to_stay_below_and_nan_if_never() -> None:
    thr = M.RECOVERY_THRESHOLD_RAD
    err = np.full(300, 0.5 * thr)
    err[100] = 2.0 * thr  # late blip: recovery only counts after it
    assert M.time_to_recover(err, DT) == pytest.approx(1.01, abs=1e-9)
    assert np.isnan(M.time_to_recover(np.full(100, 2 * thr), DT))
    err2 = np.concatenate([np.full(50, 2 * thr), np.full(150, 0.1 * thr)])
    assert M.time_to_recover(err2, DT, t_start=0.5) == pytest.approx(0.0)  # measured from t_start


def test_rms_attitude_error() -> None:
    err = np.concatenate([np.full(100, 0.2), np.full(100, 0.1)])
    assert M.rms_attitude_error(err, DT) == pytest.approx(np.sqrt((0.04 + 0.01) / 2))
    assert M.rms_attitude_error(err, DT, t_start=1.0) == pytest.approx(0.1)


def test_overshoot_about_a_fixed_axis() -> None:
    q0 = rotations.identity_quat()
    target = rotations.quat_from_axis_angle([1, 0, 0], np.deg2rad(20.0))
    angles = np.deg2rad(np.array([0.0, 10.0, 20.0, 23.0, 21.0, 20.0]))
    path = rotations.quat_from_axis_angle(np.tile([1.0, 0, 0], (angles.size, 1)), angles)
    assert M.overshoot(path, q0, target) == pytest.approx(np.deg2rad(3.0))
    assert M.overshoot(path[:3], q0, target) == pytest.approx(0.0)


def test_saturation_pct() -> None:
    u = np.full((10, 4), 0.5)
    u[0, :] = 1.0
    u[1, 0] = 0.0
    assert M.saturation_pct(u) == pytest.approx(100.0 * 5 / 40)
    assert M.saturation_pct(np.zeros((0, 4))) == 0.0


def test_mean_abs_delta() -> None:
    u = np.array([[0.0, 0.0], [0.1, 0.3], [0.1, 0.1]])
    assert M.mean_abs_delta(u) == pytest.approx((0.1 + 0.3 + 0.0 + 0.2) / 4)
    assert M.mean_abs_delta(u[:1]) == 0.0


def test_hf_energy_fraction_separates_bands() -> None:
    n, rate = 400, 100.0
    t = np.arange(n) / rate
    slow = np.stack([np.sin(2 * np.pi * 5.0 * t)] * 4, axis=1)  # 5 Hz (bin-exact: 20 cycles)
    fast = np.stack([np.sin(2 * np.pi * 40.0 * t)] * 4, axis=1)
    assert M.hf_energy_fraction(slow, rate) < 1e-6
    assert M.hf_energy_fraction(fast, rate) > 1 - 1e-6
    assert M.hf_energy_fraction(slow + 0.5 * fast, rate) == pytest.approx(0.25 / 1.25, abs=1e-6)
    assert M.hf_energy_fraction(np.full((n, 4), 0.3), rate) == 0.0  # DC only


def test_success_window_and_crash() -> None:
    thr = M.RECOVERY_THRESHOLD_RAD
    err = np.concatenate([np.full(100, 3 * thr), np.full(150, 0.5 * thr)])
    assert M.success(err, DT, hold_s=1.0)
    assert not M.success(err, DT, hold_s=2.0)
    assert not M.success(err, DT, crashed=True)


def test_position_error() -> None:
    p = np.array([[0.0, 0.0, 1.0], [0.03, 0.04, 1.0]])
    np.testing.assert_allclose(M.position_error(p, [0.0, 0.0, 1.0]), [0.0, 0.05])
