"""ActionDelay, DisturbanceModel and ImuNoiseModel."""

from __future__ import annotations

import numpy as np
import pytest

from dronez.sim import (
    ActionDelay,
    DisturbanceConfig,
    DisturbanceModel,
    ImpulseSpec,
    ImuNoiseConfig,
    ImuNoiseModel,
    TrueState,
)
from dronez.utils import rotations

pytestmark = pytest.mark.fast


def _true(accel: float = 9.81) -> TrueState:
    return TrueState(
        pos_w=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linvel_w=np.zeros(3),
        angvel_b=np.array([0.1, -0.2, 0.3]),
        omega=np.full(4, 800.0),
        thrust=np.full(4, 1.2),
        accel_b=np.array([0.0, 0.0, accel]),
    )


@pytest.mark.parametrize("n", [0, 1, 3, 10])
def test_action_delay_delays_by_n_pushes(n: int) -> None:
    ad = ActionDelay(n)
    seq = [np.full(4, float(k)) for k in range(1, 30)]
    out = [ad.push(u) for u in seq]
    for k, o in enumerate(out):
        expected = seq[max(k - n, 0)]  # hold-first until the line has filled
        np.testing.assert_array_equal(o, expected)


def test_action_delay_copies_and_resets() -> None:
    ad = ActionDelay(2)
    u = np.array([1.0, 2.0, 3.0, 4.0])
    ad.push(u)
    u[:] = 99.0  # mutating the caller's array must not change the stored command
    ad.push(np.zeros(4))
    np.testing.assert_array_equal(ad.push(np.zeros(4)), [1.0, 2.0, 3.0, 4.0])
    ad.reset(np.full(4, 0.5))
    np.testing.assert_array_equal(ad.push(np.ones(4)), np.full(4, 0.5))
    with pytest.raises(ValueError, match="n_steps"):
        ActionDelay(-1)


def test_scripted_impulse_window() -> None:
    cfg = DisturbanceConfig(
        impulses=(
            ImpulseSpec(t_s=0.5, duration_s=0.1, force_w_n=(1, 2, 3), torque_w_nm=(0, 0, 0.1)),
        )
    )
    dm = DisturbanceModel()
    dm.reset(np.random.default_rng(0), cfg)
    f, tq = dm.wrench(0.49, _true())
    assert not f.any()
    assert not tq.any()
    f, tq = dm.wrench(0.55, _true())
    np.testing.assert_allclose(f, [1, 2, 3])
    np.testing.assert_allclose(tq, [0, 0, 0.1])
    f, _ = dm.wrench(0.6, _true())
    assert not f.any()


def test_default_config_is_no_disturbance() -> None:
    dm = DisturbanceModel()
    dm.reset(np.random.default_rng(1))
    f, tq = dm.wrench(1.0, _true())
    assert not f.any()
    assert not tq.any()


def test_wind_and_gusts_deterministic_per_seed() -> None:
    cfg = DisturbanceConfig(
        wind_mean_w_m_s=(3.0, 0.0, 0.0), wind_gust_std_m_s=1.0, random_impulse_count=3,
        random_impulse_force_n=0.5,
    )  # fmt: skip
    a, b, c = DisturbanceModel(), DisturbanceModel(), DisturbanceModel()
    a.reset(np.random.default_rng(7), cfg)
    b.reset(np.random.default_rng(7), cfg)
    c.reset(np.random.default_rng(8), cfg)
    ts = np.linspace(0.0, 10.0, 200)
    fa = np.array([a.wrench(t, _true())[0] for t in ts])
    fb = np.array([b.wrench(t, _true())[0] for t in ts])
    fc = np.array([c.wrench(t, _true())[0] for t in ts])
    np.testing.assert_array_equal(fa, fb)
    assert not np.allclose(fa, fc)
    # a +x wind pushes the drone toward +x
    assert fa[:, 0].mean() > 0.0
    # gust field has roughly the requested RMS
    wind = np.array([a.wind_w(t) for t in np.linspace(0, 200, 4000)])
    assert wind[:, 1].std() == pytest.approx(1.0, rel=0.3)


def test_imu_noise_statistics_and_determinism() -> None:
    cfg = ImuNoiseConfig(gyro_bias_init_std=0.0, accel_bias_init_std=0.0, gyro_bias_rw=0.0,
                         accel_bias_rw=0.0)  # fmt: skip
    m = ImuNoiseModel(cfg)
    rng = np.random.default_rng(3)
    m.reset(rng)
    dt = 0.01
    g = np.array([m.apply(_true(), dt, rng).gyro_b for _ in range(20000)])
    assert g.mean(axis=0) == pytest.approx([0.1, -0.2, 0.3], abs=3e-4)
    assert g.std(axis=0) == pytest.approx(cfg.gyro_noise_density / np.sqrt(dt), rel=0.05)

    def run(seed: int) -> np.ndarray:
        mm = ImuNoiseModel()
        r = np.random.default_rng(seed)
        mm.reset(r)
        return np.array([mm.apply(_true(), dt, r).accel_b for _ in range(50)])

    np.testing.assert_array_equal(run(5), run(5))
    assert not np.allclose(run(5), run(6))


def test_imu_ideal_is_passthrough_and_bias_walks() -> None:
    m = ImuNoiseModel(ImuNoiseConfig.ideal())
    rng = np.random.default_rng(0)
    m.reset(rng)
    s = m.apply(_true(), 0.01, rng)
    np.testing.assert_allclose(s.gyro_b, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(s.accel_b, [0.0, 0.0, 9.81])
    np.testing.assert_allclose(s.quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    # random walk: bias std grows like rw * sqrt(t)
    m2 = ImuNoiseModel(
        ImuNoiseConfig(gyro_noise_density=0.0, gyro_bias_init_std=0.0, gyro_bias_rw=1e-2,
                       accel_noise_density=0.0, accel_bias_init_std=0.0, accel_bias_rw=0.0,
                       quat_noise_rad=0.0)
    )  # fmt: skip
    finals = []
    for seed in range(300):
        r = np.random.default_rng(seed)
        m2.reset(r)
        for _ in range(100):
            s = m2.apply(_true(), 0.01, r)
        finals.append(s.gyro_b[0] - 0.1)
    assert np.std(finals) == pytest.approx(1e-2 * np.sqrt(1.0), rel=0.15)


def test_attitude_noise_is_small_rotation() -> None:
    m = ImuNoiseModel(ImuNoiseConfig(quat_noise_rad=0.01))
    rng = np.random.default_rng(0)
    m.reset(rng)
    angs = [
        float(rotations.geodesic_angle(m.apply(_true(), 0.01, rng).quat_wxyz, [1, 0, 0, 0]))
        for _ in range(2000)
    ]
    assert np.sqrt(np.mean(np.square(angs))) == pytest.approx(0.01 * np.sqrt(3.0), rel=0.1)
