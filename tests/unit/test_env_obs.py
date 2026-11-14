"""T3 (ADR-0006 §2): observation content with sensor noise off."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dronez.utils import rotations as rot

pytestmark = pytest.mark.fast

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"
R6, GRAV, GYRO, VZ, HIST = slice(0, 6), slice(6, 9), slice(9, 12), 12, slice(13, 29)
POS_H, VXY_H = slice(29, 32), slice(32, 34)
TOL = 1e-6  # float32 observations


def _roll(angle: float) -> np.ndarray:
    return rot.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), angle)


def _reset(env: Any, api: ModuleType, **kw: Any) -> tuple[np.ndarray, dict[str, Any]]:
    obs, info = env.reset(seed=1, options=api.options(**kw))
    return np.asarray(obs, dtype=np.float64), info


def test_target_reached_gives_identity_error_and_gravity_down(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    q = rot.quat_from_yaw(0.7)
    obs, info = _reset(env, api, quat=q, q_target=q)
    # the harness must have honoured the target we asked for (ADR §9 target_schedule)
    np.testing.assert_allclose(np.abs(info["target"]["quat_wxyz"] @ q), 1.0, atol=1e-9)
    np.testing.assert_allclose(obs[R6], [1, 0, 0, 0, 1, 0], atol=TOL)
    np.testing.assert_allclose(obs[GRAV], [0, 0, -1], atol=TOL)
    np.testing.assert_allclose(obs[GYRO], 0.0, atol=TOL)
    assert obs[VZ] == pytest.approx(0.0, abs=TOL)
    np.testing.assert_array_equal(obs[HIST], 0.0)  # prefilled with a0 = 0


def test_known_90_degree_roll(api: ModuleType, mk: Any) -> None:
    """q = Rx(+90 deg), target identity: R_err = Rx(90) = [[1,0,0],[0,0,-1],[0,1,0]].

    Columns 0 and 1, column-major: [R00, R10, R20, R01, R11, R21] = [1, 0, 0, 0, 0, 1].
    Gravity in the body frame: body +y points up after a +x roll, so g_b = (0, -1, 0).
    """
    env = mk(ATT)
    obs, _ = _reset(env, api, quat=_roll(np.pi / 2), q_target=rot.identity_quat())
    np.testing.assert_allclose(obs[R6], [1, 0, 0, 0, 0, 1], atol=TOL)
    np.testing.assert_allclose(obs[GRAV], [0, -1, 0], atol=TOL)


def test_inverted_is_a_valid_observation(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    obs, _ = _reset(env, api, quat=_roll(np.pi), q_target=rot.identity_quat())
    np.testing.assert_allclose(obs[R6], [1, 0, 0, 0, -1, 0], atol=TOL)  # Rx(180) = diag(1, -1, -1)
    np.testing.assert_allclose(obs[GRAV], [0, 0, 1], atol=TOL)


def test_gyro_vz_and_history_come_from_the_right_sources(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    a0 = np.array([0.1, -0.2, 0.15, 0.05])
    obs, _ = _reset(
        env, api, omega_b=(0.3, -0.2, 0.1), vel=(0.5, 0.0, 0.8), a0=a0, q_target=rot.identity_quat()
    )
    np.testing.assert_allclose(obs[GYRO], [0.3, -0.2, 0.1], atol=TOL)
    # vel_z_w is the world-z velocity of the CoM (the body origin differs by omega x r, ~mm/s)
    assert obs[VZ] == pytest.approx(0.8, abs=0.02)
    np.testing.assert_allclose(obs[HIST].reshape(4, 4), np.tile(a0, (4, 1)), atol=TOL)


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_observation_is_heading_invariant(api: ModuleType, mk: Any, env_id: str) -> None:
    """Rotating q, q_t, p, p_t and v by a world yaw leaves the whole observation unchanged."""
    env = mk(env_id)
    rng = np.random.default_rng(11)
    for trial in range(5):
        q = rot.random_quat(rng)
        psi_t = rng.uniform(-np.pi, np.pi)
        q_t = rot.quat_from_yaw(psi_t) if env_id == HOV else rot.random_tilted_quat(rng, 0.35)
        p = np.array([0.3, -0.2, 3.1])
        p_t = np.array([0.0, 0.0, 3.0])
        v = np.array([0.4, -0.3, 0.2])
        w_b = np.array([0.2, 0.1, -0.3])
        a0 = rng.uniform(-0.2, 0.2, 4)
        kw = {"omega_b": w_b, "a0": a0}
        base, _ = _reset(env, api, pos=p, quat=q, vel=v, q_target=q_t, p_target=p_t, **kw)
        psi = rng.uniform(-np.pi, np.pi)
        q_z = rot.quat_from_yaw(psi)
        rotated, _ = _reset(
            env,
            api,
            pos=rot.quat_rotate(q_z, p),
            quat=rot.quat_mul(q_z, q),
            vel=rot.quat_rotate(q_z, v),
            q_target=rot.quat_mul(q_z, q_t),
            p_target=rot.quat_rotate(q_z, p_t),
            **kw,
        )
        np.testing.assert_allclose(rotated, base, atol=2e-5, err_msg=f"trial {trial}, yaw {psi}")


def test_hover_error_blocks_are_in_the_target_heading_frame(api: ModuleType, mk: Any) -> None:
    """psi_t = 90 deg, p - p_t = (1, 0, 0) world: pos_err_h = Rz(-90) (1, 0, 0) = (0, -1, 0)."""
    env = mk(HOV)
    q_t = rot.quat_from_yaw(np.pi / 2)
    p_t = np.array([0.0, 0.0, 2.0])
    start = p_t + np.array([1.0, 0.0, 0.0])
    obs, info = _reset(env, api, pos=start, quat=q_t, q_target=q_t, p_target=p_t)
    np.testing.assert_allclose(info["target"]["pos_w"], p_t, atol=1e-9)
    np.testing.assert_allclose(obs[POS_H], [0.0, -1.0, 0.0], atol=2e-3)  # CoM vs body origin ~mm
    # a world velocity of (0, 0.5, 0) is +0.5 m/s along the target heading's x axis
    obs, _ = _reset(env, api, pos=p_t, quat=q_t, vel=(0.0, 0.5, 0.0), q_target=q_t, p_target=p_t)
    np.testing.assert_allclose(obs[VXY_H], [0.5, 0.0], atol=2e-3)
    assert obs[POS_H][2] == pytest.approx(0.0, abs=2e-3)


def test_observation_never_contains_true_state(api: ModuleType, mk: Any) -> None:
    """Sensor noise on: the obs gyro is the *sensed* gyro, not the true angular velocity."""
    env = mk(ATT)
    kw = {"omega_b": (0.3, -0.2, 0.1), "q_target": rot.identity_quat(), "noise": None}
    obs, info = _reset(env, api, **kw)
    true_w = np.asarray(info["true"]["angvel_b"], dtype=np.float64)
    np.testing.assert_allclose(true_w, [0.3, -0.2, 0.1], atol=1e-9)
    assert not np.allclose(obs[GYRO], true_w, atol=1e-9), "obs gyro equals the truth: no noise?"
    np.testing.assert_allclose(obs[GYRO], info["sensed"]["gyro_b"], atol=TOL)
