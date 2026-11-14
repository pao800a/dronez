"""T6 + the long-rollout halves of T4 / T7 (ADR-0006 §10): the envs against the real model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dronez.utils import rotations as rot
from env_helpers import ATT, HOV, check_step, fly_pid

pytestmark = pytest.mark.physics

ENV_IDS = [ATT, HOV]


def _level_start(api: Any, env: Any, **kw: Any) -> tuple[Any, dict[str, Any]]:
    return env.reset(seed=7, options=api.options(q_target=[1.0, 0, 0, 0], **kw))


# --- T6: nominal zero action is hover; motor signs through the env -------------------------------
def test_zero_action_hovers_for_half_a_second(api: Any, mk: Any) -> None:
    """a = 0 is the nominal hover command (ADR §3): |v_z| < 5 cm/s and tilt < 2 deg after 0.5 s.

    The 0.06 mm CoM offset of the CAD model tilts the airframe ~1 deg by then (ADR estimate)."""
    env = mk(ATT)
    _, info = _level_start(api, env, pos=(0.0, 0.0, 3.0))
    z0 = info["true"]["pos_w"][2]
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert not terminated
        assert not truncated
    true = info["true"]
    assert abs(true["linvel_w"][2]) < 0.05
    assert abs(true["pos_w"][2] - z0) < 0.02
    assert np.degrees(rot.tilt_angle(true["quat_wxyz"])) < 2.0
    assert np.degrees(info["metrics"]["tilt_rad"]) < 2.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize(
    ("motors", "axis", "expected_sign", "name"),
    [
        ((0, 3), 0, -1.0, "right motors M0+M3 -> roll rate x < 0"),
        ((0, 2), 1, -1.0, "front motors M0+M2 -> pitch rate y < 0 (nose up)"),
        ((2, 3), 2, +1.0, "CW pair M2+M3 -> yaw rate z > 0 (yaw left)"),
        ((0, 1), 2, -1.0, "CCW pair M0+M1 -> yaw rate z < 0"),
    ],
)
def test_motor_action_signs_through_the_env(  # noqa: PLR0917
    api: Any, mk: Any, motors: tuple[int, int], axis: int, expected_sign: float, name: str,
    sign: float,
) -> None:  # fmt: skip
    """docs/conventions.md sign table, observed on the *true* body rate after 0.1 s."""
    env = mk(ATT)
    _level_start(api, env, pos=(0.0, 0.0, 3.0))
    a = np.zeros(4, dtype=np.float32)
    a[list(motors)] = 0.1 * sign
    for _ in range(10):
        _, _, terminated, _, info = env.step(a)
        assert not terminated
    w = np.asarray(info["true"]["angvel_b"], dtype=np.float64)
    assert w[axis] * expected_sign * sign > 0.02, f"{name}: angvel_b = {w}"
    # the other two axes stay small (the pair cancels there)
    others = [i for i in range(3) if i != axis]
    assert np.max(np.abs(w[others])) < 0.35 * abs(w[axis]) + 0.05, f"{name}: crosstalk {w}"


# --- T4: 10 000 random steps ---------------------------------------------------------------------
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reward_terms_over_10k_random_steps(mk: Any, env_id: str) -> None:
    env = mk(env_id)
    rng = np.random.default_rng(21)
    env.reset(seed=21)
    n_term = n_att_pos = 0
    for k in range(10_000):
        # a mix of hard random actions (they crash) and gentle ones (long episodes, no clipping)
        a = rng.uniform(-1, 1, 4) if (k // 500) % 2 == 0 else 0.05 * rng.standard_normal(4)
        _, r, terminated, truncated, info = env.step(a.astype(np.float32))
        check_step(env_id, r, info["reward_terms"], terminated)
        n_term += terminated
        n_att_pos += r > 0.5
        if terminated or truncated:
            env.reset()
    assert n_term > 20, "the random policy should crash many episodes"
    assert n_att_pos > 100, "some steps should be near-hover (reward > 0.5)"


# --- T7: truncation exactly at 500 / 800 steps --------------------------------------------------
@pytest.mark.parametrize(("env_id", "n_steps"), [(ATT, 500), (HOV, 800)])
def test_default_episode_lengths_truncate_exactly(mk: Any, env_id: str, n_steps: int) -> None:
    env = mk(env_id)
    traj = fly_pid(env, env_id, n_steps + 50)
    assert len(traj) == n_steps, "episode did not run to the time limit"
    for terminated, truncated, info in traj[:-1]:
        assert not terminated
        assert not truncated
        assert info["termination"] is None
    terminated, truncated, info = traj[-1]
    assert truncated is True
    assert terminated is False
    assert info["termination"] == "time_limit"
    assert info["episode_success"] is True  # the PID holds the target through the final second
