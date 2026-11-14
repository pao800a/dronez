"""T7 (ADR-0006 §5): every termination reason, the time limit, and the NaN guard."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.fast

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"
ZERO = np.zeros(4, dtype=np.float32)


def _run_until_done(
    env: Any, action: Any, max_steps: int
) -> tuple[int, bool, bool, dict[str, Any]]:
    for k in range(1, max_steps + 1):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return k, terminated, truncated, info
    pytest.fail(f"no termination within {max_steps} steps")


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_free_fall_ends_in_ground_contact(api: ModuleType, mk: Any, env_id: str) -> None:
    env = mk(env_id)
    kw = {"pos": (0.0, 0.0, 0.5), "p_target": (0.0, 0.0, 0.5)}
    env.reset(seed=0, options=api.options(q_target=[1, 0, 0, 0], **kw))
    k, terminated, truncated, info = _run_until_done(env, -np.ones(4, np.float32), 100)
    assert terminated
    assert not truncated
    assert info["termination"] == "ground_contact"
    assert 20 < k < 60  # 0.5 m free fall takes ~0.3 s
    assert info["reward_terms"]["crash"] == pytest.approx(-10.0)


def test_rate_limit_is_25_rad_s(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    env.reset(seed=0, options=api.options(omega_b=(30.0, 0.0, 0.0), q_target=[1, 0, 0, 0]))
    _, _, terminated, truncated, info = env.step(ZERO)
    assert terminated
    assert not truncated
    assert info["termination"] == "rate_limit"
    # just inside the limit the episode goes on
    env.reset(seed=0, options=api.options(omega_b=(20.0, 0.0, 0.0), q_target=[1, 0, 0, 0]))
    _, _, terminated, _, info = env.step(ZERO)
    assert not terminated
    assert info["termination"] is None


@pytest.mark.parametrize(
    ("vel", "name"),
    [((200.0, 0.0, 0.0), "horizontal > 20 m"), ((0.0, 0.0, 300.0), "z > z0 + 10 m")],
)
def test_attitude_out_of_bounds(
    api: ModuleType, mk: Any, vel: tuple[float, ...], name: str
) -> None:
    env = mk(ATT)
    env.reset(seed=0, options=api.options(vel=vel, q_target=[1, 0, 0, 0]))
    k, terminated, _, info = _run_until_done(env, ZERO, 60)
    assert terminated, name
    assert info["termination"] == "out_of_bounds", name
    assert k <= 12  # 20 m at 200 m/s is 10 policy steps; 10 m at 300 m/s is 4


def test_hover_out_of_bounds_is_3_m_from_the_target(api: ModuleType, mk: Any) -> None:
    env = mk(HOV)
    p_t = (0.0, 0.0, 2.0)
    env.reset(
        seed=0,
        options=api.options(pos=p_t, vel=(60.0, 0.0, 0.0), q_target=[1, 0, 0, 0], p_target=p_t),
    )
    k, terminated, _, info = _run_until_done(env, ZERO, 60)
    assert terminated
    assert info["termination"] == "out_of_bounds"
    assert k <= 8  # 3 m at 60 m/s is ~5 policy steps


def test_nan_state_is_a_crash_with_a_finite_observation(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    env.reset(seed=0, options=api.options(q_target=[1, 0, 0, 0]))
    obs1, *_ = env.step(ZERO)
    api.sim_of(env).data.qvel[:] = np.nan  # a numerical blow-up between two policy steps
    obs2, r, terminated, truncated, info = env.step(ZERO)
    assert terminated
    assert not truncated
    assert info["termination"] == "crash"
    assert np.all(np.isfinite(obs2)), "the returned obs must be finite"
    np.testing.assert_array_equal(obs2, obs1)  # ADR §5: 'the last finite one'
    assert np.isfinite(r)
    assert info["reward_terms"]["crash"] == pytest.approx(-10.0)


def test_nan_action_is_a_crash_too(api: ModuleType, mk: Any) -> None:
    """SB3 can never send NaN, but a broken controller can: it must not poison the sim."""
    env = mk(ATT)
    env.reset(seed=0)
    obs, r, terminated, _, info = env.step(np.full(4, np.nan, dtype=np.float32))
    assert terminated
    assert info["termination"] == "crash"
    assert np.all(np.isfinite(obs))
    assert np.isfinite(r)


@pytest.mark.parametrize(("env_id", "default_len_s"), [(ATT, 5.0), (HOV, 8.0)])
def test_default_episode_lengths_are_5_and_8_seconds(
    mk: Any, env_id: str, default_len_s: float
) -> None:
    env = mk(env_id)
    assert env.unwrapped.max_steps == round(default_len_s / 0.01)
    assert env.unwrapped.max_steps == {ATT: 500, HOV: 800}[env_id]


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_time_limit_truncates_exactly_and_never_terminates(
    api: ModuleType, mk: Any, env_id: str
) -> None:
    env = mk(env_id, episode_len_s=0.3)  # 30 steps; a = 0 is a stable hover for that long
    p_t = (0.0, 0.0, 2.0)
    env.reset(seed=0, options=api.options(pos=p_t, q_target=[1, 0, 0, 0], p_target=p_t))
    for k in range(1, 31):
        _, _, terminated, truncated, info = env.step(ZERO)
        assert not terminated, f"terminated at step {k}: {info['termination']}"
        assert truncated == (k == 30), f"truncated flag wrong at step {k}"
        assert info["termination"] == ("time_limit" if k == 30 else None)
    # a new episode starts the count from zero again
    env.reset(seed=1, options=api.options(pos=p_t, q_target=[1, 0, 0, 0], p_target=p_t))
    _, _, _, truncated, _ = env.step(ZERO)
    assert not truncated
