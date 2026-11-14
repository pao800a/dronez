"""T4 (ADR-0006 §4): reward terms sum to r, stay inside the table ranges, and cannot be gamed."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dronez.utils import rotations as rot
from env_helpers import EPS, check_step

pytestmark = pytest.mark.fast

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_pure_reward_function_random_inputs(api: ModuleType, env_id: str) -> None:
    kind = "attitude" if env_id == ATT else "hover"
    w = api.RewardWeights.for_kind(kind)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        a = rng.uniform(-1, 1, 4) * rng.choice([0.05, 1.0])
        a_prev = rng.uniform(-1, 1, 4) * rng.choice([0.0, 1.0])
        if rng.random() < 0.3:
            a = np.sign(a)  # saturation
        terminated = bool(rng.random() < 0.2)
        r, terms = api.compute_reward(
            kind,
            w,
            att_err=float(rng.uniform(0, np.pi)) * rng.choice([0.0, 0.01, 1.0]),
            angvel_norm=float(rng.uniform(0, 30)) * rng.choice([0.0, 0.1, 1.0]),
            vz=float(rng.normal(0, 3)),
            vel_norm=float(rng.uniform(0, 5)),
            pos_err=float(rng.uniform(0, 4)) * rng.choice([0.0, 0.05, 1.0]),
            a=a,
            a_prev=a_prev,
            terminated=terminated,
        )
        check_step(env_id, r, terms, terminated)


def test_pure_reward_perfect_state_is_the_maximum(api: ModuleType) -> None:
    for kind, best in (("attitude", 1.5), ("hover", 1.7)):
        w = api.RewardWeights.for_kind(kind)
        z = np.zeros(4)
        r, terms = api.compute_reward(
            kind, w, att_err=0.0, angvel_norm=0.0, vz=0.0, vel_norm=0.0, pos_err=0.0,
            a=z, a_prev=z, terminated=False,
        )  # fmt: skip
        assert r == pytest.approx(best, abs=1e-12)
        assert terms["penalty_clip"] == 0.0


def test_dying_always_loses_to_surviving(api: ModuleType) -> None:
    """The penalty floor makes every surviving step >= 0 and the crash term costs 10."""
    w = api.RewardWeights.for_kind("attitude")
    worst_a = np.array([1.0, -1.0, 1.0, -1.0])
    r_live, _ = api.compute_reward(
        "attitude", w, att_err=np.pi, angvel_norm=30.0, vz=5.0, vel_norm=5.0, pos_err=5.0,
        a=worst_a, a_prev=-worst_a, terminated=False,
    )  # fmt: skip
    r_dead, _ = api.compute_reward(
        "attitude", w, att_err=0.0, angvel_norm=0.0, vz=0.0, vel_norm=0.0, pos_err=0.0,
        a=np.zeros(4), a_prev=np.zeros(4), terminated=True,
    )  # fmt: skip
    assert r_live >= -EPS
    assert r_dead < r_live


# --- through the env -----------------------------------------------------------------------------
@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_env_reward_terms_sum_and_ranges(mk: Any, env_id: str) -> None:
    env = mk(env_id)
    rng = np.random.default_rng(3)
    env.reset(seed=3)
    seen_terminal = 0
    for _ in range(400):
        a = env.action_space.sample() if rng.random() < 0.5 else rng.uniform(-1, 1, 4) * 0.1
        _, r, terminated, truncated, info = env.step(a.astype(np.float32))
        check_step(env_id, r, info["reward_terms"], terminated)
        if terminated or truncated:
            seen_terminal += terminated
            env.reset()
    assert seen_terminal > 0, "random +-1 actions should crash within 400 steps"


def test_env_perfect_state_gives_maximum_reward(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    q = rot.quat_from_yaw(0.3)
    env.reset(seed=0, options=api.options(quat=q, q_target=q))
    _, r, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert not terminated
    # level, at rest, a = 0 is nominal hover: after 10 ms the state is perfect to ~1e-4
    assert r == pytest.approx(1.5, abs=5e-3)
    assert info["reward_terms"]["alive"] == 1.0
    assert info["reward_terms"]["act_mag"] == 0.0
    assert info["reward_terms"]["act_smooth"] == 0.0


def test_env_att_term_is_monotone_in_the_geodesic_error(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    prev = 0.0
    for deg in (0.0, 5.0, 20.0, 60.0, 120.0, 179.0):
        e = np.deg2rad(deg)
        env.reset(
            seed=0,
            options=api.options(
                quat=rot.quat_from_axis_angle([1.0, 0, 0], e), q_target=[1, 0, 0, 0]
            ),
        )
        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        att = info["reward_terms"]["att"]
        assert not terminated
        assert att == pytest.approx(-0.5 * e / np.pi, abs=2e-3)  # e barely changes in 10 ms
        assert att <= prev + EPS
        if deg > 0:
            assert att < prev
        prev = att


def test_env_reward_uses_the_true_state_not_the_sensed_one(api: ModuleType, mk: Any) -> None:
    """Heavy sensor noise changes the obs but not the reward of an identical trajectory."""
    from dronez.sim import ImuNoiseConfig

    loud = ImuNoiseConfig(gyro_noise_density=0.05, quat_noise_rad=0.05)
    rewards = []
    for noise in ("ideal", loud):
        env = mk(ATT)
        env.reset(seed=4, options=api.options(noise=noise, q_target=[1, 0, 0, 0]))
        rewards.append(env.step(np.full(4, 0.1, dtype=np.float32))[1])
    assert rewards[0] == pytest.approx(rewards[1], abs=1e-9)
