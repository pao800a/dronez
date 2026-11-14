"""T8 (ADR-0006 §10): determinism D1-D3, seed sensitivity, and D2 (stream independence)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dronez.sim import DisturbanceConfig, ImuNoiseConfig

pytestmark = pytest.mark.fast

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"


def _rollout(env: Any, seed: int, n: int = 60, action_seed: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(action_seed)
    obs, info = env.reset(seed=seed)
    rec: dict[str, list[Any]] = {"obs": [obs], "r": [], "flags": [], "terms": [], "true": []}
    rec["true"].append(np.concatenate([np.ravel(v) for v in info["true"].values()]))
    for _ in range(n):
        a = rng.uniform(-1, 1, 4).astype(np.float32) * 0.3
        obs, r, term, trunc, info = env.step(a)
        rec["obs"].append(obs)
        rec["r"].append(r)
        rec["flags"].append((term, trunc, info["termination"]))
        rec["terms"].append(tuple(info["reward_terms"].values()))
        rec["true"].append(np.concatenate([np.ravel(v) for v in info["true"].values()]))
        if term or trunc:
            break
    return rec


def _assert_identical(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert len(a["r"]) == len(b["r"])
    np.testing.assert_array_equal(np.array(a["obs"]), np.array(b["obs"]))  # bitwise
    assert a["r"] == b["r"]
    assert a["flags"] == b["flags"]
    assert a["terms"] == b["terms"]
    np.testing.assert_array_equal(np.array(a["true"]), np.array(b["true"]))


@pytest.mark.parametrize("env_id", [ATT, HOV])
@pytest.mark.parametrize("level", [0, 4])  # nominal, and DR + wind + impulses + noise scaling
def test_d1_same_seed_same_actions_is_bitwise_identical(
    api: ModuleType, mk: Any, env_id: str, level: int
) -> None:
    if env_id == HOV:
        level = 0 if level == 0 else 2  # the Hover table has H0..H3
    e1, e2 = mk(env_id, start_level=level), mk(env_id, start_level=level)
    first = _rollout(e1, seed=123)
    _assert_identical(first, _rollout(e2, seed=123))  # a fresh instance
    _assert_identical(first, _rollout(e1, seed=123))  # the same instance, after other episodes


def test_d3_reset_with_the_same_seed_gives_the_same_first_obs(mk: Any) -> None:
    env = mk(ATT, start_level=4)
    o1, i1 = env.reset(seed=77)
    env.step(np.zeros(4, dtype=np.float32))
    env.reset(seed=3)
    o2, i2 = env.reset(seed=77)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(i1["true"]["quat_wxyz"], i2["true"]["quat_wxyz"])
    assert i1["episode_seed"] == i2["episode_seed"] == 77


def test_unseeded_resets_differ_and_the_seed_is_reported(mk: Any) -> None:
    env = mk(ATT)
    o1, i1 = env.reset(seed=1)
    o2, i2 = env.reset(seed=2)
    assert not np.array_equal(o1, o2)
    assert i1["episode_seed"] != i2["episode_seed"]
    _, i3 = env.reset()  # drawn from the env's own np_random: still replayable from its report
    o3a, _ = env.reset(seed=i3["episode_seed"])
    _, i3b = env.reset(seed=i3["episode_seed"])
    assert i3b["episode_seed"] == i3["episode_seed"]
    np.testing.assert_array_equal(i3["true"]["quat_wxyz"], i3b["true"]["quat_wxyz"])
    assert o3a.shape == o1.shape


def test_distinct_seeds_give_distinct_episodes(mk: Any) -> None:
    env = mk(ATT, start_level=4)
    a = _rollout(env, seed=10)
    b = _rollout(env, seed=11)
    assert not np.array_equal(np.array(a["obs"][0]), np.array(b["obs"][0]))
    n = min(len(a["obs"]), len(b["obs"]))
    assert not np.array_equal(np.array(a["obs"][:n]), np.array(b["obs"][:n]))


def _init_signature(env: Any, info: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Initial qpos/qvel of the body origin (the CoM of ``info["true"]`` moves with DR) + target."""
    data = env.unwrapped.sim.data
    return (
        np.array(data.qpos),
        np.array(data.qvel),
        np.asarray(info["target"]["quat_wxyz"]),
        np.asarray(info["target"]["pos_w"]),
    )


def test_d2_noise_dr_and_disturbances_never_change_the_init_draws(api: ModuleType, mk: Any) -> None:
    """The initial state and the target depend on ``rng_init`` only (ADR-0006 §10)."""
    env = mk(ATT, start_level=4)
    ref = _init_signature(env, env.reset(seed=9)[1])
    nominal = env.unwrapped.nominal
    rd = api.DomainRandomizer(nominal)
    variants: list[dict[str, Any]] = [
        {"noise": ImuNoiseConfig.ideal()},
        {"noise": ImuNoiseConfig(gyro_noise_density=0.05)},
        {"disturbance": DisturbanceConfig(wind_mean_w_m_s=(3.0, 0.0, 0.0), wind_gust_std_m_s=1.0)},
        {"disturbance": DisturbanceConfig(random_impulse_count=2, random_impulse_force_n=2.0)},
        {"phys_params": rd.sample(np.random.default_rng(0), 1.0)},
    ]
    for opts in variants:
        got = _init_signature(env, env.reset(seed=9, options=opts)[1])
        for g, r in zip(got, ref, strict=True):
            np.testing.assert_array_equal(g, r, err_msg=str(sorted(opts)))


def test_d2_dr_width_does_not_change_the_init_draws(mk: Any) -> None:
    """Levels 4, 5, 6 share the init distribution and differ only in d: same seed, same start."""
    sigs = []
    for level in (4, 5, 6):
        env = mk(ATT, start_level=level)
        sigs.append(_init_signature(env, env.reset(seed=31)[1]))
    for s in sigs[1:]:
        for g, r in zip(s, sigs[0], strict=True):
            np.testing.assert_array_equal(g, r)
