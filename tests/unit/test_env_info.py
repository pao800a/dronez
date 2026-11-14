"""T11 + T13 (ADR-0006 §2, §3, §10): info levels, privileged obs, and action clipping."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.fast

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"
ZERO = np.zeros(4, dtype=np.float32)

# docs/architecture.md §6: the keys marked (m) are what info_level "minimal" may emit
MINIMAL_KEYS = {
    "reward_terms",
    "metrics",
    "termination",
    "episode_seed",
    "curriculum_level",
    "episode_success",
    "privileged_obs",
}
REQUIRED_MINIMAL = {"reward_terms", "metrics", "termination", "episode_seed", "curriculum_level"}
FULL_ONLY = {"sensed", "true", "params", "disturbance", "target"}


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_minimal_info_carries_only_the_training_keys(mk: Any, env_id: str) -> None:
    env = mk(env_id, info_level="minimal", episode_len_s=0.2)
    env.reset(seed=0)
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(ZERO)
        assert set(info) <= MINIMAL_KEYS, f"extra keys {set(info) - MINIMAL_KEYS}"
        assert set(info) >= REQUIRED_MINIMAL
        assert not (FULL_ONLY & set(info))
        assert "privileged_obs" not in info
        assert ("episode_success" in info) == (terminated or truncated)
        assert set(info["metrics"]) >= {"att_err_rad", "tilt_rad", "saturation_frac"}
    assert truncated
    assert "episode_success" in info


@pytest.mark.parametrize("env_id", [ATT, HOV])
def test_full_info_has_the_measurement_and_truth_blocks(mk: Any, env_id: str) -> None:
    env = mk(env_id, info_level="full")
    _, info = env.reset(seed=0)
    _, _, _, _, info = env.step(ZERO)
    assert set(info) >= FULL_ONLY
    assert set(info["sensed"]) >= {"quat_wxyz", "gyro_b", "accel_b"}
    assert set(info["true"]) >= {"pos_w", "quat_wxyz", "linvel_w", "angvel_b"}
    assert set(info["params"]) >= {"mass_kg", "k_f", "tau_m_s", "latency_steps", "battery_scale"}
    assert set(info["disturbance"]) >= {"force_w_n", "torque_w_nm", "wind_w_m_s"}
    assert set(info["target"]) >= {"quat_wxyz", "pos_w"}


def test_info_level_never_changes_the_dynamics(mk: Any) -> None:
    runs = []
    for level in ("full", "minimal"):
        env = mk(ATT, info_level=level, start_level=4)
        rng = np.random.default_rng(0)
        env.reset(seed=5)
        out = []
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(rng.uniform(-0.3, 0.3, 4).astype(np.float32))
            out.append((obs.tobytes(), r, term, trunc))
        runs.append(out)
    assert runs[0] == runs[1]


@pytest.mark.parametrize("env_id", [ATT, HOV])
@pytest.mark.parametrize("info_level", ["full", "minimal"])
def test_privileged_obs_only_when_enabled(mk: Any, env_id: str, info_level: str) -> None:
    off = mk(env_id, info_level=info_level)
    off.reset(seed=1)
    assert "privileged_obs" not in off.step(ZERO)[4]
    on = mk(env_id, info_level=info_level, privileged_obs=True)
    on.reset(seed=1)
    info = on.step(ZERO)[4]
    priv = info["privileged_obs"]
    assert priv.shape == (46,)
    assert priv.dtype == np.float32
    assert np.all(np.isfinite(priv))
    assert on.observation_space.shape[0] == {ATT: 29, HOV: 34}[env_id], "obs must not grow"


def test_privileged_obs_content(api: ModuleType, mk: Any) -> None:
    env = mk(ATT, privileged_obs=True)
    q = [np.cos(0.1), np.sin(0.1), 0.0, 0.0]
    obs, info = env.reset(
        seed=1, options=api.options(quat=q, q_target=[1, 0, 0, 0], omega_b=(0.1, 0.2, 0.3))
    )
    priv = info["privileged_obs"]
    sl = api.layout_of(env).privileged_slices
    np.testing.assert_allclose(priv[sl["r_err_6d"]], obs[0:6], atol=1e-6)  # noise off: same as obs
    np.testing.assert_allclose(priv[sl["gravity_b"]], obs[6:9], atol=1e-6)
    np.testing.assert_allclose(priv[sl["angvel_b"]], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_array_equal(priv[sl["pos_err_h"]], 0.0)  # zeros in Attitude
    # level 0 is nominal (d = 0): every scale is 1, no latency, no disturbance
    np.testing.assert_allclose(priv[sl["mass_scale"]], 1.0)
    np.testing.assert_allclose(priv[sl["inertia_scale"]], 1.0)
    np.testing.assert_allclose(priv[sl["k_f_scale"]], 1.0)
    np.testing.assert_allclose(priv[sl["k_m_scale"]], 1.0)
    np.testing.assert_allclose(priv[sl["battery_scale"]], 1.0)
    np.testing.assert_array_equal(priv[sl["latency_s"]], 0.0)
    np.testing.assert_array_equal(priv[sl["force_w_n"]], 0.0)
    np.testing.assert_allclose(priv[sl["tau_m_s"]], 0.030, atol=1e-6)
    omega = priv[sl["omega_norm"]]
    assert np.all((omega > 0.2) & (omega < 0.5))  # hover is ~0.32 of omega_max


def test_privileged_obs_shows_the_domain_randomization(api: ModuleType, mk: Any) -> None:
    env = mk(ATT, privileged_obs=True, start_level=4)
    sl = api.layout_of(env).privileged_slices
    scales = []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        priv = info["privileged_obs"]
        scales.append(float(priv[sl["mass_scale"]][0]))
        assert priv[sl["mass_scale"]][0] == pytest.approx(
            info["params"]["mass_kg"] / env.unwrapped.nominal.mass_kg, rel=1e-6
        )
    assert np.std(scales) > 0.01


# --- T13: out-of-range actions are clipped, and the clipped value is what everything sees --------
WILD = np.array([3.0, -3.0, 0.5, -0.2], dtype=np.float32)
CLIPPED = np.array([1.0, -1.0, 0.5, -0.2], dtype=np.float32)


def test_out_of_range_action_equals_its_clipped_version(api: ModuleType, mk: Any) -> None:
    out = []
    for a in (WILD, CLIPPED):
        env = mk(ATT)
        env.reset(seed=2, options=api.options(q_target=[1, 0, 0, 0]))
        out.append(env.step(a))
    (o1, r1, t1, tr1, i1), (o2, r2, t2, tr2, i2) = out
    np.testing.assert_array_equal(o1, o2)  # bitwise: physics, history and sensors all agree
    assert (r1, t1, tr1) == (r2, t2, tr2)
    assert i1["reward_terms"] == i2["reward_terms"]
    np.testing.assert_array_equal(i1["true"]["thrust_n"], i2["true"]["thrust_n"])


def test_clipped_action_enters_the_history_and_the_reward(api: ModuleType, mk: Any) -> None:
    env = mk(ATT)
    obs, _ = env.reset(seed=2, options=api.options(q_target=[1, 0, 0, 0]))
    np.testing.assert_array_equal(obs[13:29], 0.0)
    obs, _, _, _, info = env.step(WILD)
    np.testing.assert_allclose(obs[13:17], CLIPPED, atol=1e-7)  # newest first
    np.testing.assert_array_equal(obs[17:29], 0.0)
    terms = info["reward_terms"]
    mean_sq = float(np.mean(CLIPPED.astype(np.float64) ** 2))  # 0.5725
    assert terms["act_mag"] == pytest.approx(-0.05 * mean_sq)
    assert terms["act_smooth"] == pytest.approx(-0.25 * mean_sq)  # a_prev = a0 = 0
    assert terms["act_sat"] == pytest.approx(-0.1 * 0.5)  # two of four at |a| >= 0.99
    assert info["metrics"]["saturation_frac"] == pytest.approx(0.5)
    # the next step: history shifts by one slot, newest first
    nxt = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    obs, *_ = env.step(nxt)
    np.testing.assert_allclose(obs[13:17], nxt, atol=1e-7)
    np.testing.assert_allclose(obs[17:21], CLIPPED, atol=1e-7)
    np.testing.assert_array_equal(obs[21:29], 0.0)


def test_history_keeps_only_the_last_four_actions(mk: Any) -> None:
    env = mk(ATT)
    env.reset(seed=0)
    acts = [np.full(4, v, dtype=np.float32) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    for a in acts:
        obs, *_ = env.step(a)
    hist = obs[13:29].reshape(4, 4)
    np.testing.assert_allclose(hist[:, 0], [0.5, 0.4, 0.3, 0.2], atol=1e-7)
