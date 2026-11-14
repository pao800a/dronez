"""T12 (ADR-0006 §2): frozen observation normalization with adaptive reward normalization."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

pytestmark = pytest.mark.fast

ATT = "DronezAttitude-v0"


def _vecnorm(api: ModuleType, venv: DummyVecEnv, layout: Any) -> VecNormalize:
    """The training-time construction of ADR-0006 §2 (``VecNormalize`` + a frozen ``obs_rms``)."""
    vn = VecNormalize(
        venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99
    )
    vn.obs_rms = api.FixedRunningMeanStd(mean=layout.offset, var=layout.scale**2)
    return vn


@pytest.fixture
def setup(api: ModuleType, model_cache: None) -> Any:
    venv = DummyVecEnv([api.make_fn(ATT, episode_len_s=1.0) for _ in range(2)])
    layout = api.ObsLayout.attitude()
    vn = _vecnorm(api, venv, layout)
    vn.seed(0)
    yield vn, layout
    vn.close()


def _rollout(vn: VecNormalize, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vn.reset()
    for _ in range(n):
        vn.step(rng.uniform(-0.3, 0.3, (vn.num_envs, 4)).astype(np.float32))


def test_statistics_equal_the_layout_and_do_not_move_after_rollouts(setup: Any) -> None:
    vn, layout = setup
    mean0, var0 = np.array(vn.obs_rms.mean), np.array(vn.obs_rms.var)
    np.testing.assert_array_equal(mean0, layout.offset)
    np.testing.assert_array_equal(var0, layout.scale**2)
    count0 = vn.obs_rms.count
    assert vn.training, "the point is that stats are frozen while training=True"
    for k in range(2):
        _rollout(vn, 150, seed=k)  # 2 rollouts, with episode resets in between (1 s episodes)
    np.testing.assert_array_equal(vn.obs_rms.mean, mean0)
    np.testing.assert_array_equal(vn.obs_rms.var, var0)
    assert vn.obs_rms.count == count0


def test_reward_statistics_stay_adaptive(setup: Any) -> None:
    vn, _ = setup
    count0 = vn.ret_rms.count
    _rollout(vn, 100, seed=0)
    assert vn.ret_rms.count > count0 + 100
    assert vn.ret_rms.var != 1.0


def test_update_is_a_noop_and_copy_keeps_the_class(api: ModuleType, setup: Any) -> None:
    vn, layout = setup
    rms = vn.obs_rms
    assert isinstance(rms, api.FixedRunningMeanStd)
    rms.update(np.random.default_rng(0).normal(5.0, 3.0, (64, layout.dim)))
    rms.update_from_moments(np.full(layout.dim, 5.0), np.full(layout.dim, 9.0), 1000)
    np.testing.assert_array_equal(rms.mean, layout.offset)
    np.testing.assert_array_equal(rms.var, layout.scale**2)
    for dup in (rms.copy(), copy.copy(rms), copy.deepcopy(rms)):
        assert isinstance(dup, api.FixedRunningMeanStd)
        np.testing.assert_array_equal(dup.var, layout.scale**2)
    assert dup is not rms


def test_normalized_obs_follow_the_formula_and_the_clip(setup: Any) -> None:
    vn, layout = setup
    obs = vn.reset()
    raw = vn.get_original_obs()
    expected = np.clip(
        (raw - layout.offset) / np.sqrt(layout.scale**2 + vn.epsilon), -vn.clip_obs, vn.clip_obs
    )
    np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-6)
    # r_err_6d of an identity error is [1,0,0,0,1,0] / scale 1; gyro is divided by 5
    assert obs.shape == (2, layout.dim)
    # an extreme raw value saturates at +-clip_obs
    big = np.full((1, layout.dim), 1e6)
    np.testing.assert_array_equal(np.abs(vn.normalize_obs(big)), vn.clip_obs)


def test_fixed_normalization_wrapper_agrees_with_vecnormalize(api: ModuleType, setup: Any) -> None:
    vn, layout = setup
    x = np.random.default_rng(0).normal(0.0, 3.0, (8, layout.dim)).astype(np.float32)
    ref = vn.normalize_obs(x)
    np.testing.assert_allclose(layout.normalize(x), ref, rtol=1e-6, atol=1e-6)


def test_vecnormalize_pkl_reloads_to_the_fixed_class(
    api: ModuleType, setup: Any, tmp_path: Path
) -> None:
    vn, layout = setup
    _rollout(vn, 60, seed=3)
    path = tmp_path / "vecnormalize.pkl"
    vn.save(str(path))
    assert path.is_file()
    venv2 = DummyVecEnv([api.make_fn(ATT)])
    try:
        loaded = VecNormalize.load(str(path), venv2)
        assert isinstance(loaded.obs_rms, api.FixedRunningMeanStd)
        np.testing.assert_array_equal(loaded.obs_rms.mean, layout.offset)
        np.testing.assert_array_equal(loaded.obs_rms.var, layout.scale**2)
        assert loaded.clip_obs == 10.0
        assert loaded.epsilon == vn.epsilon
        assert loaded.ret_rms.count == vn.ret_rms.count  # the reward stats travel along
        # and the pickle carries mu, sigma^2, eps and the clip that the ONNX export needs
        with path.open("rb") as f:
            raw = pickle.load(f)
        assert hasattr(raw.obs_rms, "mean")
        assert hasattr(raw.obs_rms, "var")
        loaded.training = True
        loaded.reset()
        loaded.step(np.zeros((1, 4), dtype=np.float32))
        np.testing.assert_array_equal(loaded.obs_rms.var, layout.scale**2)  # still frozen
    finally:
        venv2.close()
