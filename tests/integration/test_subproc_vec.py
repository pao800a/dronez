"""The env factory survives process boundaries: ``SubprocVecEnv`` workers, seeding, levels."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

pytestmark = pytest.mark.integration


def test_subproc_workers_step_seed_and_switch_level(api: ModuleType, model_cache: None) -> None:
    method = "spawn" if sys.platform == "win32" else "forkserver"
    venv = SubprocVecEnv([api.make_fn("DronezAttitude-v0", rank=i) for i in range(2)], method)
    try:
        venv.seed(5)
        obs = venv.reset()
        assert obs.shape == (2, 29)
        assert not np.array_equal(obs[0], obs[1]), "ranks must be seeded differently (seed + rank)"
        venv.env_method("set_curriculum_level", 2)
        assert venv.env_method("get_curriculum_level") == [2, 2]
        for _ in range(30):
            obs, rew, done, infos = venv.step(np.zeros((2, 4), dtype=np.float32))
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rew))
        assert len(infos) == 2
        assert done.shape == (2,)
    finally:
        venv.close()
