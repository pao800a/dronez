"""Gymnasium envs, observation layout, reward, domain randomization and curriculum.

Importing this package registers ``DronezAttitude-v0`` and ``DronezHover-v0``. May import ``utils``,
``cad``, ``sim`` and ``control``; never ``rl`` or ``eval``. Worker processes must not import torch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from dronez.envs.attitude_env import AttitudeEnv
from dronez.envs.base import DronezBaseEnv
from dronez.envs.config import EnvConfig
from dronez.envs.curriculum import ATTITUDE_LEVELS, HOVER_LEVELS, Curriculum, LevelSpec
from dronez.envs.hover_env import HoverEnv
from dronez.envs.obs import ObsBlock, ObsLayout
from dronez.envs.randomization import DomainRandomizer, DRConfig, DRDraw
from dronez.envs.rewards import RewardWeights, compute_reward, term_names
from dronez.envs.wrappers import FixedObsNormalization

ENV_IDS = {
    "DronezAttitude-v0": "dronez.envs.attitude_env:AttitudeEnv",
    "DronezHover-v0": "dronez.envs.hover_env:HoverEnv",
}

for _id, _entry in ENV_IDS.items():
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_entry)


@dataclass(frozen=True)
class EnvFactory:
    """Picklable env constructor (``SubprocVecEnv`` workers). Never reseeds (ADR-0006 §10)."""

    env_id: str
    cfg: EnvConfig
    drone: str = "default_5in"
    sim: str = "default"
    rank: int = 0
    seed: int = 0

    def __call__(self) -> gym.Env[Any, Any]:
        return gym.make(self.env_id, env_cfg=self.cfg, sim_cfg=self.sim, drone=self.drone)


def make_env(
    env_id: str,
    cfg: EnvConfig | None = None,
    rank: int = 0,
    seed: int = 0,
    *,
    drone: str = "default_5in",
    sim: str = "default",
) -> Callable[[], gym.Env[Any, Any]]:
    """Factory for ``DummyVecEnv`` / ``SubprocVecEnv``. ``rank`` / ``seed`` are recorded only: the
    ranks are seeded by ``PPO(seed=...)`` through ``env.reset(seed=...)``."""
    return EnvFactory(env_id, cfg or EnvConfig(), drone, sim, rank, seed)


__all__ = [
    "ATTITUDE_LEVELS",
    "ENV_IDS",
    "HOVER_LEVELS",
    "AttitudeEnv",
    "Curriculum",
    "DRConfig",
    "DRDraw",
    "DomainRandomizer",
    "DronezBaseEnv",
    "EnvConfig",
    "EnvFactory",
    "FixedObsNormalization",
    "HoverEnv",
    "LevelSpec",
    "ObsBlock",
    "ObsLayout",
    "RewardWeights",
    "compute_reward",
    "make_env",
    "term_names",
]
