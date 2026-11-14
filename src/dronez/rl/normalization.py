"""Frozen observation normalization (ADR-0006 §2).

``VecNormalize`` runs its observation statistics on running moments, which a curriculum breaks:
they fit level 0 (tiny rates, near-constant gravity direction) and then clip later levels at
+-10 sigma. We replace ``obs_rms`` by ``FixedRunningMeanStd`` whose mean / var come from the
``ObsLayout`` offsets / scales^2: ``update`` is a no-op, ``copy`` keeps the class, and the pickled
``vecnormalize.pkl`` still carries mu, sigma^2, epsilon and the clip for the ONNX graph. Reward
normalization (``ret_rms``) stays adaptive. The class must stay importable (it is unpickled).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from dronez.envs.obs import ObsLayout


class FixedRunningMeanStd(RunningMeanStd):
    """``RunningMeanStd`` with constant moments."""

    def __init__(self, mean: ArrayLike, var: ArrayLike, count: float = 1.0e4) -> None:
        m = np.array(mean, dtype=np.float64)
        super().__init__(epsilon=count, shape=m.shape)
        self.mean = m
        self.var = np.array(var, dtype=np.float64)
        self.count = float(count)

    def copy(self) -> FixedRunningMeanStd:
        return FixedRunningMeanStd(self.mean.copy(), self.var.copy(), float(self.count))

    def combine(self, other: RunningMeanStd) -> None:
        return

    def update(self, arr: np.ndarray) -> None:
        return

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        return


def make_vec_normalize(
    venv: VecEnv,
    layout: ObsLayout,
    *,
    gamma: float = 0.99,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    norm_reward: bool = True,
    training: bool = True,
) -> VecNormalize:
    """``VecNormalize`` with frozen observation statistics from ``layout``."""
    vn = VecNormalize(
        venv,
        training=training,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
    )
    # SB3 annotates obs_rms as the dict-obs variant here; FixedRunningMeanStd is a RunningMeanStd
    vn.obs_rms = FixedRunningMeanStd(mean=layout.offset, var=layout.scale**2)  # type: ignore[assignment]
    return vn
