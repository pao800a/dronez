"""Gymnasium wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from dronez.envs.obs import ObsLayout

F32 = NDArray[np.float32]


class FixedObsNormalization(gym.ObservationWrapper[F32, F32, F32]):
    """Applies the frozen ``ObsLayout`` normalization ``clip((x - offset) / scale, +-clip)``.

    Equals what VecNormalize with ``FixedRunningMeanStd`` (and the ONNX graph) computes, up to the
    ``epsilon`` of the variance; useful to run a policy on a single env without VecNormalize.
    """

    def __init__(self, env: gym.Env[Any, Any], layout: ObsLayout, clip: float = 10.0) -> None:
        super().__init__(env)
        self.layout = layout
        self.clip = clip
        self.observation_space = gym.spaces.Box(-clip, clip, (layout.dim,), np.float32)

    def observation(self, observation: F32) -> F32:
        return self.layout.normalize(observation, self.clip)
