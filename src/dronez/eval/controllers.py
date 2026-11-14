"""Controller construction for the eval harness, including learned policies.

``ControllerSpec`` is a small picklable factory (``make_controller`` of docs/architecture.md §2): a
worker process calls it to get a fresh controller. Baselines use the **nominal** vehicle and the
env's command map; ``PolicyController`` applies the checkpoint's frozen normalization itself (it
reads only ``obs``) and takes the deterministic action ``clip(mean)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dronez.control import CascadedPID, CommandMap, GeometricSO3, NominalVehicle
from dronez.control.base import Controller
from dronez.sim import Sensed

F32 = NDArray[np.float32]


class PolicyController:
    """A trained PPO policy as a ``Controller`` (motors mode). Imports torch lazily."""

    action_mode: Literal["motors", "ctbr"] = "motors"

    def __init__(self, model_zip: Path, vecnorm_pkl: Path, name: str = "policy") -> None:
        import pickle  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from stable_baselines3 import PPO  # noqa: PLC0415

        torch.set_num_threads(1)
        self.name = name
        self.model = PPO.load(str(model_zip), device="cpu")
        with Path(vecnorm_pkl).open("rb") as fh:
            vn = pickle.load(fh)
        self._mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
        self._var = np.asarray(vn.obs_rms.var, dtype=np.float64)
        self._eps = float(vn.epsilon)
        self._clip = float(vn.clip_obs)
        self._inv_std = 1.0 / np.sqrt(self._var + self._eps)

    @classmethod
    def from_checkpoint(cls, run_id: str, ckpt: str | None = None) -> PolicyController:
        """``runs/<run_id>/checkpoints/<ckpt>`` (``best`` | ``latest`` | ``step_...``)."""
        from dronez.rl.checkpoint import MODEL_FILE, VECNORM_FILE, resolve_checkpoint  # noqa: PLC0415

        path = resolve_checkpoint(run_id, ckpt)
        return cls(path / MODEL_FILE, path / VECNORM_FILE, name=f"{run_id}/{path.name}")

    def normalize(self, obs: F32) -> F32:
        x = np.clip(
            (np.asarray(obs, dtype=np.float64) - self._mean) * self._inv_std,
            -self._clip,
            self._clip,
        )
        return np.asarray(x, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        del seed  # memoryless: the action history is part of the obs

    def act(self, obs: F32, sensed: Sensed) -> F32:
        del sensed  # policies read the raw obs only
        a, _ = self.model.predict(self.normalize(obs), deterministic=True)
        return np.asarray(a, dtype=np.float32)


@dataclass(frozen=True)
class ControllerSpec:
    """Picklable factory: ``ControllerSpec("pid")()`` -> a fresh ``Controller``."""

    kind: Literal["pid", "geometric", "policy"]
    drone: str = "default_5in"
    sim: str = "default"
    authority: float = 1.0
    run_id: str | None = None
    ckpt: str | None = None

    def label(self) -> str:
        return self.kind if self.kind != "policy" else f"policy:{self.run_id}"

    def __call__(self) -> Controller:
        if self.kind == "policy":
            if self.run_id is None:
                raise ValueError("a policy spec needs run_id")
            return PolicyController.from_checkpoint(self.run_id, self.ckpt)
        from dronez.cad.schema import load_drone_config  # noqa: PLC0415
        from dronez.sim import load_sim_config  # noqa: PLC0415

        cfg = load_drone_config(self.drone)
        vehicle = NominalVehicle.from_drone(cfg)
        cmap = CommandMap.from_vehicle(vehicle, self.authority)
        dt = load_sim_config(self.sim).policy_dt_s
        ctrl: Any = (
            CascadedPID(vehicle, dt=dt, cmap=cmap)
            if self.kind == "pid"
            else GeometricSO3(vehicle, dt=dt, cmap=cmap)
        )
        return ctrl  # type: ignore[no-any-return]
