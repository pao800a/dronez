"""The ONE place where the env/RL tests learn the names and signatures of the Phase 4 API.

ADR-0006 fixes the behaviour; the Python spelling (modules, constructors, attribute names) is
resolved here, lazily, symbol by symbol, so that a rename in ``src/dronez/{envs,rl,eval}`` is a
one-line fix in this file and a symbol that does not exist yet only fails the tests that use it.

Tests reach this module through the ``api`` fixture (``tests/conftest.py``): ``api.AttitudeEnv``,
``api.hf_rms``, ``api.make(...)``. Nothing else in ``tests/`` imports the Phase 4 packages.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from dronez.sim import ImuNoiseConfig

ATT = "DronezAttitude-v0"
HOV = "DronezHover-v0"

# symbol -> modules that may define it (first hit wins). Edit here after a rename.
_LOCATIONS: dict[str, tuple[str, ...]] = {
    # envs
    "EnvConfig": ("dronez.envs", "dronez.envs.config", "dronez.envs.base"),
    "ObsLayout": ("dronez.envs", "dronez.envs.obs", "dronez.envs.layout"),
    "DomainRandomizer": ("dronez.envs", "dronez.envs.randomization"),
    "make_env": ("dronez.envs", "dronez.envs.factory", "dronez.envs.base"),
    "AttitudeEnv": ("dronez.envs", "dronez.envs.attitude"),
    "HoverEnv": ("dronez.envs", "dronez.envs.hover"),
    # rl
    "FixedRunningMeanStd": ("dronez.rl.normalization", "dronez.rl"),
    "CurriculumCallback": ("dronez.rl.callbacks", "dronez.rl"),
    "TrainConfig": ("dronez.rl", "dronez.rl.config"),
    "load_train_config": ("dronez.rl", "dronez.rl.config"),
    "train": ("dronez.rl", "dronez.rl.train"),
    "load_checkpoint": ("dronez.rl", "dronez.rl.checkpoint"),
    "list_checkpoints": ("dronez.rl", "dronez.rl.checkpoint"),
    "make_vec_normalize": ("dronez.rl", "dronez.rl.normalization"),
    "Curriculum": ("dronez.envs.curriculum", "dronez.envs"),
    "ATTITUDE_LEVELS": ("dronez.envs.curriculum", "dronez.envs"),
    "HOVER_LEVELS": ("dronez.envs.curriculum", "dronez.envs"),
    "DRConfig": ("dronez.envs.randomization", "dronez.envs"),
    "compute_reward": ("dronez.envs.rewards", "dronez.envs"),
    "RewardWeights": ("dronez.envs.rewards", "dronez.envs"),
    # eval (ADR §9)
    "hf_rms": ("dronez.eval.metrics",),
    "success_recovery": ("dronez.eval.metrics",),
    "hf_energy_fraction": ("dronez.eval.metrics",),
    "mean_abs_delta": ("dronez.eval.metrics",),
    "time_to_recover": ("dronez.eval.metrics",),
}


def __getattr__(name: str) -> Any:
    """Lazy symbol lookup: ``env_api.hf_rms`` imports ``dronez.eval.metrics`` on first use."""
    if name.startswith("__") or name not in _LOCATIONS:
        raise AttributeError(name)
    tried: list[str] = []
    missing_module = True
    for mod_name in _LOCATIONS[name]:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("dronez"):
                raise
            tried.append(f"{mod_name} (no module)")
            continue
        missing_module = False
        if hasattr(mod, name):
            return getattr(mod, name)
        tried.append(f"{mod_name} (no attribute)")
    msg = f"env_api: `{name}` not found; looked in {', '.join(tried)}"
    if missing_module:
        pytest.skip(f"{msg}: the module is not written yet")
    pytest.fail(msg, pytrace=False)


def _sym(name: str) -> Any:
    return __getattr__(name)


# --- construction -------------------------------------------------------------------------------
def make(env_id: str = ATT, **cfg: Any) -> Any:
    """One env (rank 0, seed 0) with ``EnvConfig(**cfg)``. ``info_level`` defaults to ``full``."""
    cfg.setdefault("info_level", "full")
    config = _sym("EnvConfig")(**cfg)
    return _sym("make_env")(env_id, config, rank=0, seed=0)()


def make_fn(env_id: str = ATT, rank: int = 0, seed: int = 0, **cfg: Any) -> Any:
    """The thunk ``make_env`` returns (for ``DummyVecEnv`` / ``SubprocVecEnv``)."""
    cfg.setdefault("info_level", "full")
    return _sym("make_env")(env_id, _sym("EnvConfig")(**cfg), rank=rank, seed=seed)


def sim_of(env: Any) -> Any:
    """The ``DroneSim`` inside the env."""
    return env.unwrapped.sim


def layout_of(env: Any) -> Any:
    """The ``ObsLayout`` of the env."""
    return env.unwrapped.layout


# --- reset options (ADR-0006 §9, plain data) ------------------------------------------------------
def options(
    *,
    pos: Any = (0.0, 0.0, 3.0),
    quat: Any = (1.0, 0.0, 0.0, 0.0),
    vel: Any = (0.0, 0.0, 0.0),
    omega_b: Any = (0.0, 0.0, 0.0),
    a0: Any = (0.0, 0.0, 0.0, 0.0),
    q_target: Any = None,
    p_target: Any = None,
    noise: Any = "ideal",
    **extra: Any,
) -> dict[str, Any]:
    """``reset(options=...)`` dict.

    ``qvel0`` = (world linear velocity of the body origin, body-frame angular velocity), the
    ``DroneSim.reset`` convention. ``noise="ideal"`` switches sensor noise off; ``None`` leaves
    the env default."""
    opts: dict[str, Any] = {
        "qpos0": np.concatenate([np.asarray(pos, float), np.asarray(quat, float)]),
        "qvel0": np.concatenate([np.asarray(vel, float), np.asarray(omega_b, float)]),
        "a0": np.asarray(a0, float),
    }
    if q_target is not None:
        opts["target_schedule"] = [(0.0, np.asarray(q_target, float))]
    if p_target is not None:
        opts["pos_target"] = np.asarray(p_target, float)
    if noise == "ideal":
        opts["noise"] = ImuNoiseConfig.ideal()
    elif noise is not None:
        opts["noise"] = noise
    opts.update(extra)
    return opts


def set_level(venv_or_env: Any, level: int) -> None:
    """``VecEnv.env_method("set_curriculum_level", k)`` (or the plain method on a bare env)."""
    if hasattr(venv_or_env, "env_method"):
        venv_or_env.env_method("set_curriculum_level", level)
    else:
        venv_or_env.unwrapped.set_curriculum_level(level)
