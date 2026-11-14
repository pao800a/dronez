"""``TrainConfig``: one YAML per stage (``configs/train/<stage>.yaml``), ADR-0006 §8."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dronez.envs.config import EnvConfig
from dronez.utils.config import load_model


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSection(_Strict):
    stage: str = "smoke"
    seed: int = 0
    total_timesteps: int = Field(default=8192, gt=0)
    n_envs: int = Field(default=2, ge=1)
    device: Literal["cpu", "xpu"] = "cpu"
    vec_env: Literal["dummy", "subproc"] = "dummy"
    #: torch threads of the main process (workers always run with OMP/MKL = 1)
    torch_threads: int = Field(default=1, ge=1)


class PPOSection(_Strict):
    n_steps: int = 1024  # per env
    batch_size: int = 256
    n_epochs: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03


class PolicySection(_Strict):
    pi: tuple[int, ...] = (64, 64)
    vf: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    activation: Literal["tanh", "relu"] = "tanh"
    ortho_init: bool = True


class NormalizeSection(_Strict):
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    norm_reward: bool = True
    gamma: float = 0.99


class CurriculumSection(_Strict):
    enabled: bool = False
    #: ``eval_every`` steps between deterministic mini-evals; ``eval_episodes`` per eval
    eval_every: int = Field(default=1_000_000, gt=0)
    eval_episodes: int = Field(default=100, ge=1)
    consecutive: int = Field(default=2, ge=1)
    #: finish the run when this level is reached (None: run the full budget)
    stop_level: int | None = None


class InitFrom(_Strict):
    run_id: str
    checkpoint: str | None = None  # best | latest | step_...; default: best, else latest


class TrainConfig(_Strict):
    run: RunSection = RunSection()
    env_id: Literal["DronezAttitude-v0", "DronezHover-v0"] = "DronezAttitude-v0"
    env: EnvConfig = EnvConfig(info_level="minimal")
    drone: str = "default_5in"
    sim: str = "default"
    ppo: PPOSection = PPOSection()
    policy: PolicySection = PolicySection()
    normalize: NormalizeSection = NormalizeSection()
    curriculum: CurriculumSection = CurriculumSection()
    #: checkpoint every N steps (rounded up to a rollout boundary)
    checkpoint_every: int = Field(default=500_000, gt=0)
    keep_checkpoints: int = Field(default=5, ge=1)
    #: episodes of the final deterministic mini-eval (smoke: 4); 0 = none
    final_eval_episodes: int = Field(default=0, ge=0)
    #: stage hand-off: weights + ret_rms from an earlier run, plus ``curriculum`` start level
    init_from: InitFrom | None = None
    status_every_s: float = Field(default=30.0, gt=0.0)


def load_train_config(name_or_path: str | Path) -> TrainConfig:
    return load_model(TrainConfig, name_or_path, "train")
