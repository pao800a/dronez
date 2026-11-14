"""SB3 callbacks: status, checkpoints, TensorBoard scalars and curriculum promotion."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from dronez.envs.curriculum import Curriculum
from dronez.rl.checkpoint import prune_checkpoints, save_checkpoint
from dronez.rl.run_dir import RunDir
from dronez.utils.device import describe_devices

log = logging.getLogger(__name__)

TERMINATION_KEYS = ("crash", "rate_limit", "ground_contact", "out_of_bounds", "time_limit")


def _vecnorm(env: VecEnv | None) -> VecNormalize | None:
    e = env
    while e is not None:
        if isinstance(e, VecNormalize):
            return e
        e = getattr(e, "venv", None)
    return None


class InfoLogCallback(BaseCallback):
    """TensorBoard scalars from ``info`` (rollout means), ADR-0006 §4.

    ``reward/<term>`` (per-step mean of the weighted contribution), ``episode/success_rate``,
    ``episode/termination/<reason>`` (fraction of episodes), ``action/saturation_frac``,
    ``action/mean_abs_delta``, ``obs/clip_frac``, ``metrics/att_err_rad``.
    """

    def __init__(self, curriculum: Curriculum | None = None, clip_obs: float = 10.0) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.clip_obs = clip_obs
        self._terms: dict[str, float] = defaultdict(float)
        self._steps = 0
        self._sat = 0.0
        self._att = 0.0
        self._dact = 0.0
        self._dact_n = 0
        self._clip = 0.0
        self._clip_n = 0
        self._ep_done = 0
        self._ep_succ = 0
        self._ep_term: dict[str, int] = defaultdict(int)
        self._prev_a: np.ndarray | None = None
        self._prev_done: np.ndarray = np.zeros(0, dtype=bool)

    def _reset_acc(self) -> None:
        self._terms.clear()
        self._steps = self._dact_n = self._clip_n = self._ep_done = self._ep_succ = 0
        self._sat = self._att = self._dact = self._clip = 0.0
        self._ep_term.clear()

    def _on_rollout_start(self) -> None:
        self._reset_acc()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals["infos"]
        dones = self.locals["dones"]
        for info in infos:
            for k, v in info["reward_terms"].items():
                self._terms[k] += v
            m = info["metrics"]
            self._sat += m["saturation_frac"]
            self._att += m["att_err_rad"]
            self._steps += 1
        a = np.asarray(self.locals["clipped_actions"], dtype=np.float64)
        if self._prev_a is not None and self._prev_a.shape == a.shape:
            ok = ~self._prev_done
            if ok.any():
                self._dact += float(np.abs(a[ok] - self._prev_a[ok]).mean(axis=1).sum())
                self._dact_n += int(ok.sum())
        self._prev_a, self._prev_done = a, np.asarray(dones, dtype=bool)
        new_obs = self.locals["new_obs"]
        self._clip += float(np.mean(np.abs(new_obs) >= self.clip_obs - 1e-6, axis=1).sum())
        self._clip_n += new_obs.shape[0]
        for info, done in zip(infos, dones, strict=True):
            if done:
                self._ep_done += 1
                self._ep_succ += bool(info.get("episode_success", False))
                self._ep_term[str(info.get("termination"))] += 1
        return True

    def _on_rollout_end(self) -> None:
        n = max(self._steps, 1)
        rec = self.logger.record
        for k, v in self._terms.items():
            rec(f"reward/{k}", v / n)
        rec("action/saturation_frac", self._sat / n)
        rec("metrics/att_err_rad", self._att / n)
        if self._dact_n:
            rec("action/mean_abs_delta", self._dact / self._dact_n)
        if self._clip_n:
            rec("obs/clip_frac", self._clip / self._clip_n)
        if self._ep_done:
            rec("episode/success_rate", self._ep_succ / self._ep_done)
            for k in TERMINATION_KEYS:
                rec(f"episode/termination/{k}", self._ep_term.get(k, 0) / self._ep_done)
        if self.curriculum is not None:
            rec("curriculum/level", self.curriculum.level)
            rec("curriculum/dr_scale", self.curriculum.spec.dr_scale)
            rec("curriculum/stalled", float(self.curriculum.stalled))


class StatusCallback(BaseCallback):
    """Writes ``status.json`` at most every ``every_s`` seconds (and at rollout ends)."""

    def __init__(
        self,
        run: RunDir,
        total_timesteps: int,
        base: dict[str, Any],
        every_s: float = 30.0,
        curriculum: Curriculum | None = None,
    ) -> None:
        super().__init__()
        self.run, self.total, self.base, self.every_s = run, total_timesteps, base, every_s
        self.curriculum = curriculum
        self._t0 = time.time()
        self._last = 0.0
        self._ts0 = 0
        self.latest_checkpoint: str | None = None
        self.best_checkpoint: str | None = None
        self.best_metric: dict[str, Any] | None = None
        self.last_eval: dict[str, Any] | None = None

    def _on_training_start(self) -> None:
        self._ts0 = self.model.num_timesteps
        self.write("running")

    def write(self, state: str, error: str | None = None) -> None:
        done = self.model.num_timesteps
        dt = max(time.time() - self._t0, 1e-6)
        sps = (done - self._ts0) / dt
        remaining = max(self.total - done, 0)
        buf = self.model.ep_info_buffer
        scalars = {
            "ep_rew_mean": float(np.mean([e["r"] for e in buf])) if buf else None,
            "ep_len_mean": float(np.mean([e["l"] for e in buf])) if buf else None,
        }
        for key in ("train/approx_kl", "train/explained_variance"):
            if key in self.model.logger.name_to_value:
                scalars[key.split("/")[1]] = float(self.model.logger.name_to_value[key])
        self.run.write_status(
            state=state,
            timesteps_done=done,
            timesteps_total=self.total,
            sps=sps,
            eta_s=remaining / sps if sps > 0 else None,
            curriculum_level=self.curriculum.level if self.curriculum else None,
            latest_checkpoint=self.latest_checkpoint,
            best_checkpoint=self.best_checkpoint,
            best_metric=self.best_metric,
            last_eval=self.last_eval,
            last_scalars=scalars,
            error=error,
            **self.base,
        )
        self._last = time.time()

    def _on_step(self) -> bool:
        if time.time() - self._last >= self.every_s:
            self.write("running")
        return True

    def _on_rollout_end(self) -> None:
        self.write("running")


class CheckpointCallback(BaseCallback):
    """``step_<012d>`` checkpoints every ``every`` steps (model + VecNormalize, together)."""

    def __init__(
        self,
        run: RunDir,
        every: int,
        keep: int,
        status: StatusCallback,
        curriculum: Curriculum | None = None,
    ) -> None:
        super().__init__()
        self.run, self.every, self.keep, self.status = run, every, keep, status
        self.curriculum = curriculum
        self._next = every
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        self._next = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.num_timesteps >= self._next:
            self.save()
            self._next = (self.model.num_timesteps // self.every + 1) * self.every

    def save(self, name: str | None = None, metric: dict[str, Any] | None = None) -> Path:
        ts = self.model.num_timesteps
        path = self.run.checkpoints / (name or f"step_{ts:012d}")
        save_checkpoint(
            cast("PPO", self.model),
            _vecnorm(self.training_env),
            path,
            {
                "timesteps": ts,
                "wall_time_s": time.time() - self._t0,
                "curriculum_level": self.curriculum.level if self.curriculum else None,
                "curriculum": self.curriculum.state_dict() if self.curriculum else None,
                "metric": metric,
            },
        )
        if name is None:
            self.status.latest_checkpoint = path.name
            prune_checkpoints(self.run.root, self.keep)
        return path


def run_deterministic_eval(
    model: PPO, eval_env: VecEnv, level: int, n_episodes: int, seed: int
) -> dict[str, float]:
    """Deterministic episodes at ``level`` with fixed seeds; every env runs the same number of
    episodes (so short, failing episodes are not over-represented). Returns success rate, mean
    return and the termination fractions."""
    n_envs = eval_env.num_envs
    per_env = -(-n_episodes // n_envs)
    eval_env.env_method("set_curriculum_level", level)
    eval_env.seed(seed)
    obs = eval_env.reset()
    done_count = np.zeros(n_envs, dtype=int)
    succ, rets = [], []
    term: dict[str, int] = defaultdict(int)
    ret = np.zeros(n_envs)
    while (done_count < per_env).any():
        actions, _ = model.predict(obs, deterministic=True)  # type: ignore[arg-type]  # Box obs
        obs, rewards, dones, infos = eval_env.step(actions)
        ret += rewards
        for i, (d, info) in enumerate(zip(dones, infos, strict=True)):
            if not d:
                continue
            if done_count[i] < per_env:
                succ.append(bool(info.get("episode_success", False)))
                rets.append(float(ret[i]))
                term[str(info.get("termination"))] += 1
            done_count[i] += 1
            ret[i] = 0.0
    n = len(succ)
    out = {"success_rate": float(np.mean(succ)), "mean_return": float(np.mean(rets)), "n": n}
    out.update({f"term_{k}": v / n for k, v in term.items()})
    return out


class CurriculumCallback(BaseCallback):
    """Promotion by deterministic mini-evals (ADR-0006 §6).

    Every ``eval_every`` steps: ``eval_episodes`` deterministic episodes at the current level with
    seeds fixed per round on a separate (idle during training) eval env; promote after
    ``consecutive`` successes at the level's threshold and >= ``min_steps`` at the level. Promotion
    sets the level on every training env (``env_method("set_curriculum_level", k)``).
    """

    def __init__(  # noqa: PLR0917
        self,
        curriculum: Curriculum,
        eval_env: VecEnv,
        eval_every: int,
        eval_episodes: int,
        seed: int,
        status: StatusCallback,
        checkpoint: CheckpointCallback,
        stop_level: int | None = None,
    ) -> None:
        super().__init__()
        self.curriculum, self.eval_env = curriculum, eval_env
        self.eval_every, self.eval_episodes, self.seed = eval_every, eval_episodes, seed
        self.status, self.checkpoint, self.stop_level = status, checkpoint, stop_level
        self.round = 0
        self._next = eval_every
        self._stop = False
        self._best: tuple[int, float] = (-1, -1.0)

    def _on_training_start(self) -> None:
        ts = self.model.num_timesteps
        self._next = (ts // self.eval_every + 1) * self.eval_every
        self.training_env.env_method("set_curriculum_level", self.curriculum.level)

    def _on_step(self) -> bool:
        return not self._stop

    def _on_rollout_end(self) -> None:
        ts = self.model.num_timesteps
        if ts < self._next:
            return
        self._next = (ts // self.eval_every + 1) * self.eval_every
        level = self.curriculum.level
        res = run_deterministic_eval(
            cast("PPO", self.model),
            self.eval_env,
            level,
            self.eval_episodes,
            self.seed + 1000 * self.round,
        )
        self.round += 1
        promoted = self.curriculum.update(res["success_rate"], ts)
        rec = self.logger.record
        rec("curriculum/eval_success_rate", res["success_rate"])
        rec("curriculum/eval_mean_return", res["mean_return"])
        self.status.last_eval = {"level": level, "timesteps": ts, **res}
        log.info("mini-eval level %d at %d steps: %s", level, ts, res)
        if (level, res["success_rate"]) > self._best:
            self._best = (level, res["success_rate"])
            self.status.best_metric = {
                "name": "eval_success_rate",
                "value": res["success_rate"],
                "level": level,
            }
            self.checkpoint.save("best", metric=self.status.best_metric)
            self.status.best_checkpoint = "best"
        if promoted:
            new = self.curriculum.level
            log.info("curriculum promoted %d -> %d at %d steps", level, new, ts)
            self.training_env.env_method("set_curriculum_level", new)
            if self.stop_level is not None and new >= self.stop_level:
                self._stop = True
        if self.curriculum.stalled:
            log.warning(
                "curriculum stalled at level %d (max_steps exceeded)", self.curriculum.level
            )


def system_info() -> dict[str, Any]:
    import gymnasium  # noqa: PLC0415 - heavy, only for the run manifest
    import mujoco  # noqa: PLC0415 - heavy, only for the run manifest
    import stable_baselines3  # noqa: PLC0415 - heavy, only for the run manifest
    import torch  # noqa: PLC0415 - heavy, only for the run manifest

    return {
        "torch": str(torch.__version__),
        "sb3": str(stable_baselines3.__version__),
        "gymnasium": str(gymnasium.__version__),
        "mujoco": str(mujoco.__version__),
        "devices": {
            k: v if isinstance(v, (bool, int)) or v is None else str(v)
            for k, v in describe_devices().items()
        },
    }
