"""PPO training (SB3) for the Dronez envs: ``dronez-train --config <yaml>``.

Process model (docs/architecture.md §5.1): env workers are ``SubprocVecEnv`` processes
(``forkserver`` on Linux, ``spawn`` on Windows) with ``OMP/MKL_NUM_THREADS = 1``; each owns one
MjModel. The main process runs PPO on the CPU with ``torch_threads`` threads. Runs go to
``$DRONEZ_DATA_DIR/runs/<run_id>/`` (never the repo).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# worker processes inherit these: set before anything spawns (and before numpy/torch import them)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from dronez.cad.artifacts import load_artifacts  # noqa: E402
from dronez.cad.schema import load_drone_config  # noqa: E402
from dronez.envs import ATTITUDE_LEVELS, HOVER_LEVELS, Curriculum, EnvFactory  # noqa: E402
from dronez.envs.config import EnvConfig  # noqa: E402
from dronez.envs.obs import ObsLayout  # noqa: E402
from dronez.rl.callbacks import (  # noqa: E402
    CheckpointCallback,
    CurriculumCallback,
    InfoLogCallback,
    StatusCallback,
    run_deterministic_eval,
    system_info,
)
from dronez.rl.checkpoint import (  # noqa: E402
    MODEL_FILE,
    latest_checkpoint,
    load_vecnormalize,
    read_meta,
    resolve_checkpoint,
)
from dronez.rl.config import TrainConfig, load_train_config  # noqa: E402
from dronez.rl.normalization import make_vec_normalize  # noqa: E402
from dronez.rl.run_dir import RunDir, default_run_id, git_info, utc_now, validate_run_id  # noqa: E402
from dronez.utils.config import stable_hash  # noqa: E402
from dronez.utils.device import resolve_device  # noqa: E402

log = logging.getLogger("dronez.train")


def _layout(cfg: TrainConfig) -> ObsLayout:
    return ObsLayout.attitude() if cfg.env_id == "DronezAttitude-v0" else ObsLayout.hover()


def _levels(cfg: TrainConfig) -> tuple[Any, ...]:
    if cfg.env.levels is not None:
        return cfg.env.levels
    return ATTITUDE_LEVELS if cfg.env_id == "DronezAttitude-v0" else HOVER_LEVELS


def make_vec_env(
    cfg: TrainConfig, n_envs: int, env_cfg: EnvConfig | None = None, kind: str | None = None
) -> VecEnv:
    """``n_envs`` env processes (``subproc``) or in-process envs (``dummy``). Not seeded here:
    ``PPO(seed=...)`` seeds the ranks through ``env.reset(seed=...)``."""
    ec = env_cfg or cfg.env
    fns = [
        EnvFactory(cfg.env_id, ec, cfg.drone, cfg.sim, rank=i, seed=cfg.run.seed)
        for i in range(n_envs)
    ]
    kind = kind or cfg.run.vec_env
    if kind == "subproc":
        method = "spawn" if sys.platform == "win32" else "forkserver"
        return SubprocVecEnv(fns, start_method=method)  # type: ignore[arg-type]
    return DummyVecEnv(fns)  # type: ignore[arg-type]


def _build_model(cfg: TrainConfig, venv: VecNormalize, run: RunDir, device: str) -> PPO:
    p, pol = cfg.ppo, cfg.policy
    act = torch.nn.Tanh if pol.activation == "tanh" else torch.nn.ReLU
    return PPO(
        "MlpPolicy",
        venv,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        target_kl=p.target_kl,
        policy_kwargs={
            "net_arch": {"pi": list(pol.pi), "vf": list(pol.vf)},
            "activation_fn": act,
            "log_std_init": pol.log_std_init,
            "ortho_init": pol.ortho_init,
        },
        seed=cfg.run.seed,
        device=device,
        verbose=0,
    )


def _setup_logging(run: RunDir) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger("dronez")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(run.log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _write_config(cfg: TrainConfig, run: RunDir, model_hash: str, git: tuple[str, bool]) -> str:
    chash = stable_hash(cfg)
    body = {
        "train_config": cfg.model_dump(mode="json"),
        "config_hash": chash,
        "model_hash": model_hash,
        "seed": cfg.run.seed,
        "git_sha": git[0],
        "git_dirty": git[1],
        "versions": system_info(),
        "created_at": utc_now(),
    }
    run.config_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return chash


def train(
    cfg: TrainConfig,
    run_id: str | None = None,
    resume: bool = False,
    config_stem: str = "train",
) -> Path:
    """Run one training stage. Returns the run directory."""
    run_id = validate_run_id(run_id or default_run_id(config_stem))
    run = RunDir(run_id)
    _setup_logging(run)
    device = resolve_device(cfg.run.device)
    torch.set_num_threads(cfg.run.torch_threads)
    git = git_info()
    art = load_artifacts(
        load_drone_config(cfg.drone)
    )  # fail early (and once) if the cache is missing
    model_hash = art.mass_props.model_hash
    chash = _write_config(cfg, run, model_hash, git) if not resume else stable_hash(cfg)
    layout = _layout(cfg)
    base_status: dict[str, Any] = {
        "started_at": run.read_status().get("started_at", utc_now()),
        "n_envs": cfg.run.n_envs,
        "device": device,
        "seed": cfg.run.seed,
        "git_sha": git[0],
        "git_dirty": git[1],
        "config_hash": chash,
        "model_hash": model_hash,
        "stage": cfg.run.stage,
    }
    run.write_status(
        state="starting", error=None, timesteps_total=cfg.run.total_timesteps, **base_status
    )
    log.info(
        "run %s (%s), n_envs %d, device %s, seed %d",
        run_id,
        cfg.run.stage,
        cfg.run.n_envs,
        device,
        cfg.run.seed,
    )

    venv: VecEnv | None = None
    eval_venv: VecEnv | None = None
    status: StatusCallback | None = None
    model: PPO | None = None
    try:
        raw = make_vec_env(cfg, cfg.run.n_envs)
        mon = VecMonitor(raw)
        resume_dir = latest_checkpoint(run.root) if resume else None
        if resume and resume_dir is None:
            raise FileNotFoundError(f"--resume: no checkpoint in {run.root}")
        n = cfg.normalize
        if resume_dir is not None:
            vn = load_vecnormalize(resume_dir, mon)
            vn.training = True
        else:
            vn = make_vec_normalize(
                mon,
                layout,
                gamma=n.gamma,
                clip_obs=n.clip_obs,
                clip_reward=n.clip_reward,
                norm_reward=n.norm_reward,
            )
        venv = vn

        if resume_dir is not None:
            meta = read_meta(resume_dir)
            model = PPO.load(str(resume_dir / MODEL_FILE), env=vn, device=device)
            log.info("resumed from %s (%d steps)", resume_dir.name, model.num_timesteps)
        else:
            meta = {}
            model = _build_model(cfg, vn, run, device)
            if cfg.init_from is not None:
                src = resolve_checkpoint(cfg.init_from.run_id, cfg.init_from.checkpoint)
                model.set_parameters(str(src / MODEL_FILE), exact_match=True, device=device)
                vn.ret_rms = load_vecnormalize(src).ret_rms
                meta = read_meta(src)
                log.info("initialized from %s", src)
        model.set_logger(configure(str(run.tb), ["tensorboard"]))

        # curriculum
        cur: Curriculum | None = None
        cc = cfg.curriculum
        levels = _levels(cfg)
        cur = Curriculum(levels, start_level=cfg.env.start_level, consecutive=cc.consecutive)
        if resume_dir is not None and meta.get("curriculum"):
            cur.load_state_dict(meta["curriculum"])
        # a hand-off keeps its own start level (from the config) but not the counters
        done = model.num_timesteps
        remaining = cfg.run.total_timesteps - done
        if remaining <= 0:
            log.info("nothing to do: %d >= %d steps", done, cfg.run.total_timesteps)
        status = StatusCallback(run, cfg.run.total_timesteps, base_status, cfg.status_every_s, cur)
        ckpt = CheckpointCallback(run, cfg.checkpoint_every, cfg.keep_checkpoints, status, cur)
        callbacks: list[Any] = [InfoLogCallback(cur, n.clip_obs), status, ckpt]
        if cc.enabled:
            eval_cfg = cfg.env.model_copy(update={"info_level": "minimal"})
            eval_raw = make_vec_env(cfg, cfg.run.n_envs, eval_cfg)
            eval_venv = make_vec_normalize(
                VecMonitor(eval_raw), layout, gamma=n.gamma, clip_obs=n.clip_obs,
                clip_reward=n.clip_reward, norm_reward=False, training=False,
            )  # fmt: skip
            callbacks.append(
                CurriculumCallback(
                    cur,
                    eval_venv,
                    cc.eval_every,
                    cc.eval_episodes,
                    cfg.run.seed + 7,
                    status,
                    ckpt,
                    cc.stop_level,
                )
            )
        else:
            # keep ``set_curriculum_level`` consistent with the config even without promotion
            venv.env_method("set_curriculum_level", cur.level)

        def _sigterm(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _sigterm)
        t0 = time.time()
        interrupted = False
        try:
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    callback=CallbackList(callbacks),
                    reset_num_timesteps=resume_dir is None,
                    progress_bar=False,
                )
        except KeyboardInterrupt:
            interrupted = True
            log.warning("interrupted: saving a checkpoint")

        ckpt.save()
        if cfg.final_eval_episodes > 0 and not interrupted:
            if eval_venv is None:
                eval_cfg = cfg.env.model_copy(update={"info_level": "minimal"})
                eval_raw = make_vec_env(
                    cfg, min(cfg.run.n_envs, cfg.final_eval_episodes), eval_cfg, "dummy"
                )
                eval_venv = make_vec_normalize(
                    VecMonitor(eval_raw), layout, gamma=n.gamma, clip_obs=n.clip_obs,
                    clip_reward=n.clip_reward, norm_reward=False, training=False,
                )  # fmt: skip
            res = run_deterministic_eval(
                model, eval_venv, cur.level, cfg.final_eval_episodes, cfg.run.seed + 99
            )
            status.last_eval = {
                "level": cur.level,
                "timesteps": model.num_timesteps,
                "final": True,
                **res,
            }
            log.info("final mini-eval: %s", res)
        final = "stopped" if interrupted else "finished"
        status.write(final)
        log.info("%s after %.1f s, %d steps", final, time.time() - t0, model.num_timesteps)
        return run.root
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            log.exception("training failed")
        if status is not None and model is not None:
            status.write("failed", error=f"{type(exc).__name__}: {exc}")
        else:
            run.write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for v in (venv, eval_venv):
            if v is not None:
                try:
                    v.close()
                except Exception:  # pragma: no cover
                    log.debug("close failed", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dronez-train", description="Train a Dronez PPO policy.")
    parser.add_argument("--config", required=True, help="name in configs/train/ or a YAML path")
    parser.add_argument("--run-id", help="[a-z0-9-]+; default <config>-<UTC timestamp>")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--device", choices=["cpu", "xpu"], help="override run.device")
    parser.add_argument("--total-timesteps", type=int, help="override run.total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override run.n_envs")
    parser.add_argument("--seed", type=int, help="override run.seed")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    over: dict[str, Any] = {}
    if args.device:
        over["device"] = args.device
    if args.total_timesteps:
        over["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        over["n_envs"] = args.n_envs
    if args.seed is not None:
        over["seed"] = args.seed
    if over:
        cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update=over)})
    if args.resume and not args.run_id:
        parser.error("--resume needs --run-id")
    stem = Path(args.config).stem
    out = train(cfg, run_id=args.run_id, resume=args.resume, config_stem=stem)
    print(f"run dir: {out}")
    print(f"run id : {out.name}")
    np.set_printoptions()
    return 0
