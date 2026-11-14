"""Throughput benchmark for Dronez training (ADR-0006 §8, architecture §5.4).

Measures, on this machine:

1. ``raw``        single env in-process, random actions (env cost only);
2. ``collect``    ``SubprocVecEnv`` x n, random actions (env + IPC), for each ``--n-envs``;
3. ``ppo``        effective SPS of PPO with the ADR-0006 hyperparameters (``n_steps = 16384 // n``,
                  batch 4096, 5 epochs, pi [64, 64] / vf [256, 256]): collection + update wall time
                  over ``--rollouts`` rollouts, for each n and each ``--torch-threads``;
4. ``xpu``        the same PPO measurement on ``device="xpu"`` if ``torch.xpu.is_available()``,
                  otherwise recorded as skipped (BUILD_PROMPT: CPU-vs-XPU benchmark for the ADR).

Recommendation: the largest n whose marginal effective-SPS gain over the previous n is >= 10 %
(ADR-0006 §8), and the faster torch thread count. Writes
``$DRONEZ_DATA_DIR/benchmarks/sps-<utc>.json`` and prints a table.

    uv run python scripts/benchmark_sps.py                     # defaults: n = 1 2 3 4
    uv run python scripts/benchmark_sps.py --n-envs 2 4 --rollouts 1 --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import dronez.envs  # noqa: E402, F401 - registers the env ids
from dronez.envs import EnvConfig, ObsLayout  # noqa: E402
from dronez.rl.config import TrainConfig  # noqa: E402
from dronez.rl.normalization import make_vec_normalize  # noqa: E402
from dronez.rl.train import make_vec_env  # noqa: E402
from dronez.utils import paths  # noqa: E402
from dronez.utils.device import xpu_available  # noqa: E402

ROLLOUT_STEPS = 16384  # ADR-0006 §8: n_steps = 16384 // n_envs
MARGINAL_GAIN = 0.10


def _env_cfg() -> EnvConfig:
    return EnvConfig(info_level="minimal")


def bench_raw(steps: int, seed: int) -> dict[str, Any]:
    env = gym.make("DronezAttitude-v0", env_cfg=_env_cfg())
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    acts = rng.uniform(-0.3, 0.3, size=(steps, 4)).astype(np.float32)
    resets = 0
    t0 = time.perf_counter()
    for a in acts:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
            resets += 1
    dt = time.perf_counter() - t0
    env.close()
    return {"steps": steps, "wall_s": dt, "sps": steps / dt, "resets": resets}


def bench_collect(n_envs: int, steps_per_env: int, seed: int) -> dict[str, Any]:
    cfg = TrainConfig(env=_env_cfg())
    venv = make_vec_env(cfg, n_envs, kind="subproc")
    venv.seed(seed)
    venv.reset()
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-0.3, 0.3, size=(steps_per_env, n_envs, 4)).astype(np.float32)
    for a in acts[:50]:  # warm-up: workers fully started
        venv.step(a)
    t0 = time.perf_counter()
    for a in acts:
        venv.step(a)
    dt = time.perf_counter() - t0
    venv.close()
    total = steps_per_env * n_envs
    return {"n_envs": n_envs, "steps": total, "wall_s": dt, "sps": total / dt}


def _ppo(cfg: TrainConfig, venv: Any, n_envs: int, device: str) -> PPO:
    p, pol = cfg.ppo, cfg.policy
    return PPO(
        "MlpPolicy",
        venv,
        learning_rate=p.learning_rate,
        n_steps=ROLLOUT_STEPS // n_envs,
        batch_size=4096,
        n_epochs=5,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        target_kl=p.target_kl,
        policy_kwargs={
            "net_arch": {"pi": list(pol.pi), "vf": list(pol.vf)},
            "activation_fn": torch.nn.Tanh,
            "log_std_init": pol.log_std_init,
            "ortho_init": pol.ortho_init,
        },
        seed=cfg.run.seed,
        device=device,
        verbose=0,
    )


def bench_ppo(n_envs: int, threads: int, rollouts: int, device: str = "cpu") -> dict[str, Any]:
    """Effective SPS = timesteps / wall time of ``learn`` (collection + GAE + updates)."""
    torch.set_num_threads(threads)
    cfg = TrainConfig(env=_env_cfg())
    venv = make_vec_normalize(make_vec_env(cfg, n_envs, kind="subproc"), ObsLayout.attitude())
    model = _ppo(cfg, venv, n_envs, device)
    model.learn(total_timesteps=ROLLOUT_STEPS // n_envs * n_envs)  # warm-up rollout (not timed)
    total = ROLLOUT_STEPS // n_envs * n_envs * rollouts
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    venv.close()
    return {
        "n_envs": n_envs,
        "torch_threads": threads,
        "device": device,
        "steps": total,
        "wall_s": dt,
        "sps": total / dt,
    }


def recommend(ppo_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Best thread count per n, then the largest n with >= 10 % marginal gain over the previous."""
    best: dict[int, dict[str, Any]] = {}
    for r in ppo_rows:
        if r["device"] != "cpu":
            continue
        if r["n_envs"] not in best or r["sps"] > best[r["n_envs"]]["sps"]:
            best[r["n_envs"]] = r
    ns = sorted(best)
    if not ns:
        return {}
    chosen = ns[0]
    for prev, n in itertools.pairwise(ns):  # no early stop: a flat step (e.g. odd n) can recover
        if best[n]["sps"] >= (1.0 + MARGINAL_GAIN) * best[prev]["sps"]:
            chosen = n
    r = best[chosen]
    return {
        "n_envs": chosen,
        "torch_threads": r["torch_threads"],
        "effective_sps": r["sps"],
        "rule": f"largest n with >= {MARGINAL_GAIN:.0%} marginal gain (ADR-0006 §8)",
        "eta_4p5h_steps": int(4.5 * 3600 * r["sps"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-envs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--torch-threads", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--rollouts", type=int, default=2, help="timed PPO rollouts per config")
    ap.add_argument("--raw-steps", type=int, default=5000)
    ap.add_argument("--collect-steps", type=int, default=2000, help="steps per env")
    ap.add_argument("--quick", action="store_true", help="fewer steps (sanity check only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--recompute", type=Path, default=None, help="re-derive the recommendation of a result JSON"
    )
    args = ap.parse_args(argv)
    if args.recompute is not None:
        stored = json.loads(args.recompute.read_text(encoding="utf-8"))
        stored["recommendation"] = recommend(stored["ppo"])
        args.recompute.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        print(f"recommendation: {stored['recommendation']}")
        return 0
    if args.quick:
        args.raw_steps, args.collect_steps, args.rollouts = 1000, 300, 1

    res: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "processor": platform.processor(),
        },
        "env": "DronezAttitude-v0",
        "env_config": _env_cfg().model_dump(mode="json", include={"info_level"}),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    print(f"raw single env ({args.raw_steps} steps) ...", flush=True)
    res["raw"] = bench_raw(args.raw_steps, args.seed)
    print(f"  {res['raw']['sps']:.0f} SPS", flush=True)

    res["collect"] = []
    for n in args.n_envs:
        r = bench_collect(n, args.collect_steps, args.seed)
        res["collect"].append(r)
        print(f"collect subproc n={n}: {r['sps']:.0f} SPS", flush=True)

    res["ppo"] = []
    for n in args.n_envs:
        for th in args.torch_threads:
            r = bench_ppo(n, th, args.rollouts)
            res["ppo"].append(r)
            print(f"ppo n={n} threads={th}: {r['sps']:.0f} effective SPS", flush=True)

    if xpu_available():
        n = max(args.n_envs)
        r = bench_ppo(n, 1, args.rollouts, device="xpu")
        res["xpu"] = r
        print(f"ppo xpu n={n}: {r['sps']:.0f} effective SPS", flush=True)
    else:
        res["xpu"] = {"skipped": True, "reason": "torch.xpu.is_available() is False"}
        print("xpu: skipped (torch.xpu.is_available() is False)", flush=True)

    res["recommendation"] = recommend(res["ppo"])
    out = args.out or paths.data_dir() / "benchmarks" / (
        "sps-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"recommendation: {res['recommendation']}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
