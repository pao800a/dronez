"""Suite execution: scenarios x seeds -> per-episode metrics, aggregates, parallel workers.

``run_episode`` drives the Gymnasium env (docs/architecture.md §7). ``run_suite`` fans the
episodes out to worker processes (one env each, fresh controller per process);
``run_baseline_suite`` is the in-process variant over an existing env (tests, small suites). Gates
live in ``gates.py``.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from dronez.control.base import Controller
from dronez.envs.config import EnvConfig
from dronez.eval import metrics as M
from dronez.eval.controllers import ControllerSpec
from dronez.eval.runner import Trajectory, run_episode
from dronez.eval.scenarios import EvalSuite, Scenario, episode_seed, make_plan
from dronez.utils import rotations

SCHEMA_VERSION = 2

#: G3 thresholds (docs/BUILD_PROMPT.md gate G3): recover below 5 deg within 1 s; hover within 5 cm.
G3_RECOVERY_S = 1.0
G3_HOVER_POS_M = 0.05
#: statistics window for the smoothness / tilt metrics (ADR-0006 §9)
T_START_S = 1.0


@dataclass
class ScenarioResult:
    scenario: Scenario
    episodes: list[dict[str, Any]]
    trajectories: list[Trajectory]


def _event_time(sc: Scenario) -> float:
    return {"step_attitude": sc.step_time_s, "impulse": sc.impulse_time_s}.get(sc.kind, 0.0)


def episode_metrics(traj: Trajectory, sc: Scenario) -> dict[str, float | bool | str | None]:
    """All metrics of one episode in SI units (rad, s, m). ``nan`` where undefined."""
    dt = traj.dt
    rate = 1.0 / dt
    err = M.attitude_error(traj.quat_wxyz, traj.quat_target)
    t_ev = _event_time(sc)
    k_ev = round(t_ev / dt)
    pos_err = M.position_error(traj.pos_w, traj.pos_target)
    k1 = round(T_START_S / dt)
    tilt = np.asarray(
        [float(rotations.tilt_angle(q)) for q in traj.quat_wxyz[k1:]], dtype=np.float64
    )
    crashed = traj.crashed
    out: dict[str, float | bool | str | None] = {
        "seed": float(traj.seed),
        "crashed": crashed,
        "termination": traj.termination,
        "ground_contact": traj.termination == "ground_contact",
        "time_to_recover_s": M.time_to_recover(err, dt, t_start=t_ev),
        "rms_att_err_rad": M.rms_attitude_error(err, dt, t_start=t_ev),
        "max_att_err_rad": float(np.max(err[min(k_ev, len(err) - 1) :])),
        "final_att_err_rad": float(err[-1]),
        "saturation_pct": M.saturation_pct(traj.a, -1.0, 1.0),
        "mean_abs_delta_u": M.mean_abs_delta(traj.u, rate, T_START_S),
        "hf_rms_u": M.hf_rms(traj.u, rate, t_start=T_START_S),
        "hf_energy_frac_gt20hz": M.hf_energy_fraction(traj.u[k1:], rate),
        "min_height_m": float(np.min(traj.pos_w[:, 2])),
        "tilt_sq_sum": float(np.sum(tilt**2)),
        "tilt_n": float(tilt.size),
    }
    if sc.t_max_s is not None:
        out["success_recovery"] = M.success_recovery(err, dt, sc.t_max_s, crashed)
    if sc.kind in {"recovery", "step_attitude"}:
        q_start = traj.quat_wxyz[0] if sc.kind == "recovery" else traj.quat_target[0]
        out["overshoot_rad"] = M.overshoot(traj.quat_wxyz[k_ev:], q_start, traj.quat_target[-1])
    if sc.kind in {"hover_hold", "impulse"}:
        out["max_pos_err_m"] = float(np.max(pos_err))
        out["rms_pos_err_m"] = float(np.sqrt(np.mean(pos_err**2)))
        out["final_pos_err_m"] = float(pos_err[-1])
    ok = M.success(err, dt, hold_s=sc.hold_s, crashed=crashed)
    if sc.kind == "hover_hold" and not sc.random_yaw:
        ok = ok and float(out["max_pos_err_m"]) <= G3_HOVER_POS_M  # type: ignore[arg-type]
    out["success"] = ok
    return out


def _clean(x: float) -> float | None:
    return None if (isinstance(x, float) and not math.isfinite(x)) else float(x)


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-metric ``mean / std / min / max / p95`` over episodes, plus ``success_rate``."""
    keys = [
        k
        for k, v in episodes[0].items()
        if k not in {"seed", "success", "crashed"} and isinstance(v, (int, float, np.floating))
    ]
    agg: dict[str, Any] = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "crash_rate": float(np.mean([e["crashed"] for e in episodes])),
    }
    for k in keys:
        v = np.array([e[k] for e in episodes], dtype=np.float64)
        finite = v[np.isfinite(v)]
        agg[k] = {
            "mean": _clean(float(np.mean(finite))) if finite.size else None,
            "std": _clean(float(np.std(finite))) if finite.size else None,
            "min": _clean(float(np.min(finite))) if finite.size else None,
            "max": _clean(float(np.max(finite))) if finite.size else None,
            "p95": _clean(float(np.percentile(finite, 95))) if finite.size else None,
            "n_nan": int(v.size - finite.size),
        }
    return agg


def g3_gate(results: dict[str, ScenarioResult]) -> dict[str, Any]:
    """Gate G3: recovery from 30 deg within 1 s for every seed; hover within 5 cm for every seed."""
    criteria: list[dict[str, Any]] = []
    rec = results.get("recovery_30deg")
    if rec is not None:
        ttr = np.array([e["time_to_recover_s"] for e in rec.episodes], dtype=np.float64)
        worst = float(np.nanmax(ttr)) if np.any(np.isfinite(ttr)) else float("inf")
        ok = bool(np.all(np.isfinite(ttr)) and worst <= G3_RECOVERY_S)
        criteria.append(
            {
                "name": "recovery_30deg: time to < 5 deg, worst seed [s]",
                "value": _clean(worst),
                "threshold": G3_RECOVERY_S,
                "pass": ok,
            }
        )
    hov = results.get("hover_hold")
    if hov is not None:
        worst = float(max(e["max_pos_err_m"] for e in hov.episodes))
        criteria.append(
            {
                "name": "hover_hold: max position error over 10 s, worst seed [m]",
                "value": worst,
                "threshold": G3_HOVER_POS_M,
                "pass": worst <= G3_HOVER_POS_M,
            }
        )
    return {"pass": bool(criteria) and all(c["pass"] for c in criteria), "criteria": criteria}


# -- in-process ---------------------------------------------------------------------------------
def run_baseline_suite(
    controller: Controller, env: Any, suite: EvalSuite, keep_traj: int | None = None
) -> dict[str, ScenarioResult]:
    """Run every scenario sequentially on ``env`` (a ``DronezBaseEnv`` with ``info_level=full``)."""
    results: dict[str, ScenarioResult] = {}
    for si, sc in enumerate(suite.scenarios):
        eps: list[dict[str, Any]] = []
        trajs: list[Trajectory] = []
        for ei in range(sc.n_episodes):
            plan = make_plan(sc, episode_seed(suite.seed, si, ei))
            traj = run_episode(env, controller, plan)
            eps.append(episode_metrics(traj, sc))
            if keep_traj is None or len(trajs) < keep_traj:
                trajs.append(traj)
        results[sc.name] = ScenarioResult(sc, eps, trajs)
    return results


# -- parallel -----------------------------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _worker_init() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def _worker_task(
    args: tuple[ControllerSpec, EnvConfig, str, str, Scenario, int, int, list[int], int],
) -> tuple[int, list[tuple[int, dict[str, Any], Trajectory | None]]]:
    spec, env_cfg, drone, sim, sc, suite_seed, si, eps, keep = args
    import gymnasium as gym  # noqa: PLC0415

    import dronez.envs  # noqa: F401, PLC0415

    key = f"{env_cfg.hash()}|{drone}|{sim}"
    if _WORKER.get("key") != key:
        _WORKER["env"] = gym.make("DronezAttitude-v0", env_cfg=env_cfg, sim_cfg=sim, drone=drone)
        _WORKER["key"] = key
    if _WORKER.get("spec") != spec:
        _WORKER["ctrl"] = spec()
        _WORKER["spec"] = spec
    env, ctrl = _WORKER["env"], _WORKER["ctrl"]
    out = []
    for ei in eps:
        traj = run_episode(env, ctrl, make_plan(sc, episode_seed(suite_seed, si, ei)))
        out.append((ei, episode_metrics(traj, sc), traj if ei < keep else None))
    return si, out


def run_suite(
    make_controller: ControllerSpec,
    suite: EvalSuite,
    env_cfg: EnvConfig | None = None,
    *,
    drone: str = "default_5in",
    sim: str = "default",
    n_workers: int = 1,
    keep_traj: int = 4,
    only: set[str] | None = None,
) -> dict[str, ScenarioResult]:
    """Run ``suite`` with ``n_workers`` processes. ``make_controller`` is a picklable
    ``ControllerSpec``; episodes are seeded by ``SeedSequence([suite.seed, scenario, episode])`` so
    every controller sees the same episodes. ``only`` restricts the scenario names."""
    cfg = (env_cfg or EnvConfig()).model_copy(update={"info_level": "full"})
    n_workers = max(1, n_workers)
    tasks = []
    for si, sc in enumerate(suite.scenarios):
        if only is not None and sc.name not in only:
            continue
        chunk = max(1, math.ceil(sc.n_episodes / (n_workers * 2)))
        idx = list(range(sc.n_episodes))
        for k in range(0, sc.n_episodes, chunk):
            tasks.append(
                (
                    make_controller,
                    cfg,
                    drone,
                    sim,
                    sc,
                    suite.seed,
                    si,
                    idx[k : k + chunk],
                    keep_traj,
                )
            )
    collected: dict[int, list[tuple[int, dict[str, Any], Trajectory | None]]] = {}
    if n_workers == 1:
        _worker_init()
        for t in tasks:
            si, rows = _worker_task(t)
            collected.setdefault(si, []).extend(rows)
    else:
        method = "spawn" if sys.platform == "win32" else "forkserver"
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context(method), initializer=_worker_init
        ) as pool:
            for si, rows in pool.map(_worker_task, tasks):
                collected.setdefault(si, []).extend(rows)
    results: dict[str, ScenarioResult] = {}
    for si, sc in enumerate(suite.scenarios):
        if si not in collected:
            continue
        rows = sorted(collected[si], key=lambda r: r[0])
        results[sc.name] = ScenarioResult(
            sc, [r[1] for r in rows], [r[2] for r in rows if r[2] is not None]
        )
    return results


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
