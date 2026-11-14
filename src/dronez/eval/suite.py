"""Baseline suite execution: scenarios x seeds -> per-episode metrics, aggregates, G3 verdict.

Phase 3 runs the baselines on the nominal plant with sensor noise (``SimEnv``). Phase 4 generalizes
this to ``run_suite`` over the Gymnasium env and worker processes.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

from dronez.control.base import Controller
from dronez.eval import metrics as M
from dronez.eval.runner import SimEnv, Trajectory, run_episode
from dronez.eval.scenarios import EvalSuite, Scenario, episode_seed, make_plan

SCHEMA_VERSION = 1

#: G3 thresholds (docs/BUILD_PROMPT.md gate G3): recover below 5 deg within 1 s; hover within 5 cm.
G3_RECOVERY_S = 1.0
G3_HOVER_POS_M = 0.05


@dataclass
class ScenarioResult:
    scenario: Scenario
    episodes: list[dict[str, Any]]
    trajectories: list[Trajectory]


def _event_time(sc: Scenario) -> float:
    return {"step_attitude": sc.step_time_s, "impulse": sc.impulse_time_s}.get(sc.kind, 0.0)


def episode_metrics(traj: Trajectory, sc: Scenario) -> dict[str, float | bool]:
    """All metrics of one episode in SI units (rad, s, m). ``nan`` where undefined."""
    dt = traj.dt
    err = M.attitude_error(traj.quat_wxyz, traj.quat_target)
    t_ev = _event_time(sc)
    k_ev = round(t_ev / dt)
    pos_err = M.position_error(traj.pos_w, traj.pos_target)
    out: dict[str, float | bool] = {
        "seed": float(traj.seed),
        "crashed": traj.crashed,
        "time_to_recover_s": M.time_to_recover(err, dt, t_start=t_ev),
        "rms_att_err_rad": M.rms_attitude_error(err, dt, t_start=t_ev),
        "max_att_err_rad": float(np.max(err[k_ev:])),
        "final_att_err_rad": float(err[-1]),
        "saturation_pct": M.saturation_pct(traj.u),
        "mean_abs_delta_u": M.mean_abs_delta(traj.u),
        "hf_energy_frac_gt20hz": M.hf_energy_fraction(traj.u, 1.0 / dt),
        "min_height_m": float(np.min(traj.pos_w[:, 2])),
    }
    if sc.kind in {"recovery", "step_attitude"}:
        q_start = traj.quat_wxyz[0] if sc.kind == "recovery" else traj.quat_target[0]
        out["overshoot_rad"] = M.overshoot(traj.quat_wxyz[k_ev:], q_start, traj.quat_target[-1])
    if sc.kind in {"hover_hold", "impulse"}:
        out["max_pos_err_m"] = float(np.max(pos_err))
        out["rms_pos_err_m"] = float(np.sqrt(np.mean(pos_err**2)))
        out["final_pos_err_m"] = float(pos_err[-1])
    ok = M.success(err, dt, hold_s=sc.hold_s, crashed=traj.crashed)
    if sc.kind == "hover_hold":
        ok = ok and float(out["max_pos_err_m"]) <= G3_HOVER_POS_M
    out["success"] = ok
    return out


def _clean(x: float) -> float | None:
    return None if (isinstance(x, float) and not math.isfinite(x)) else float(x)


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-metric ``mean / std / min / max / p95`` over episodes, plus ``success_rate``."""
    keys = [k for k in episodes[0] if k not in {"seed", "success", "crashed"}]
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


def run_baseline_suite(
    controller: Controller, env: SimEnv, suite: EvalSuite
) -> dict[str, ScenarioResult]:
    results: dict[str, ScenarioResult] = {}
    for si, sc in enumerate(suite.scenarios):
        eps: list[dict[str, Any]] = []
        trajs: list[Trajectory] = []
        for ei in range(sc.n_episodes):
            plan = make_plan(sc, episode_seed(suite.seed, si, ei))
            traj = run_episode(env, controller, plan)
            eps.append(episode_metrics(traj, sc))
            trajs.append(traj)
        results[sc.name] = ScenarioResult(sc, eps, trajs)
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
