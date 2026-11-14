"""Tool logic of the ``dronez-sim`` MCP server as plain functions (no MCP transport needed).

Output rule (docs/architecture.md section 10): every function returns **one JSON-serializable dict**
of at most ``MAX_BYTES`` bytes; arrays are capped at ``MAX_ARRAY`` elements (``"truncated": true``
on the owning object); numbers read from disk carry a ``"source"`` path; errors are
``{"error": "..."}`` (never tracebacks). Side effects: none except bounded in-process simulation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

MAX_BYTES = 4096
MAX_ARRAY = 32
MAX_ROLLOUT_S = 10.0
_ERR_MAX = 300

_STATUS_KEYS = (
    "state",
    "error",
    "stage",
    "timesteps_total",
    "timesteps_done",
    "sps",
    "eta_s",
    "n_envs",
    "device",
    "seed",
    "curriculum_level",
    "latest_checkpoint",
    "best_checkpoint",
    "best_metric",
    "started_at",
    "updated_at",
    "git_sha",
    "model_hash",
    "last_eval",
    "last_scalars",
)


class ToolError(Exception):
    """A user-facing error; its message goes into ``{"error": ...}``."""


def _clean(obj: Any, cap: int) -> Any:
    """JSON-safe copy: non-finite floats -> None, lists capped at ``cap`` (parent gets
    ``"truncated": true``), numpy scalars/arrays and paths converted."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        trunc = False
        for k, v in obj.items():
            cv = _clean(v, cap)
            if isinstance(v, (list, tuple)) and len(v) > cap:
                trunc = True
            out[str(k)] = cv
        if trunc:
            out["truncated"] = True
        return out
    if isinstance(obj, (list, tuple)):
        return [_clean(v, cap) for v in list(obj)[:cap]]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return _clean(obj.tolist(), cap)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return str(obj)


def _top_level_list_truncated(raw: Any, cap: int) -> bool:
    return isinstance(raw, (list, tuple)) and len(raw) > cap


def finalize(obj: Any) -> dict[str, Any]:
    """Apply the output rule; shrinks array caps until the JSON fits in ``MAX_BYTES``."""
    cap = MAX_ARRAY
    while True:
        out = _clean(obj, cap)
        if not isinstance(out, dict):
            out = {"result": out}
        s = json.dumps(out, separators=(",", ":"))
        if len(s.encode()) <= MAX_BYTES:
            return out
        if cap == 1:
            return {"error": f"output exceeds {MAX_BYTES} bytes even when truncated"}
        cap //= 2


def tool(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a tool function: output rule applied, exceptions become ``{"error": ...}``."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return finalize(fn(*args, **kwargs))
        except ToolError as exc:
            return finalize({"error": str(exc)[:_ERR_MAX]})
        except Exception as exc:
            return finalize({"error": f"{type(exc).__name__}: {exc}"[:_ERR_MAX]})

    return wrapper


# ---------------------------------------------------------------- paths / io


def _data_dir() -> Path:
    from dronez.utils import paths  # noqa: PLC0415

    return Path(paths.data_dir(create=False))


def _runs_root() -> Path:
    return _data_dir() / "runs"


def _run_root(run_id: str) -> Path:
    from dronez.rl.run_dir import validate_run_id  # noqa: PLC0415

    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    root = _runs_root() / run_id
    if not root.is_dir():
        raise ToolError(f"run {run_id!r} not found in {_runs_root()}")
    return root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ToolError(f"{path} not found") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"{path} is not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ToolError(f"{path} is not a JSON object")
    return data


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- tools


@tool
def list_runs(limit: int = 10) -> dict[str, Any]:
    """Most recently updated runs: run_id, state, timesteps_done, updated_at."""
    limit = max(1, min(int(limit), MAX_ARRAY))
    root = _runs_root()
    rows: list[tuple[float, dict[str, Any]]] = []
    if root.is_dir():
        for d in root.iterdir():
            sp = d / "status.json"
            if not d.is_dir() or not sp.is_file():
                continue
            try:
                st = json.loads(sp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                st = {"state": "unreadable"}
            rows.append(
                (
                    sp.stat().st_mtime,
                    {
                        "run_id": d.name,
                        "state": st.get("state"),
                        "timesteps_done": st.get("timesteps_done"),
                        "updated_at": st.get("updated_at") or _utc(sp.stat().st_mtime),
                    },
                )
            )
    rows.sort(key=lambda r: r[0], reverse=True)
    return {
        "runs": [r for _, r in rows[:limit]],
        "n_total": len(rows),
        "truncated": len(rows) > limit,
        "source": str(root),
    }


@tool
def training_status(run_id: str) -> dict[str, Any]:
    """The ``status.json`` subset of a run plus ETA (``eta_s`` as written by the trainer)."""
    path = _run_root(run_id) / "status.json"
    st = _read_json(path)
    out: dict[str, Any] = {k: st[k] for k in _STATUS_KEYS if k in st}
    out["run_id"] = run_id
    done, total = st.get("timesteps_done"), st.get("timesteps_total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        out["progress_frac"] = done / total
    out["source"] = str(path)
    return out


def _tb_event_dir(tb: Path) -> Path | None:
    """The newest directory under ``tb`` that holds event files (SB3 may nest)."""
    cands = [p.parent for p in tb.rglob("events.out.tfevents.*")]
    if not cands:
        return None
    return max(set(cands), key=lambda d: max(f.stat().st_mtime for f in d.glob("events.out*")))


def _downsample(n: int, points: int) -> list[int]:
    if n <= points:
        return list(range(n))
    return sorted({round(i * (n - 1) / (points - 1)) for i in range(points)})


@tool
def tb_scalars(run_id: str, tags: list[str] | None = None, points: int = 20) -> dict[str, Any]:
    """Per tag: last value, min, max and up to ``points`` downsampled (step, value) points.

    With no ``tags`` the available scalar tags are listed instead.
    """
    from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
        EventAccumulator,
    )

    tb = _run_root(run_id) / "tb"
    ev_dir = _tb_event_dir(tb) if tb.is_dir() else None
    if ev_dir is None:
        raise ToolError(f"no TensorBoard events under {tb}")
    points = max(2, min(int(points), MAX_ARRAY))
    acc = EventAccumulator(str(ev_dir), size_guidance={"scalars": 0})
    acc.Reload()
    avail = sorted(acc.Tags().get("scalars", []))
    if not tags:
        return {"available_tags": avail, "n_tags": len(avail), "source": str(ev_dir)}
    out: dict[str, Any] = {}
    for tag in tags[:MAX_ARRAY]:
        if tag not in avail:
            out[tag] = {"error": "unknown tag"}
            continue
        ev = acc.Scalars(tag)
        vals = [float(e.value) for e in ev]
        idx = _downsample(len(ev), points)
        finite = [v for v in vals if math.isfinite(v)]
        out[tag] = {
            "last": vals[-1],
            "last_step": int(ev[-1].step),
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "n": len(ev),
            "steps": [int(ev[i].step) for i in idx],
            "values": [vals[i] for i in idx],
        }
    return {"run_id": run_id, "scalars": out, "source": str(ev_dir)}


@tool
def list_checkpoints(run_id: str) -> dict[str, Any]:
    """Checkpoint directories (``step_*`` and ``best``) with their ``ckpt.json`` contents."""
    from dronez.rl.checkpoint import META_FILE, MODEL_FILE  # noqa: PLC0415

    ck = _run_root(run_id) / "checkpoints"
    dirs = (
        sorted(p for p in ck.iterdir() if p.is_dir() and not p.name.startswith("."))
        if ck.is_dir()
        else []
    )
    dirs.sort(key=lambda p: (p.name == "best", p.name))
    rows = []
    for d in dirs[-MAX_ARRAY:]:
        meta: dict[str, Any] = {}
        mf = d / META_FILE
        if mf.is_file():
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {"error": "unreadable ckpt.json"}
        rows.append(
            {
                "name": d.name,
                "complete": (d / MODEL_FILE).is_file(),
                "meta": meta,
                "source": str(mf),
            }
        )
    return {
        "run_id": run_id,
        "checkpoints": rows,
        "n_total": len(dirs),
        "truncated": len(dirs) > MAX_ARRAY,
    }


def _metrics_path(run_id: str, checkpoint: str) -> Path:
    """``eval/<run_id>/<checkpoint>/metrics.json``; ``latest`` = newest metrics.json there."""
    if "/" in run_id or "\\" in run_id or ".." in run_id or "/" in checkpoint or ".." in checkpoint:
        raise ToolError("run_id/checkpoint must not contain path separators")
    base = _data_dir() / "eval" / run_id
    if not base.is_dir():
        raise ToolError(f"no eval output for {run_id!r} in {base.parent}")
    if checkpoint == "latest":
        found = sorted(base.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise ToolError(f"no metrics.json under {base}")
        return found[-1]
    p = base / checkpoint / "metrics.json"
    if not p.is_file():
        have = sorted(d.name for d in base.iterdir() if (d / "metrics.json").is_file())
        raise ToolError(f"{p} not found; available checkpoints: {have[:8]}")
    return p


def _num(agg: dict[str, Any], key: str, stat: str) -> Any:
    v = agg.get(key)
    return v.get(stat) if isinstance(v, dict) else v


@tool
def eval_summary(run_id: str, checkpoint: str = "latest") -> dict[str, Any]:
    """Headline per-scenario metrics plus gate verdicts from an eval ``metrics.json``."""
    path = _metrics_path(run_id, checkpoint)
    m = _read_json(path)
    scen: dict[str, Any] = {}
    for name, s in list(m.get("scenarios", {}).items())[:MAX_ARRAY]:
        a = s.get("aggregate", {})
        scen[name] = {
            "n_episodes": a.get("n_episodes"),
            "success_rate": a.get("success_rate"),
            "crash_rate": a.get("crash_rate"),
            "rms_att_err_rad_mean": _num(a, "rms_att_err_rad", "mean"),
            "max_att_err_rad_max": _num(a, "max_att_err_rad", "max"),
            "time_to_recover_s_p95": _num(a, "time_to_recover_s", "p95"),
            "max_pos_err_m_max": _num(a, "max_pos_err_m", "max"),
        }
    gates = {g: v.get("pass") for g, v in m.get("gates", {}).items() if isinstance(v, dict)}
    return {
        "run_id": m.get("run_id", run_id),
        "checkpoint": m.get("checkpoint", checkpoint),
        "controller": m.get("controller"),
        "suite_hash": m.get("suite_hash"),
        "scenarios": scen,
        "gates_pass": gates,
        "source": str(path),
    }


def _latest_baseline_metrics(controller: str = "pid") -> dict[str, Any] | None:
    base = _data_dir() / "eval" / f"baseline-{controller}"
    found = (
        sorted(base.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
        if base.is_dir()
        else []
    )
    return json.loads(found[-1].read_text(encoding="utf-8")) if found else None


@tool
def gate_check(run_id: str, checkpoint: str, gate: str) -> dict[str, Any]:
    """``{gate, pass, criteria, source}``: stored verdict from ``metrics.json`` if present, else
    evaluated with ``dronez.eval.gates.check_gate`` (G4-G6; G6 uses the latest PID baseline)."""
    g = gate.upper()
    path = _metrics_path(run_id, checkpoint)
    m = _read_json(path)
    stored = m.get("gates", {}).get(g)
    if isinstance(stored, dict) and "criteria" in stored:
        res = {"pass": bool(stored.get("pass")), "criteria": stored["criteria"], "origin": "stored"}
    elif g in {"G4", "G5", "G6"}:
        from dronez.eval.gates import check_gate  # noqa: PLC0415

        base = _latest_baseline_metrics("pid") if g == "G6" else None
        res = {**check_gate(m, g, baseline=base), "origin": "computed"}
    else:
        raise ToolError(f"gate {gate!r} has no stored verdict in {path}; use G3 (stored), G4-G6")
    crit = res["criteria"]
    return {
        "gate": g,
        "pass": res["pass"],
        "origin": res["origin"],
        "criteria": crit[:MAX_ARRAY],
        "truncated": len(crit) > MAX_ARRAY,
        "source": str(path),
    }


@tool
def model_info(drone: str = "default_5in", sim: str = "default") -> dict[str, Any]:
    """Model hash, cache path, mass, CoM, principal inertia, hover command and T/W."""
    from dronez.cad.artifacts import MissingModelError, load_artifacts  # noqa: PLC0415
    from dronez.cad.schema import load_drone_config, model_hash  # noqa: PLC0415
    from dronez.sim import (  # noqa: PLC0415
        MotorModel,
        MotorParams,
        hover_command,
        hover_thrust_per_motor,
    )

    cfg = load_drone_config(drone)
    try:
        art = load_artifacts(cfg)
    except MissingModelError as exc:
        raise ToolError(str(exc)) from exc
    mp = art.mass_props
    params = MotorParams.from_drone(cfg)
    u_h = hover_command(params, mp.mass_kg)
    t_h = hover_thrust_per_motor(mp.mass_kg)
    t_max = float(MotorModel(params).params.k_f[0] * params.omega_max[0] ** 2)
    return {
        "drone": cfg.name,
        "model_hash": model_hash(cfg),
        "cache_path": str(art.root),
        "mass_kg": mp.mass_kg,
        "com_c_m": list(mp.com_c_m),
        "principal_inertia_kg_m2": list(mp.principal_moments_kg_m2),
        "hover_command": float(u_h[0]),
        "hover_thrust_per_motor_n": t_h,
        "max_thrust_per_motor_n": t_max,
        "thrust_to_weight": 4 * t_max / (mp.mass_kg * 9.81),
        "source": str(art.root / "mass_props.json"),
    }


@tool
def sim_rollout(
    controller: str = "pid",
    scenario: str = "hover_hold",
    seed: int = 0,
    duration_s: float = 2.0,
    *,
    run_id: str | None = None,
    ckpt: str | None = None,
) -> dict[str, Any]:
    """One closed-loop episode in-process; summary metrics only (``duration_s`` <= 10).

    ``controller``: ``pid`` | ``geometric`` | ``policy`` (needs ``run_id``, optional ``ckpt``).
    ``scenario``: a name from the ``baseline`` eval suite.
    """
    import gymnasium as gym  # noqa: PLC0415

    import dronez.envs  # noqa: F401, PLC0415
    from dronez.envs.config import EnvConfig  # noqa: PLC0415
    from dronez.eval.controllers import ControllerSpec  # noqa: PLC0415
    from dronez.eval.runner import run_episode  # noqa: PLC0415
    from dronez.eval.scenarios import load_suite, make_plan  # noqa: PLC0415
    from dronez.eval.suite import episode_metrics  # noqa: PLC0415

    if controller not in {"pid", "geometric", "policy"}:
        raise ToolError("controller must be pid, geometric or policy")
    if not 0.0 < duration_s <= MAX_ROLLOUT_S:
        raise ToolError(f"duration_s must be in (0, {MAX_ROLLOUT_S:g}]")
    suite = load_suite("baseline")
    by_name = {s.name: s for s in suite.scenarios}
    if scenario not in by_name:
        raise ToolError(f"unknown scenario {scenario!r}; available: {sorted(by_name)}")
    sc = by_name[scenario].model_copy(update={"duration_s": float(duration_s)})
    spec = ControllerSpec(controller, run_id=run_id, ckpt=ckpt)
    ctrl = spec()
    env = gym.make("DronezAttitude-v0", env_cfg=EnvConfig(info_level="full"))
    try:
        traj = run_episode(env, ctrl, make_plan(sc, int(seed)))
    finally:
        env.close()
    return {
        "controller": spec.label(),
        "scenario": scenario,
        "seed": int(seed),
        "duration_s": float(duration_s),
        "steps": int(len(traj.quat_wxyz) - 1),
        "crashed": bool(traj.crashed),
        "metrics": episode_metrics(traj, sc),
    }
