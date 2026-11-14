"""Gates G4-G6 (ADR-0006 §9) from per-episode metrics, and ``check_gate``.

Thresholds live in ``configs/eval/gates.yaml`` (pydantic ``GateThresholds``) and change only by ADR.
``check_gate(metrics, gate, baseline=None)`` works on a ``metrics.json`` dict (``scenarios`` ->
``episodes``), so it can be re-run on stored results. Common conditions (deterministic policy,
nominal plant, paired seeds, ``crashed = terminated``) are enforced by the suites, not here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from dronez.utils.config import load_model


class G4Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hover_rms_tilt_rad: float = 0.0349  # 2 deg
    hover_crash_count: int = 0
    recovery_lt90_success_rate: float = 0.95


class G5Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recovery_full_success_rate: float = 0.90
    ground_contact_count: int = 0


class G6Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sweep_success_rate_pooled: float = 0.90
    sweep_success_rate_min_config: float = 0.80
    latency20_success_rate: float = 0.90
    hf_rms_u_margin: float = 0.001
    mean_abs_delta_u_margin: float = 0.001
    smoothness_scenarios: tuple[str, ...] = ("hover_hold", "hover_hold_dr")


class GateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    G4: G4Thresholds = G4Thresholds()
    G5: G5Thresholds = G5Thresholds()
    G6: G6Thresholds = G6Thresholds()


def load_gate_thresholds(name_or_path: str | Path = "gates") -> GateThresholds:
    return load_model(GateThresholds, name_or_path, "eval")


def _eps(metrics: dict[str, Any], name: str) -> list[dict[str, Any]]:
    sc = metrics.get("scenarios", {}).get(name)
    return [] if sc is None else list(sc["episodes"])


def _rate(eps: list[dict[str, Any]], key: str) -> float:
    vals = [bool(e.get(key)) for e in eps]
    return float(np.mean(vals)) if vals else float("nan")


def _crit(name: str, value: float | int | bool | None, thr: Any, ok: bool) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value):
        value, ok = None, False
    return {"name": name, "value": value, "threshold": thr, "pass": bool(ok)}


def g4_criteria(metrics: dict[str, Any], t: G4Thresholds) -> list[dict[str, Any]]:
    hover = _eps(metrics, "hover_hold")
    n = sum(float(e["tilt_n"]) for e in hover)
    rms = math.sqrt(sum(float(e["tilt_sq_sum"]) for e in hover) / n) if n > 0 else float("nan")
    crashes = sum(bool(e["crashed"]) for e in hover) if hover else None
    rec = _rate(_eps(metrics, "recovery_lt90"), "success_recovery")
    return [
        _crit("hover_rms_tilt_rad", rms, t.hover_rms_tilt_rad, rms < t.hover_rms_tilt_rad),
        _crit(
            "hover_crash_count",
            crashes,
            t.hover_crash_count,
            crashes is not None and crashes <= t.hover_crash_count,
        ),
        _crit(
            "recovery_lt90_success_rate",
            rec,
            t.recovery_lt90_success_rate,
            rec >= t.recovery_lt90_success_rate,
        ),
    ]


def g5_criteria(metrics: dict[str, Any], t: G5Thresholds, g4_pass: bool) -> list[dict[str, Any]]:
    full = _eps(metrics, "recovery_full")
    rate = _rate(full, "success_recovery")
    gc = sum(bool(e["ground_contact"]) for e in full) if full else None
    return [
        _crit(
            "recovery_full_success_rate",
            rate,
            t.recovery_full_success_rate,
            rate >= t.recovery_full_success_rate,
        ),
        _crit(
            "ground_contact_count",
            gc,
            t.ground_contact_count,
            gc is not None and gc <= t.ground_contact_count,
        ),
        _crit("g4_pass", g4_pass, True, g4_pass),
    ]


def _smoothness(
    metrics: dict[str, Any], baseline: dict[str, Any] | None, t: G6Thresholds
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sc in t.smoothness_scenarios:
        pol, ref = _eps(metrics, sc), _eps(baseline or {}, sc)
        for key, margin in (
            ("hf_rms_u", t.hf_rms_u_margin),
            ("mean_abs_delta_u", t.mean_abs_delta_u_margin),
        ):
            # only episodes where neither controller terminated (paired by episode index)
            pairs = [
                (p[key], r[key])
                for p, r in zip(pol, ref, strict=False)
                if not p["crashed"] and not r["crashed"]
            ]
            if not pairs:
                out.append(_crit(f"{sc}: {key} - PID", None, margin, False))
                continue
            mp_, mr = (float(np.mean([x[i] for x in pairs])) for i in (0, 1))
            out.append(
                _crit(
                    f"{sc}: {key} - PID [policy {mp_:.5f}, pid {mr:.5f}, n={len(pairs)}]",
                    mp_ - mr,
                    margin,
                    mp_ <= mr + margin,
                )
            )
    return out


def g6_criteria(
    metrics: dict[str, Any], t: G6Thresholds, baseline: dict[str, Any] | None
) -> list[dict[str, Any]]:
    scen = metrics.get("scenarios", {})
    fixed = [n for n in scen if n.startswith("sweep_")]
    per = {n: _rate(_eps(metrics, n), "success_recovery") for n in fixed}
    pooled_eps = [e for n in [*fixed, "dr_random"] for e in _eps(metrics, n)]
    pooled = _rate(pooled_eps, "success_recovery")
    min_cfg = min(per.values()) if per else float("nan")
    lat = per.get("sweep_latency_20ms", float("nan"))
    crit = [
        _crit(
            "sweep_success_rate_pooled",
            pooled,
            t.sweep_success_rate_pooled,
            pooled >= t.sweep_success_rate_pooled,
        ),
        _crit(
            "sweep_success_rate_min_config",
            min_cfg,
            t.sweep_success_rate_min_config,
            min_cfg >= t.sweep_success_rate_min_config,
        ),
        _crit(
            "latency20_success_rate", lat, t.latency20_success_rate, lat >= t.latency20_success_rate
        ),
    ]
    return crit + _smoothness(metrics, baseline, t)


def check_gate(
    metrics: dict[str, Any],
    gate: str,
    baseline: dict[str, Any] | None = None,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    """``{"pass": bool, "criteria": [{name, value, threshold, pass}]}`` for ``G4``/``G5``/``G6``.

    ``baseline`` is the PID ``metrics.json`` on the same suite (needed for G6 smoothness). G5
    needs the G4 scenarios in the same ``metrics`` (``g4_pass`` of the same checkpoint).
    """
    th = thresholds or load_gate_thresholds()
    g = gate.upper()
    if g == "G4":
        crit = g4_criteria(metrics, th.G4)
    elif g == "G5":
        g4 = metrics.get("gates", {}).get("G4")
        if g4 is None or "hover_hold" in metrics.get("scenarios", {}):
            g4 = {"pass": all(c["pass"] for c in g4_criteria(metrics, th.G4))}
        crit = g5_criteria(metrics, th.G5, bool(g4["pass"]))
    elif g == "G6":
        crit = g6_criteria(metrics, th.G6, baseline)
    else:
        raise ValueError(f"unknown gate {gate!r}; use G4, G5 or G6")
    return {"pass": bool(crit) and all(c["pass"] for c in crit), "criteria": crit}
