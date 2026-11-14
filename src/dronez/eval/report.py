"""Write ``metrics.json``, ``report.md`` and plots for a baseline run.

Layout (docs/architecture.md §8, baselines):
``eval/baseline-<controller>/<suite_hash>-<model_hash>/`` with ``suite.resolved.yaml``,
``metrics.json``, ``report.md``, ``plots/*.png``, ``trajectories/*.npz`` (episode 0 of every
scenario) and ``videos/*.mp4``. Values are stored in SI; degrees and centimetres appear only in
``report.md`` and the plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dronez.eval import metrics as M
from dronez.eval.scenarios import EvalSuite
from dronez.eval.suite import SCHEMA_VERSION, ScenarioResult, aggregate, g3_gate, git_sha


def run_dir_for(controller: str, suite_hash: str, model_hash: str, env_hash: str = "") -> Path:
    """Baseline cache dir; the env-config hash is part of the key (ADR-0006 §9)."""
    from dronez.utils import paths  # noqa: PLC0415

    key = f"{suite_hash}-{model_hash}" + (f"-{env_hash}" if env_hash else "")
    return paths.eval_dir(f"baseline-{controller}", key)


def policy_dir_for(run_id: str, ckpt: str) -> Path:
    """``$DRONEZ_DATA_DIR/eval/<run_id>/<checkpoint>/`` (architecture §8)."""
    from dronez.utils import paths  # noqa: PLC0415

    return paths.eval_dir(run_id, ckpt)


def _json_safe(v: Any) -> Any:
    return None if (isinstance(v, float) and not np.isfinite(v)) else v


def _default_gates(results: dict[str, ScenarioResult]) -> dict[str, Any]:
    """G3 (the legacy baseline gate) when its scenarios are present."""
    legacy_hover = any(n == "hover_hold" and not r.scenario.random_yaw for n, r in results.items())
    if "recovery_30deg" in results or legacy_hover:
        return {"G3": g3_gate(results)}
    return {}


def build_metrics(  # noqa: PLR0917
    controller: str,
    suite: EvalSuite,
    model_hash: str,
    results: dict[str, ScenarioResult],
    extra: dict[str, Any],
    gates: dict[str, Any] | None = None,
    run_id: str | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    suite_hash = suite.hash()
    scenarios: dict[str, Any] = {}
    for name, res in results.items():
        eps = [{k: _json_safe(v) for k, v in e.items()} for e in res.episodes]
        scenarios[name] = {
            "kind": res.scenario.kind,
            "aggregate": aggregate(res.episodes),
            "episodes": eps,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "controller": controller,
        "run_id": run_id or f"baseline-{controller}",
        "checkpoint": checkpoint or f"{suite_hash}-{model_hash}",
        "suite_hash": suite_hash,
        "model_hash": model_hash,
        "git_sha": git_sha(),
        "units": "SI (rad, s, m); degrees only in report.md",
        "scenarios": scenarios,
        "gates": gates if gates is not None else _default_gates(results),
        **extra,
    }


def _plot_scenario(res: ScenarioResult, path: Path) -> None:
    sc = res.scenario
    two = sc.kind in {"hover_hold", "impulse"}
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(11 if two else 6, 3.6), squeeze=False)
    ax = axes[0, 0]
    for tr in res.trajectories:
        e = np.rad2deg(M.attitude_error(tr.quat_wxyz, tr.quat_target))
        ax.plot(tr.t, e, lw=0.8, alpha=0.7)
    ax.axhline(5.0, color="k", ls="--", lw=0.8, label="5 deg")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title(f"{sc.name}: geodesic attitude error ({len(res.trajectories)} seeds)")
    ax.legend(loc="upper right")
    if two:
        ax2 = axes[0, 1]
        for tr in res.trajectories:
            ax2.plot(tr.t, 100.0 * M.position_error(tr.pos_w, tr.pos_target), lw=0.8, alpha=0.7)
        ax2.axhline(5.0, color="k", ls="--", lw=0.8, label="5 cm")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("position error [cm]")
        ax2.set_title(f"{sc.name}: position error")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x: Any, scale: float = 1.0, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x * scale:.{nd}f}"


def _report_md(m: dict[str, Any], videos: list[str]) -> str:
    deg = float(np.rad2deg(1.0))
    lines = [
        f"# Eval report: {m['controller']} (suite `{m['suite_name']}`)",
        "",
        f"- suite `{m['suite_hash']}`, model `{m['model_hash']}`, git `{m['git_sha']}`",
        f"- sensor noise: {json.dumps(m['noise'])}",
        f"- control rate {m['policy_rate_hz']:.0f} Hz, physics {m['physics_rate_hz']:.0f} Hz",
        "",
    ]
    for gname, g in m["gates"].items():
        lines += [
            f"## Gate {gname}",
            "",
            f"**{'PASS' if g['pass'] else 'FAIL'}**",
            "",
            "| criterion | value | threshold | pass |",
            "|---|---|---|---|",
        ]
        for c in g["criteria"]:
            lines.append(
                f"| {c['name']} | {_fmt(c['value'], 1.0, 4)} | {c['threshold']} | {c['pass']} |"
            )
        lines.append("")
    lines += ["## Scenarios", ""]
    for name, s in m["scenarios"].items():
        a = s["aggregate"]
        lines += [
            f"### {name} ({a['n_episodes']} episodes)",
            "",
            f"success rate {a['success_rate']:.2f}, crash rate {a['crash_rate']:.2f}",
            "",
            "| metric | mean | p95 | max |",
            "|---|---|---|---|",
        ]
        rows = [
            ("time to < 5 deg [s]", "time_to_recover_s", 1.0),
            ("RMS attitude error [deg]", "rms_att_err_rad", deg),
            ("max attitude error [deg]", "max_att_err_rad", deg),
            ("overshoot [deg]", "overshoot_rad", deg),
            ("max position error [cm]", "max_pos_err_m", 100.0),
            ("RMS position error [cm]", "rms_pos_err_m", 100.0),
            ("saturation [%]", "saturation_pct", 1.0),
            ("mean |du| from 1 s [cmd units]", "mean_abs_delta_u", 1.0),
            ("HF RMS of u > 20 Hz, from 1 s [cmd units]", "hf_rms_u", 1.0),
            ("action energy > 20 Hz [fraction, report-only]", "hf_energy_frac_gt20hz", 1.0),
        ]
        for label, key, scale in rows:
            if key in a:
                lines.append(
                    f"| {label} | {_fmt(a[key]['mean'], scale, 3)} | "
                    f"{_fmt(a[key]['p95'], scale, 3)} | {_fmt(a[key]['max'], scale, 3)} |"
                )
        lines += ["", f"![{name}](plots/{name}.png)", ""]
    if videos:
        lines += ["## Videos (4x slow motion)", ""] + [f"- `{v}`" for v in videos] + [""]
    else:
        lines += ["## Videos", "", "none (not requested, or no OpenGL backend)", ""]
    lines += [
        "## Limitations",
        "",
        "- Plant, latency and noise per scenario (see `suite.resolved.yaml`); baselines fly the "
        "same envelope and env as the policy (ADR-0006).",
        "- Position/velocity come from a noisy-estimator abstraction (white noise only).",
        "- Propulsion constants are estimates (see `configs/drone/default_5in.yaml`).",
        "",
    ]
    return "\n".join(lines)


def write_report(  # noqa: PLR0917
    out: Path,
    controller: str,
    suite: EvalSuite,
    model_hash: str,
    results: dict[str, ScenarioResult],
    extra: dict[str, Any],
    videos: list[str] | None = None,
    gates: dict[str, Any] | None = None,
    run_id: str | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)
    m = build_metrics(controller, suite, model_hash, results, extra, gates, run_id, checkpoint)
    m["suite_name"] = suite.name
    (out / "suite.resolved.yaml").write_text(
        yaml.safe_dump(suite.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    for name, res in results.items():
        if not res.trajectories:
            continue
        _plot_scenario(res, out / "plots" / f"{name}.png")
        arrays: dict[str, Any] = dict(res.trajectories[0].to_npz())
        np.savez_compressed(out / "trajectories" / f"{name}_ep0.npz", **arrays)
    (out / "report.md").write_text(_report_md(m, videos or []), encoding="utf-8")
    return m
