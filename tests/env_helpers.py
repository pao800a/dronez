"""Test helpers that are independent of the Phase 4 API names (those live in ``env_api``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

ATT, HOV = "DronezAttitude-v0", "DronezHover-v0"

# ADR-0006 §4 tables, default weights: term -> (lo, hi) per-step contribution.
SHARED = {
    "alive": (1.0, 1.0),
    "rate": (-0.1, 0.0),
    "act_mag": (-0.05, 0.0),
    "act_smooth": (-1.0, 0.0),
    "act_sat": (-0.1, 0.0),
    "penalty_clip": (0.0, 1.0),
    "crash": (-10.0, 0.0),
}
RANGES = {
    ATT: SHARED | {"att": (-0.5, 0.0), "att_fine": (0.0, 0.5), "vz": (-0.1, 0.0)},
    HOV: SHARED
    | {
        "att": (-0.2, 0.0),
        "att_fine": (0.0, 0.2),
        "pos": (-0.5, 0.0),
        "pos_fine": (0.0, 0.5),
        "vel": (-0.1, 0.0),
    },
}
EPS = 1e-9


def check_step(env_id: str, r: float, terms: dict[str, float], terminated: bool) -> None:
    assert set(terms) == set(RANGES[env_id]), f"unexpected term keys {sorted(terms)}"
    assert abs(sum(terms.values()) - r) <= EPS, f"terms sum {sum(terms.values())} != r {r}"
    for k, (lo, hi) in RANGES[env_id].items():
        assert lo - EPS <= terms[k] <= hi + EPS, f"{k}={terms[k]} outside [{lo}, {hi}]"
    if terminated:
        assert terms["crash"] == pytest.approx(-10.0)
    else:
        assert terms["crash"] == 0.0, "crash term on a step that did not terminate"
        assert r >= -EPS, "a non-terminal step must never have negative reward"


# --- driving an env with a baseline ------------------------------------------------------------
def sensed_of(info: dict[str, Any]) -> Any:
    """``Sensed`` from ``info["sensed"]`` (what the eval harness hands to a controller)."""
    from dronez.sim import Sensed

    s = info["sensed"]
    return Sensed(
        quat_wxyz=np.asarray(s["quat_wxyz"], dtype=np.float64),
        gyro_b=np.asarray(s["gyro_b"], dtype=np.float64),
        accel_b=np.asarray(s["accel_b"], dtype=np.float64),
        pos_w=None if s.get("pos_w") is None else np.asarray(s["pos_w"], dtype=np.float64),
        vel_w=None if s.get("vel_w") is None else np.asarray(s["vel_w"], dtype=np.float64),
    )


def pid_for(env: Any) -> Any:
    """A nominal ``CascadedPID`` that flies this env (same envelope as the env's command map)."""
    from dronez.control import CascadedPID

    u = env.unwrapped
    return CascadedPID(u.vehicle, dt=u.dt, cmap=u.cmap)


def pid_setpoint(env_id: str, info: dict[str, Any], p0: Any) -> Any:
    """The baseline setpoint for an env: attitude hold (Attitude) or position hold (Hover)."""
    from dronez.control import Setpoint
    from dronez.utils import rotations

    q_t = np.asarray(info["target"]["quat_wxyz"], dtype=np.float64)
    if env_id == HOV:
        return Setpoint(
            pos_w=np.asarray(info["target"]["pos_w"], dtype=np.float64),
            yaw=float(rotations.yaw_angle(q_t)),
            hold_xy=True,
        )
    return Setpoint(quat_wxyz=q_t, pos_w=np.asarray(p0, dtype=np.float64), hold_xy=False)


def fly_pid(env: Any, env_id: str, n_steps: int, seed: int = 0) -> list[tuple[bool, bool, Any]]:
    """Reset (seeded) and fly ``n_steps`` with the PID: ``[(terminated, truncated, info), ...]``."""
    pid = pid_for(env)
    obs, info = env.reset(seed=seed)
    pid.reset(seed)
    p0 = np.asarray(info["true"]["pos_w"], dtype=np.float64)
    out = []
    for _ in range(n_steps):
        pid.set_setpoint(pid_setpoint(env_id, info, p0))
        a = pid.act(obs, sensed_of(info))
        obs, _, terminated, truncated, info = env.step(a)
        out.append((terminated, truncated, info))
        if terminated or truncated:
            break
    return out
