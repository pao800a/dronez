"""Closed-loop episode runner over the Gymnasium env (docs/architecture.md §7).

    obs, info = env.reset(seed=s, options=plan_options(plan, env))
    loop:  a = controller.act(obs, Sensed(**info["sensed"]))
           obs, r, term, trunc, info = env.step(a)

Controllers see ``obs`` and ``info["sensed"]`` only. The runner reads ``info["true"]`` for the
metrics record (never passed on). ``crashed`` means the env ``terminated`` (ADR-0006 §5: there is
no tilt-based crash rule; inverted is a valid state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from dronez.control.base import Controller
from dronez.envs.base import DronezBaseEnv
from dronez.envs.randomization import scale_noise
from dronez.eval.scenarios import DrSpec, EpisodePlan
from dronez.sim import PhysParams, Sensed

F64 = NDArray[np.float64]

#: per-motor thrust-constant mismatch patterns (ADR-0006 §9 G6): +-5 %
_MISMATCH = {
    "roll": np.array([-0.05, 0.05, 0.05, -0.05]),  # M0, M3 -5 %; M1, M2 +5 %
    "yaw": np.array([-0.05, -0.05, 0.05, 0.05]),  # M0, M1 -5 %; M2, M3 +5 %
}


@dataclass
class Trajectory:
    """Recorded episode. State arrays have ``n + 1`` samples (index 0 = reset); ``a``, ``u`` have
    ``n``: the clipped action and the normalized motor command ``cmap(a)``."""

    scenario: str
    seed: int
    dt: float
    t: F64
    pos_w: F64
    quat_wxyz: F64
    linvel_w: F64
    angvel_b: F64
    qpos: F64  # (n+1, 7) for video replay
    quat_target: F64
    pos_target: F64
    u: F64  # (n, 4) motor command in [0, 1]
    a: F64  # (n, 4) applied (clipped) action in [-1, 1]
    sensed_quat: F64
    crashed: bool
    termination: str | None = None

    def to_npz(self) -> dict[str, np.ndarray]:
        skip = {"scenario", "crashed", "termination"}
        return {
            k: np.asarray(v)
            for k, v in self.__dict__.items()
            if k not in skip and not isinstance(v, str)
        } | {"crashed": np.asarray(self.crashed)}


def _unwrap(env: gym.Env[Any, Any]) -> DronezBaseEnv:
    base = env.unwrapped
    if not isinstance(base, DronezBaseEnv):
        raise TypeError(f"the eval runner needs a DronezBaseEnv, got {type(base).__name__}")
    return base


def resolve_params(spec: DrSpec, plan: EpisodePlan, env: DronezBaseEnv) -> tuple[PhysParams, float]:
    """``(PhysParams, noise_scale)`` of an episode from its ``DrSpec`` and the plan's streams."""
    rd = env.randomizer
    lat = plan.latency_s
    if spec.mode == "nominal":
        params = rd.make_params(latency_s=lat)
        return params, 1.0
    if spec.mode == "sample":
        draw = rd.draw(plan.rngs["dr"], spec.scale)
        phys = draw.phys
        if lat > 0.0:  # an explicit latency overrides the drawn one
            phys.latency_steps = round(lat / env.sim_cfg.physics_dt_s)
        return phys, draw.noise_scale
    o = dict(spec.overrides)
    k_f = float(o.get("k_f", 1.0)) * (1.0 + _MISMATCH.get(str(o.get("mismatch")), np.zeros(4)))
    com = tuple(float(x) for x in o.get("com_mm", (0.0, 0.0, 0.0)))
    params = rd.make_params(
        mass=float(o.get("mass", 1.0)),
        inertia=float(o.get("inertia", 1.0)),
        com_mm=com,  # type: ignore[arg-type]
        k_f=tuple(k_f),
        k_m=float(o.get("k_m", 1.0)),
        tau_m_s=None if "tau_m_s" not in o else float(o["tau_m_s"]),
        latency_s=lat,
        battery=float(o.get("battery", 1.0)),
    )
    return params, float(o.get("noise", 1.0))


def plan_options(plan: EpisodePlan, env: DronezBaseEnv) -> dict[str, Any]:
    """``env.reset(options=...)`` for ``plan``: plain data only (ADR-0006 §9)."""
    sc = plan.scenario
    params, noise_scale = resolve_params(sc.dr, plan, env)
    return {
        "qpos0": plan.qpos0,
        "qvel0": plan.qvel0,
        "a0": plan.a0,
        "target_schedule": plan.target_schedule,
        "pos_target": plan.pos_target,
        "phys_params": params,
        "disturbance": plan.disturbance,
        "noise": scale_noise(env.cfg.noise, noise_scale),
        "episode_len_s": sc.duration_s,
    }


def sensed_of(info: dict[str, Any]) -> Sensed:
    s = info["sensed"]
    return Sensed(
        quat_wxyz=s["quat_wxyz"],
        gyro_b=s["gyro_b"],
        accel_b=s["accel_b"],
        pos_w=s.get("pos_w"),
        vel_w=s.get("vel_w"),
    )


def run_episode(env: gym.Env[Any, Any], controller: Controller, plan: EpisodePlan) -> Trajectory:
    """Closed loop for ``plan.scenario.duration_s``; the controller sees ``obs`` and ``Sensed``."""
    base = _unwrap(env)
    if base.cfg.info_level != "full":
        raise ValueError("eval needs EnvConfig(info_level='full')")
    sc = plan.scenario
    dt = base.dt
    n = round(sc.duration_s / dt)
    controller.reset(plan.seed)
    obs, info = env.reset(seed=plan.seed, options=plan_options(plan, base))
    set_sp = getattr(controller, "set_setpoint", None)

    def snap(i: dict[str, Any]) -> tuple[F64, ...]:
        tr = i["true"]
        return (
            np.asarray(tr["pos_w"], dtype=np.float64),
            np.asarray(tr["quat_wxyz"], dtype=np.float64),
            np.asarray(tr["linvel_w"], dtype=np.float64),
            np.asarray(tr["angvel_b"], dtype=np.float64),
            base.sim.data.qpos.copy(),
            np.asarray(i["target"]["quat_wxyz"], dtype=np.float64),
            np.asarray(i["sensed"]["quat_wxyz"], dtype=np.float64),
        )

    pos, quat, vel, ang, qpos, qtgt, sq = ([x] for x in snap(info))
    a_log: list[F64] = []
    u_log: list[F64] = []
    crashed = False
    reason: str | None = None
    for k in range(n):
        if set_sp is not None:
            set_sp(plan.setpoint(k * dt))
        action = controller.act(obs, sensed_of(info))
        obs, _, terminated, truncated, info = env.step(action)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a_log.append(a)
        u_log.append(base.cmap.action_to_command(a))
        for lst, val in zip((pos, quat, vel, ang, qpos, qtgt, sq), snap(info), strict=True):
            lst.append(val)
        if terminated:
            crashed, reason = True, info["termination"]
            break
        if truncated:
            reason = info["termination"]
            break
    nn = len(pos)
    return Trajectory(
        scenario=sc.name,
        seed=plan.seed,
        dt=dt,
        t=np.arange(nn, dtype=np.float64) * dt,
        pos_w=np.array(pos),
        quat_wxyz=np.array(quat),
        linvel_w=np.array(vel),
        angvel_b=np.array(ang),
        qpos=np.array(qpos),
        quat_target=np.array(qtgt),
        pos_target=np.tile(plan.pos_target, (nn, 1)),
        u=np.array(u_log).reshape(-1, 4),
        a=np.array(a_log).reshape(-1, 4),
        sensed_quat=np.array(sq),
        crashed=crashed,
        termination=reason,
    )
