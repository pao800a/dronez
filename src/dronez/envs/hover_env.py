"""``DronezHover-v0``: hold a target position and heading (Attitude obs + 5 task dims).

Target ``p_t = (0, 0, z_t)`` with a yaw-only attitude target; the position error and velocity are
expressed in the target heading frame (ADR-0006 §2).
"""

from __future__ import annotations

import math

import numpy as np

from dronez.envs.base import (
    F32,
    F64,
    DronezBaseEnv,
    Task,
    cos_uniform_angle,
    horizontal_axis,
    unit_vector,
)
from dronez.envs.curriculum import HOVER_LEVELS
from dronez.envs.obs import ObsLayout
from dronez.sim import Sensed, TrueState
from dronez.utils import rotations

_I = {
    "z_t": 0,
    "psi": 1,
    "head": 2,
    "r_off": 3,
    "off_cz": 4,
    "off_phi": 5,
    "delta_cos": 6,
    "delta_phi": 7,
    "w_mag": 8,
    "w_cz": 9,
    "w_phi": 10,
    "v_mag": 11,
    "v_cz": 12,
    "v_phi": 13,
    "a0": 14,  # 14..17
}
N_INIT = 18


class HoverEnv(DronezBaseEnv):
    KIND = "hover"
    DEFAULT_EPISODE_S = 8.0
    DEFAULT_LEVELS = HOVER_LEVELS

    def _make_layout(self) -> ObsLayout:
        return ObsLayout.hover()

    def _sample_task(self, rng: np.random.Generator, level: int) -> Task:
        u = rng.random(N_INIT)
        s = self.levels[level]
        z_lo, z_hi = self.cfg.hover_z_range_m
        p_t = np.array([0.0, 0.0, z_lo + u[_I["z_t"]] * (z_hi - z_lo)])
        psi = (2.0 * u[_I["psi"]] - 1.0) * math.pi
        q_t = rotations.quat_from_yaw(psi)
        dpsi = (2.0 * u[_I["head"]] - 1.0) * math.radians(s.heading_dev_deg)
        delta = cos_uniform_angle(u[_I["delta_cos"]], math.radians(s.delta_max_deg))
        phi_d = (2.0 * u[_I["delta_phi"]] - 1.0) * math.pi
        q0 = rotations.quat_mul(
            rotations.quat_from_axis_angle(horizontal_axis(phi_d), delta),
            rotations.quat_from_yaw(psi + dpsi),
        )
        off = (
            s.offset_m
            * u[_I["r_off"]] ** (1.0 / 3.0)
            * unit_vector(2.0 * u[_I["off_cz"]] - 1.0, (2.0 * u[_I["off_phi"]] - 1.0) * math.pi)
        )
        w = (
            s.omega_max_rad_s
            * u[_I["w_mag"]]
            * unit_vector(2.0 * u[_I["w_cz"]] - 1.0, (2.0 * u[_I["w_phi"]] - 1.0) * math.pi)
        )
        v = (
            s.v0_max_m_s
            * u[_I["v_mag"]] ** (1.0 / 3.0)
            * unit_vector(2.0 * u[_I["v_cz"]] - 1.0, (2.0 * u[_I["v_phi"]] - 1.0) * math.pi)
        )
        a0 = s.a0_max * (2.0 * u[_I["a0"] : _I["a0"] + 4] - 1.0)
        return Task(
            qpos0=np.concatenate([p_t + off, q0]),
            qvel0=np.concatenate([v, w]),
            a0=np.asarray(a0, dtype=np.float64),
            pos_target=p_t,
            schedule=[(0.0, q_t)],
        )

    def _extra_obs(self, out: F32, sensed: Sensed) -> None:
        assert sensed.pos_w is not None
        assert sensed.vel_w is not None
        q_t = self._q_t
        out[29:32] = rotations.quat_rotate_inverse(q_t, sensed.pos_w - self._pos_t)
        out[32:34] = rotations.quat_rotate_inverse(q_t, sensed.vel_w)[:2]

    def _pos_err_h(self, true: TrueState) -> F64:
        return np.asarray(rotations.quat_rotate_inverse(self._q_t, true.pos_w - self._pos_t))

    def _reward_extras(self, true: TrueState) -> tuple[float, float, float]:
        return (
            float(true.linvel_w[2]),
            float(np.linalg.norm(true.linvel_w)),
            float(np.linalg.norm(true.pos_w - self._pos_t)),
        )

    def _out_of_bounds(self, qpos: F64) -> bool:
        dx, dy, dz = qpos[0] - self._pos_t[0], qpos[1] - self._pos_t[1], qpos[2] - self._pos_t[2]
        return bool(dx * dx + dy * dy + dz * dz > self.cfg.oob_hover_m**2)

    def _step_ok(self, att_err: float, pos_err: float) -> bool:
        return pos_err < self.cfg.success_pos_m

    def _extra_metrics(self, pos_err: float) -> dict[str, float]:
        return {"pos_err_m": pos_err}
