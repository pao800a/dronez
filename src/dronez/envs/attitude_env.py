"""``DronezAttitude-v0``: level/commanded attitude, disturbance rejection, recovery up to inverted.

Heading-invariant observation (ADR-0006 §2: 6D rotation error, gravity, gyro, vertical velocity,
action history); the altitude is only softly regulated through the ``vz`` penalty.
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
from dronez.envs.curriculum import ATTITUDE_LEVELS
from dronez.envs.obs import ObsLayout
from dronez.sim import TrueState
from dronez.utils import rotations

#: indices into the fixed-length ``rng_init`` block (draw order is part of the contract, D2)
_I = {
    "mix": 0,
    "mix_k": 1,
    "z0": 2,
    "tilted": 3,
    "psi": 4,
    "tilt_cos": 5,
    "tilt_phi": 6,
    "delta_cos": 7,
    "delta_phi": 8,
    "w_mag": 9,
    "w_cz": 10,
    "w_phi": 11,
    "v_mag": 12,
    "v_cz": 13,
    "v_phi": 14,
    "a0": 15,  # 4 values: 15..18
    "step_flag": 19,
    "step_t": 20,
    "step_cos": 21,
    "step_cz": 22,
    "step_phi": 23,
}
N_INIT = 24


class AttitudeEnv(DronezBaseEnv):
    KIND = "attitude"
    DEFAULT_EPISODE_S = 5.0
    DEFAULT_LEVELS = ATTITUDE_LEVELS

    def _make_layout(self) -> ObsLayout:
        return ObsLayout.attitude()

    def _sample_task(self, rng: np.random.Generator, level: int) -> Task:
        u = rng.random(N_INIT)
        k = level
        if k >= 1 and u[_I["mix"]] < self.cfg.p_lower_level:
            k = min(int(u[_I["mix_k"]] * k), k - 1)  # uniform over the lower levels
        s = self.levels[k]
        z_lo, z_hi = self.cfg.z0_range_m
        z0 = z_lo + u[_I["z0"]] * (z_hi - z_lo)

        # target: heading hold, optionally tilted (<= 20 deg), optionally stepped mid-episode
        psi = (2.0 * u[_I["psi"]] - 1.0) * math.pi
        q_t = rotations.quat_from_yaw(psi)
        if u[_I["tilted"]] < s.p_tilted:
            tilt = cos_uniform_angle(u[_I["tilt_cos"]], math.radians(20.0))
            phi = (2.0 * u[_I["tilt_phi"]] - 1.0) * math.pi
            q_t = rotations.quat_mul(
                rotations.quat_from_axis_angle(horizontal_axis(phi), tilt), q_t
            )

        # initial attitude: error delta (cos-uniform cap) about a random horizontal world axis
        delta = cos_uniform_angle(u[_I["delta_cos"]], math.radians(s.delta_max_deg))
        phi_d = (2.0 * u[_I["delta_phi"]] - 1.0) * math.pi
        q0 = rotations.quat_mul(rotations.quat_from_axis_angle(horizontal_axis(phi_d), delta), q_t)

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

        schedule = [(0.0, q_t)]
        if u[_I["step_flag"]] < s.p_step:
            t_s = 1.0 + 2.0 * u[_I["step_t"]]
            ang = cos_uniform_angle(u[_I["step_cos"]], math.radians(30.0))
            axis = unit_vector(
                2.0 * u[_I["step_cz"]] - 1.0, (2.0 * u[_I["step_phi"]] - 1.0) * math.pi
            )
            q_new = rotations.quat_mul(rotations.quat_from_axis_angle(axis, ang), q_t)
            schedule.append((float(t_s), q_new))

        pos0 = np.array([0.0, 0.0, z0])
        return Task(
            qpos0=np.concatenate([pos0, q0]),
            qvel0=np.concatenate([v, w]),
            a0=np.asarray(a0, dtype=np.float64),
            pos_target=pos0.copy(),
            schedule=schedule,
        )

    def _reward_extras(self, true: TrueState) -> tuple[float, float, float]:
        return float(true.linvel_w[2]), 0.0, 0.0

    def _out_of_bounds(self, qpos: F64) -> bool:
        dx, dy = qpos[0] - self._p0[0], qpos[1] - self._p0[1]
        c = self.cfg
        return bool(dx * dx + dy * dy > c.oob_xy_m**2 or qpos[2] > self._p0[2] + c.oob_z_m)

    def _step_ok(self, att_err: float, pos_err: float) -> bool:
        return att_err < self._att_ok_rad

    def _extra_obs(self, out: F32, sensed: object) -> None:
        return
