"""Outer position loop and thrust-vector -> attitude target, shared by both baselines.

World-frame PD(+I) on the estimator position/velocity (``Sensed.pos_w`` / ``vel_w``) gives a
desired acceleration ``a_d``. The desired force is ``F = m (a_d + g e_z)``; its direction is the
desired body z axis ``b3d`` and the heading setpoint fixes the remaining rotation:
``b2d = b3d x b1c / |.|``, ``b1d = b2d x b3d`` with ``b1c = (cos psi, sin psi, 0)`` (Lee et al.
2010 construction, no Euler angles in the controller state).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

F64 = NDArray[np.float64]


class PositionGains(BaseModel):
    """PD(+I) gains of the position loop. ``kp`` [1/s^2], ``kd`` [1/s], ``ki`` [1/s^3]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kp_xy: float = Field(default=5.0, ge=0.0)
    kd_xy: float = Field(default=3.5, ge=0.0)
    ki_xy: float = Field(default=0.0, ge=0.0)
    kp_z: float = Field(default=12.0, ge=0.0)
    kd_z: float = Field(default=6.0, ge=0.0)
    ki_z: float = Field(default=2.0, ge=0.0)
    max_tilt_deg: float = Field(default=30.0, gt=0.0, lt=89.0)  # cap on the commanded tilt
    max_accel_z: float = Field(default=8.0, gt=0.0)  # |a_z| cap [m/s^2]
    i_limit: float = Field(default=2.0, ge=0.0)  # integral clamp [m/s^2]


def desired_rotation(b3d: F64, yaw: float) -> F64:
    """``R_d`` (columns ``b1d, b2d, b3d``) from the desired thrust direction and heading."""
    b3 = b3d / max(float(np.linalg.norm(b3d)), 1e-12)
    b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = np.cross(b3, b1c)
    n = float(np.linalg.norm(b2))
    if n < 1e-6:  # b3d (anti)parallel to the heading vector: pick any consistent frame
        b2 = np.cross(b3, np.array([0.0, 1.0, 0.0]))
        n = float(np.linalg.norm(b2))
    b2 = b2 / n
    b1 = np.cross(b2, b3)
    return np.column_stack([b1, b2, b3])


class PositionLoop:
    """World-frame PD(+I) with a clamped integral and a tilt cap on the horizontal command."""

    def __init__(self, gains: PositionGains, dt: float) -> None:
        self.g = gains
        self.dt = dt
        self._int = np.zeros(3)

    def reset(self) -> None:
        self._int = np.zeros(3)

    def update(
        self,
        pos: F64,
        vel: F64,
        pos_d: F64,
        vel_d: F64,
        *,
        hold_xy: bool,
        gravity: float,
    ) -> F64:
        """Desired world acceleration [m/s^2] (gravity not included)."""
        c = self.g
        e = pos_d - pos
        ev = vel_d - vel
        kp = np.array([c.kp_xy, c.kp_xy, c.kp_z])
        kd = np.array([c.kd_xy, c.kd_xy, c.kd_z])
        ki = np.array([c.ki_xy, c.ki_xy, c.ki_z])
        if not hold_xy:
            kp[:2] = 0.0
            kd[:2] = 0.0
            ki[:2] = 0.0
        a = kp * e + kd * ev + self._int
        self._int = np.clip(self._int + ki * e * self.dt, -c.i_limit, c.i_limit)
        a[2] = float(np.clip(a[2], -c.max_accel_z, c.max_accel_z))
        a_xy_max = (gravity + a[2]) * np.tan(np.deg2rad(c.max_tilt_deg))
        n_xy = float(np.linalg.norm(a[:2]))
        if n_xy > a_xy_max > 0.0:
            a[:2] *= a_xy_max / n_xy
        return a
