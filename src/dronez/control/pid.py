"""Cascaded PID: geodesic attitude error -> body-rate setpoint -> rate PID -> torque -> mixer.

* Attitude loop (P): ``q_e = q_d* (x) q`` (``rotations.quat_error``); its rotation vector (shortest
  path, ``w >= 0``) is the geodesic error expressed in the body frame, and
  ``omega_sp = -kp_att * rotvec(q_e)`` clipped per axis. No Euler angles anywhere.
* Rate loop (``RateController``): PID on the body rates in **angular-acceleration units**,
  ``alpha = kp (w_sp - w) + ki int(w_sp - w) + kd d(-w_f)/dt`` (derivative on the low-pass
  filtered *measurement*, never on the setpoint), then ``tau = J alpha + w x J w`` with the nominal
  inertia tensor ``J``. Because ``J`` multiplies the output, the gains are inertia-independent
  bandwidths and the torque scale is derived from the CAD inertia.
* Anti-windup: the integrator is clamped to ``+-i_limit`` and frozen while the mixer reports
  saturation (conditional integration).

Gains: ``configs/control/pid.yaml`` (rationale in the file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.control.base import NominalVehicle
from dronez.control.baseline import BaselineController
from dronez.control.command_map import CommandMap
from dronez.control.position import PositionGains
from dronez.utils import rotations
from dronez.utils.config import load_model

F64 = NDArray[np.float64]
Vec3 = Annotated[tuple[float, float, float], Field()]


class AttitudeGains(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kp: tuple[float, float, float] = (8.0, 8.0, 4.0)  # [1/s] roll, pitch, yaw
    max_rate_rad_s: tuple[float, float, float] = (8.0, 8.0, 4.0)


class RateGains(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kp: tuple[float, float, float] = (30.0, 30.0, 15.0)  # [1/s]
    ki: tuple[float, float, float] = (10.0, 10.0, 5.0)  # [1/s^2]
    kd: tuple[float, float, float] = (0.0, 0.0, 0.0)  # [-] (angular acceleration feedback)
    i_limit: tuple[float, float, float] = (20.0, 20.0, 10.0)  # [rad/s^2]
    d_cutoff_hz: float = Field(default=25.0, gt=0.0)


class PidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attitude: AttitudeGains = AttitudeGains()
    rate: RateGains = RateGains()
    position: PositionGains = PositionGains()


def load_pid_config(name_or_path: str | Path = "pid") -> PidConfig:
    return load_model(PidConfig, name_or_path, "control")


class RateController:
    """PID on body rates with anti-windup and D-on-measurement; outputs a body torque [N m]."""

    def __init__(self, inertia: F64, gains: RateGains, dt: float) -> None:
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.g = gains
        self.dt = dt
        self.kp = np.array(gains.kp)
        self.ki = np.array(gains.ki)
        self.kd = np.array(gains.kd)
        self.i_limit = np.array(gains.i_limit)
        # one-pole low-pass on the measurement before differentiating
        self._alpha = dt / (dt + 1.0 / (2.0 * np.pi * gains.d_cutoff_hz))
        self.integral: F64 = np.zeros(3)
        self._w_f: F64 | None = None

    def reset(self) -> None:
        self.integral = np.zeros(3)
        self._w_f = None

    def update(self, omega_sp: F64, omega: F64, freeze_integral: bool = False) -> F64:
        """One step. ``freeze_integral`` holds the integrator (saturated actuators)."""
        err = omega_sp - omega
        if self._w_f is None:
            w_prev = w_f = np.asarray(omega, dtype=np.float64).copy()
        else:
            w_prev = self._w_f
            w_f = w_prev + self._alpha * (omega - w_prev)
        self._w_f = w_f
        d_meas = -(w_f - w_prev) / self.dt  # derivative of -omega: opposes measured acceleration
        if not freeze_integral:
            self.integral = np.clip(
                self.integral + self.ki * err * self.dt, -self.i_limit, self.i_limit
            )
        alpha_cmd = self.kp * err + self.integral + self.kd * d_meas
        return np.asarray(self.inertia @ alpha_cmd + np.cross(omega, self.inertia @ omega))


class CascadedPID(BaselineController):
    """Attitude P -> rate PID -> mixer (plus the optional position loop of the base class)."""

    def __init__(
        self,
        vehicle: NominalVehicle,
        config: PidConfig | None = None,
        dt: float = 0.01,
        cmap: CommandMap | None = None,
    ) -> None:
        cfg = config if config is not None else load_pid_config()
        super().__init__(vehicle, cfg.position, dt, cmap)
        self.cfg = cfg
        self.rate = RateController(vehicle.inertia_kg_m2, cfg.rate, dt)
        self._kp_att = np.array(cfg.attitude.kp)
        self._max_rate = np.array(cfg.attitude.max_rate_rad_s)

    def _reset_inner(self) -> None:
        self.rate.reset()

    def attitude_error_rotvec(self, q: F64, q_d: F64) -> F64:
        """Geodesic error as a rotation vector [rad], body frame (shortest path)."""
        axis, angle = rotations.quat_to_axis_angle(rotations.quat_error(q, q_d))
        return np.asarray(axis * angle)

    def _torque(self, q: F64, gyro: F64, r_d: F64) -> F64:
        q_d = rotations.rotmat_to_quat(r_d)
        e = self.attitude_error_rotvec(q, q_d)
        omega_sp = np.clip(-self._kp_att * e, -self._max_rate, self._max_rate)
        return self.rate.update(omega_sp, gyro, freeze_integral=self.last_mix.saturated)
