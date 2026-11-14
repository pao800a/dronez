"""Geometric SO(3) attitude controller (Lee, Leok, McClamroch, CDC 2010) with a hover outer loop.

Attitude errors on SO(3) (no Euler angles, no quaternion unwinding):

    e_R     = 1/2 vee(R_d^T R - R^T R_d)           (= sin(theta) * axis for a rotation theta)
    e_Omega = Omega - R^T R_d Omega_d               (Omega_d = 0 here)
    tau     = -k_R e_R - k_Omega e_Omega + Omega x J Omega
              - J (hat(Omega) R^T R_d Omega_d - R^T R_d dOmega_d)   (zero for a constant target)

The gains are inertia-scaled, ``k_R = J diag(wn^2)`` and ``k_Omega = J diag(2 zeta wn)``, i.e. a
critically damped second-order response per axis designed from the nominal inertia
(``configs/control/geometric.yaml``). The position loop and collective thrust are shared with the
PID baseline (``BaselineController``); ``tau`` goes through the same ``Mixer``.
"""

from __future__ import annotations

from pathlib import Path

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


class GeometricAttitudeGains(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    wn_rad_s: tuple[float, float, float] = (14.0, 14.0, 7.0)  # natural frequency per axis
    zeta: tuple[float, float, float] = (0.9, 0.9, 0.9)  # damping ratio per axis


class GeometricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attitude: GeometricAttitudeGains = GeometricAttitudeGains()
    position: PositionGains = Field(default_factory=PositionGains)


def load_geometric_config(name_or_path: str | Path = "geometric") -> GeometricConfig:
    return load_model(GeometricConfig, name_or_path, "control")


def vee(m: F64) -> F64:
    """Inverse of the hat map for a skew-symmetric 3x3 matrix."""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def hat(v: F64) -> F64:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class GeometricSO3(BaselineController):
    """Lee et al. SO(3) attitude controller plus the shared position loop."""

    def __init__(
        self,
        vehicle: NominalVehicle,
        config: GeometricConfig | None = None,
        dt: float = 0.01,
        cmap: CommandMap | None = None,
    ) -> None:
        cfg = config if config is not None else load_geometric_config()
        super().__init__(vehicle, cfg.position, dt, cmap)
        self.cfg = cfg
        wn = np.array(cfg.attitude.wn_rad_s)
        zeta = np.array(cfg.attitude.zeta)
        j = vehicle.inertia_kg_m2
        self.k_r = j @ np.diag(wn**2)
        self.k_omega = j @ np.diag(2.0 * zeta * wn)

    def _reset_inner(self) -> None:
        return

    def attitude_errors(self, q: F64, gyro: F64, r_d: F64) -> tuple[F64, F64]:
        """``(e_R, e_Omega)`` for a constant target attitude (``Omega_d = 0``)."""
        r = rotations.quat_to_rotmat(q)
        e_r = 0.5 * vee(r_d.T @ r - r.T @ r_d)
        return e_r, gyro.copy()

    def _torque(self, q: F64, gyro: F64, r_d: F64) -> F64:
        e_r, e_w = self.attitude_errors(q, gyro, r_d)
        j = self._inertia
        return np.asarray(-self.k_r @ e_r - self.k_omega @ e_w + np.cross(gyro, j @ gyro))
