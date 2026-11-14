"""Action <-> normalized motor command map (ADR-0006 §3).

Motors mode: the policy action ``a in [-1, 1]^4`` (motor index order) maps to the normalized
command of ``MotorModel.command_to_omega`` (which is **linear in omega**) by

    u = clip(u_h + delta_u * clip(a, -1, 1), 0, 1),    delta_u = authority * min(u_h, 1 - u_h)

with ``u_h`` the (scalar) hover command of the **nominal** motor and the **nominal** mass. The map
is built from nominal quantities only (it is never randomized), so it equals the constants baked
into the ONNX graph. ``a = 0`` is nominal hover, ``a = -1 / +1`` is ``u_h -/+ delta_u``.

``thrust_limits`` gives the per-motor thrust at ``a = -1 / +1``; the baseline mixers use it as
their saturation limits, so saturation handling respects the policy's envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dronez.sim.motor_model import MotorModel, MotorParams, hover_command

if TYPE_CHECKING:
    from dronez.control.base import NominalVehicle

F64 = NDArray[np.float64]

COMMAND_MAP_TYPE = "affine_omega"


class CommandMap:
    """``u = clip(u_h + delta_u * a, 0, 1)`` with constants from the nominal motor and mass."""

    def __init__(self, motor: MotorParams, mass_kg: float, authority: float = 1.0) -> None:
        if not 0.0 < authority <= 1.0:
            raise ValueError("authority must be in (0, 1]")
        self.motor = motor.copy()
        self.mass_kg = float(mass_kg)
        self.authority = float(authority)
        u_h = float(np.mean(hover_command(self.motor, self.mass_kg)))
        if not 0.0 < u_h < 1.0:
            raise ValueError(f"hover command {u_h} is outside (0, 1): the drone cannot hover")
        self.u_hover = u_h
        self.delta_u = self.authority * min(u_h, 1.0 - u_h)

    @classmethod
    def from_vehicle(cls, vehicle: NominalVehicle, authority: float = 1.0) -> CommandMap:
        return cls(vehicle.motor, vehicle.mass_kg, authority)

    # -- maps ------------------------------------------------------------------------------------
    def action_to_command(self, a: ArrayLike) -> F64:
        """Action (clipped to [-1, 1]) -> normalized command ``u in [0, 1]``."""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        return np.asarray(np.clip(self.u_hover + self.delta_u * a, 0.0, 1.0))

    def command_to_action(self, u: ArrayLike) -> F64:
        """Inverse of ``action_to_command`` on its range; result clipped to [-1, 1]."""
        u = np.asarray(u, dtype=np.float64)
        return np.asarray(np.clip((u - self.u_hover) / self.delta_u, -1.0, 1.0))

    def thrust_limits(self, motor: MotorParams | None = None) -> tuple[F64, F64]:
        """Per-motor thrust ``(T_lo, T_hi)`` [N] at ``a = -1 / +1`` (nominal motor by default)."""
        m = MotorModel(self.motor if motor is None else motor)
        lo = m.command_to_omega(self.action_to_command(-np.ones(4)))
        hi = m.command_to_omega(self.action_to_command(np.ones(4)))
        k_f = np.broadcast_to(m.params.k_f, (4,))
        return (
            np.asarray(k_f * np.broadcast_to(lo, (4,)) ** 2, dtype=np.float64),
            np.asarray(k_f * np.broadcast_to(hi, (4,)) ** 2, dtype=np.float64),
        )

    def meta(self) -> dict[str, Any]:
        """The ``policy_meta.action.command_map`` entry."""
        return {
            "type": COMMAND_MAP_TYPE,
            "u_hover": self.u_hover,
            "delta_u": self.delta_u,
            "authority": self.authority,
        }
