"""Shared controller plumbing: the ``Controller`` protocol, the nominal vehicle, setpoints.

Baselines read only ``Sensed`` (attitude, gyro and the optional position/velocity estimate) and a
*nominal* vehicle description built from the drone YAML and the CAD mass properties. They never see
domain-randomized parameters or ground truth (docs/architecture.md §7).

``thrusts_to_command`` is the **single** place where per-motor thrusts become the action returned
by ``act``. Today the action is the normalized motor command ``u in [0, 1]`` (``MotorModel``
inverse); the Phase 4 env action map (``a = 2u - 1`` or similar) only has to change there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dronez.sim.config import GRAVITY_M_S2
from dronez.sim.motor_model import MotorModel, MotorParams
from dronez.sim.state import Sensed

if TYPE_CHECKING:
    from dronez.cad.artifacts import ModelArtifacts
    from dronez.cad.schema import DroneConfig

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]


@runtime_checkable
class Controller(Protocol):
    """One interface for baselines and learned policies (docs/architecture.md §7)."""

    action_mode: Literal["motors", "ctbr"]

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: NDArray[np.float32], sensed: Sensed) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class NominalVehicle:
    """Nominal (non-randomized) airframe as the controllers know it.

    ``hubs_b`` are the rotor hubs relative to the **centre of mass** (FLU, ``(4, 3)``), taken from
    the CAD mass properties: the CoM is ~0.06 mm off the hub centroid, so the mixer must not
    assume symmetric positions. ``spin`` is ``s_i`` (+1 CCW) from the YAML.
    """

    mass_kg: float
    inertia_kg_m2: F64  # (3, 3) about the CoM, FLU axes
    hubs_b: F64  # (4, 3)
    spin: F64  # (4,)
    motor: MotorParams
    gravity: float = GRAVITY_M_S2

    @classmethod
    def from_drone(cls, cfg: DroneConfig, art: ModelArtifacts | None = None) -> NominalVehicle:
        """From the YAML plus the cached CAD mass properties (``dronez-build-model`` first)."""
        if art is None:
            from dronez.cad.artifacts import load_artifacts  # noqa: PLC0415

            art = load_artifacts(cfg)
        mp = art.mass_props
        return cls(
            mass_kg=float(mp.mass_kg),
            inertia_kg_m2=np.array(mp.inertia_com_kg_m2, dtype=np.float64),
            hubs_b=np.array(mp.motor_hubs_b_m, dtype=np.float64),
            spin=np.array(cfg.spin_signs, dtype=np.float64),
            motor=MotorParams.from_drone(cfg),
        )

    @property
    def weight_n(self) -> float:
        return self.mass_kg * self.gravity

    @property
    def thrust_limits_n(self) -> tuple[F64, F64]:
        """Per-motor ``(T_min, T_max)`` [N] from ``k_f omega^2`` at the speed limits."""
        p = self.motor
        return (
            np.broadcast_to(p.k_f * p.omega_min**2, (4,)).astype(np.float64),
            np.broadcast_to(p.k_f * p.omega_max**2, (4,)).astype(np.float64),
        )


@dataclass
class Setpoint:
    """What the controller is asked to track (world frame, SI).

    * ``pos_w is None``: no position loop; collective thrust holds ``m g`` along the target
      attitude's vertical component (no altitude feedback).
    * ``pos_w`` set: altitude PD(+I) always; horizontal PD only if ``hold_xy``.
    * ``quat_wxyz`` set: explicit attitude target (overrides the tilt the position loop would
      command); otherwise the attitude comes from the thrust vector and ``yaw``.
    """

    pos_w: F64 | None = None
    vel_w: F64 = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    quat_wxyz: F64 | None = None
    hold_xy: bool = True


def thrusts_to_command(thrusts: F64, motor: MotorModel) -> F32:
    """Per-motor thrust [N] -> the action returned by ``act`` (normalized command in [0, 1])."""
    return np.asarray(motor.thrust_to_command(thrusts), dtype=np.float32)
