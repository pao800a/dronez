"""``SimConfig``: physics rate, integrator, ground plane and optional rotor effects.

All values are SI. The default lives in ``configs/sim/default.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from dronez.utils.config import load_model

PosFloat = Annotated[float, Field(gt=0.0)]

#: Standard gravity [m/s^2]; the single place the sim and ``hover_command`` take it from.
GRAVITY_M_S2 = 9.81


class SimConfig(BaseModel):
    """Simulation settings. Frozen: build a new one (``model_copy(update=...)``) to change it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    physics_dt_s: PosFloat = 0.001
    decimation: Annotated[int, Field(ge=1)] = 10
    integrator: Literal["euler", "implicit", "implicitfast", "rk4"] = "rk4"
    ground_plane: bool = True
    prop_disk_collision: bool = False
    rotor_drag: bool = False
    ground_effect: bool = False
    #: Rotor reaction (-sum s_i I_r domega_i/dt) and gyroscopic (-Omega x H) torques from the
    #: spinning rotors (``propulsion.rotor_inertia_kg_m2``; 0 if the YAML omits it).
    rotor_inertia: bool = True

    # -- optional effects (used only behind the flags above); order-of-magnitude placeholders ----
    #: Rotor (induced/profile) drag: F_b = -k_d * sum(omega) * v_xy_b  [N / (rad/s * m/s)].
    rotor_drag_coeff: Annotated[float, Field(ge=0.0)] = 2.0e-5
    #: Ground effect (Cheeseman-Bennett): T/T0 = 1 / (1 - (R / (4 z))^2), clipped at this value.
    ground_effect_max_gain: Annotated[float, Field(ge=1.0)] = 1.5
    #: Collision geom friction (sliding, torsional, rolling) used for all collision geoms.
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.decimation

    @property
    def policy_rate_hz(self) -> float:
        return 1.0 / self.policy_dt_s


def load_sim_config(name_or_path: str | Path = "default") -> SimConfig:
    """Load ``configs/sim/<name>.yaml`` (or an explicit path)."""
    return load_model(SimConfig, name_or_path, "sim")
