"""Simulation: MjSpec builder, motor model, sensor/disturbance models and the ``DroneSim`` plant.

May import ``utils`` and ``cad.schema`` / ``cad.artifacts`` only (never build123d).
"""

from dronez.sim.builder import (
    ModelIndex,
    NominalInertial,
    build_index,
    build_model,
    build_model_from_artifacts,
    build_spec,
    motor_gear,
)
from dronez.sim.config import GRAVITY_M_S2, SimConfig, load_sim_config
from dronez.sim.disturbances import (
    ActionDelay,
    DisturbanceConfig,
    DisturbanceModel,
    ImpulseSpec,
)
from dronez.sim.motor_model import (
    MotorModel,
    MotorParams,
    ground_effect_gain,
    hover_command,
    hover_thrust_per_motor,
    rotor_drag_force_b,
)
from dronez.sim.plant import (
    DroneSim,
    PhysParams,
    inertia_triangle_ok,
    project_inertia_triangle,
)
from dronez.sim.sensors import ImuNoiseConfig, ImuNoiseModel
from dronez.sim.state import Sensed, TrueState

__all__ = [
    "GRAVITY_M_S2",
    "ActionDelay",
    "DisturbanceConfig",
    "DisturbanceModel",
    "DroneSim",
    "ImpulseSpec",
    "ImuNoiseConfig",
    "ImuNoiseModel",
    "ModelIndex",
    "MotorModel",
    "MotorParams",
    "NominalInertial",
    "PhysParams",
    "Sensed",
    "SimConfig",
    "TrueState",
    "build_index",
    "build_model",
    "build_model_from_artifacts",
    "build_spec",
    "ground_effect_gain",
    "hover_command",
    "hover_thrust_per_motor",
    "inertia_triangle_ok",
    "load_sim_config",
    "motor_gear",
    "project_inertia_triangle",
    "rotor_drag_force_b",
]
