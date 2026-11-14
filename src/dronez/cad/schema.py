"""Pydantic schemas of the drone config and of ``mass_props.json`` (no CAD kernel imports).

Every length is in metres, given in the CAD frame C: FLU axes, xy origin at the centroid of the
motor hubs, z = 0 at the top face of the bottom plate. See ``docs/architecture.md`` §3 and §4.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronez.utils.config import load_model, stable_hash

# Bump whenever a change to the CAD code alters any artifact. ``tests/unit/test_cad_version.py``
# fingerprints ``src/dronez/cad/*.py`` and fails when the code changed without a bump.
CAD_VERSION = "1"

MASS_PROPS_SCHEMA_VERSION = 1

Vec3 = Annotated[tuple[float, float, float], Field(description="metres, frame C")]
Vec2 = tuple[float, float]
PosFloat = Annotated[float, Field(gt=0.0)]
Spin = Literal["ccw", "cw"]

#: Quad-X motor table, PX4 order: (name, sign of x, sign of y, spin). conventions.md is the source.
QUAD_X_LAYOUT: tuple[tuple[str, int, int, Spin], ...] = (
    ("FR", +1, -1, "ccw"),
    ("RL", -1, +1, "ccw"),
    ("FL", +1, +1, "cw"),
    ("RR", -1, -1, "cw"),
)

_TOL = 1e-6


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Material(_Frozen):
    density_kg_m3: PosFloat
    source: str = Field(min_length=1)


class MotorSpec(_Frozen):
    name: str
    position_m: Vec3  # hub / prop plane, frame C
    spin: Spin


class PlateSpec(_Frozen):
    size_m: Vec2
    thickness_m: PosFloat
    corner_radius_m: Annotated[float, Field(ge=0.0)] = 0.0


class ArmSpec(_Frozen):
    width_m: PosFloat
    thickness_m: PosFloat
    inner_radius_m: Annotated[float, Field(ge=0.0)]  # radial start of the arm along the diagonal


class StandoffSpec(_Frozen):
    x_m: PosFloat  # mirrored to four corners
    y_m: PosFloat
    outer_diameter_m: PosFloat
    inner_diameter_m: Annotated[float, Field(ge=0.0)]
    height_m: PosFloat


class BoxSpec(_Frozen):
    size_m: Vec3
    center_m: Vec3


class CameraMountSpec(_Frozen):
    base_thickness_m: PosFloat
    cheek_thickness_m: PosFloat
    cheek_extra_height_m: Annotated[float, Field(ge=0.0)]


class CameraSpec(BoxSpec):
    mount: CameraMountSpec


class StrapSpec(_Frozen):
    width_m: PosFloat
    thickness_m: PosFloat
    x_positions_m: tuple[float, ...]


class Geometry(_Frozen):
    min_prop_clearance_m: Annotated[float, Field(ge=0.0)]
    bottom_plate: PlateSpec
    top_plate: PlateSpec
    arms: ArmSpec
    standoffs: StandoffSpec
    imu_position_m: Vec3
    blocks: dict[str, BoxSpec]
    camera: CameraSpec
    strap: StrapSpec


class MotorBase(_Frozen):
    diameter_m: PosFloat
    height_m: PosFloat


class MotorStator(_Frozen):
    outer_diameter_m: PosFloat
    inner_diameter_m: PosFloat
    height_m: PosFloat


class MotorBell(_Frozen):
    outer_diameter_m: PosFloat
    wall_m: PosFloat
    top_thickness_m: PosFloat
    height_m: PosFloat
    gap_above_base_m: Annotated[float, Field(ge=0.0)]


class MotorMagnets(_Frozen):
    thickness_m: PosFloat
    height_m: PosFloat


class MotorShaft(_Frozen):
    diameter_m: PosFloat
    protrusion_above_hub_m: Annotated[float, Field(ge=0.0)]


class MotorGeometry(_Frozen):
    name: str
    base: MotorBase
    stator: MotorStator
    bell: MotorBell
    magnets: MotorMagnets
    shaft: MotorShaft


class PropGeometry(_Frozen):
    diameter_m: PosFloat
    pitch_m: PosFloat
    blade_count: Annotated[int, Field(ge=2)]
    hub_diameter_m: PosFloat
    hub_height_m: PosFloat
    blade_root_radius_m: PosFloat
    blade_chord_m: PosFloat
    blade_thickness_m: PosFloat


class Propulsion(_Frozen):
    k_f: PosFloat  # N/(rad/s)^2
    k_m: PosFloat  # N*m/(rad/s)^2
    omega_min_rad_s: Annotated[float, Field(ge=0.0)]
    omega_max_rad_s: PosFloat
    tau_m_s: PosFloat
    rotor_inertia_kg_m2: Annotated[float, Field(ge=0.0)] | None = None
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _omega_order(self) -> Propulsion:
        if self.omega_min_rad_s >= self.omega_max_rad_s:
            raise ValueError("omega_min_rad_s must be below omega_max_rad_s")
        return self


class Battery(_Frozen):
    cells: Annotated[int, Field(ge=1)]
    cell_voltage_nominal_v: PosFloat
    capacity_mah: PosFloat
    dimensions_m: Vec3
    position_m: Vec3
    material: str


class MeasuredOverrides(_Frozen):
    """Weighed values that replace the CAD ones (each set field replaces its CAD counterpart)."""

    mass_kg: PosFloat | None = None
    com_c_m: Vec3 | None = None
    inertia_com_kg_m2: tuple[Vec3, Vec3, Vec3] | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _needs_source(self) -> MeasuredOverrides:
        if self.set_fields() and not self.source:
            raise ValueError("measured_overrides: `source` is required when any value is set")
        return self

    def set_fields(self) -> list[str]:
        return [
            f for f in ("mass_kg", "com_c_m", "inertia_com_kg_m2") if getattr(self, f) is not None
        ]


class CadSettings(_Frozen):
    mesh_tolerance_m: PosFloat
    mesh_angular_tolerance_rad: PosFloat
    max_triangles_per_part: Annotated[int, Field(ge=12)]


class DroneConfig(_Frozen):
    name: str
    layout: Literal["quad_x"]
    geometry: Geometry
    materials: dict[str, Material]
    parts_material: dict[str, str]
    motors: list[MotorSpec]
    motor: MotorGeometry
    prop: PropGeometry
    propulsion: Propulsion
    battery: Battery
    target_auw_kg: PosFloat
    measured_overrides: MeasuredOverrides = MeasuredOverrides()
    cad: CadSettings

    # -- derived quantities (never typed by hand) -------------------------------------------
    @property
    def motor_positions_c_m(self) -> list[tuple[float, float, float]]:
        return [m.position_m for m in self.motors]

    @property
    def spin_signs(self) -> list[int]:
        """``s_i``: +1 for CCW (seen from above), -1 for CW. Order follows ``motors``."""
        return [1 if m.spin == "ccw" else -1 for m in self.motors]

    @property
    def hub_radius_m(self) -> float:
        """Distance from the hub centroid to each hub (a * sqrt(2))."""
        x, y, _ = self.motors[0].position_m
        return math.hypot(x, y)

    @property
    def prop_plane_height_m(self) -> float:
        """Height of the prop mid-plane above the bottom of the motor base."""
        m, p = self.motor, self.prop
        return m.base.height_m + m.bell.gap_above_base_m + m.bell.height_m + p.hub_height_m / 2.0

    @property
    def standoff_top_z_m(self) -> float:
        return self.geometry.standoffs.height_m

    # -- validation -------------------------------------------------------------------------
    @model_validator(mode="after")
    def _check(self) -> DroneConfig:
        self._check_motors()
        self._check_materials()
        self._check_geometry()
        return self

    def _check_motors(self) -> None:
        if len(self.motors) != len(QUAD_X_LAYOUT):
            raise ValueError(f"quad_x needs {len(QUAD_X_LAYOUT)} motors, got {len(self.motors)}")
        radii = set()
        zs = set()
        for i, (m, (name, sx, sy, spin)) in enumerate(zip(self.motors, QUAD_X_LAYOUT, strict=True)):
            x, y, z = m.position_m
            if m.name != name or m.spin != spin:
                raise ValueError(
                    f"motor M{i} must be {name} spinning {spin} (PX4 Quad-X order); "
                    f"got {m.name} {m.spin}"
                )
            if x * sx <= 0 or y * sy <= 0:
                raise ValueError(f"motor M{i} ({name}) is in the wrong quadrant: {m.position_m}")
            if abs(abs(x) - abs(y)) > _TOL:
                raise ValueError(f"motor M{i}: |x| != |y| (hubs must sit on the diagonals)")
            radii.add(round(abs(x), 6))
            zs.add(round(z, 6))
        if len(radii) != 1 or len(zs) != 1:
            raise ValueError("all hubs must share |x|=|y| and the same z")

    def _check_materials(self) -> None:
        for part, key in self.parts_material.items():
            if key not in self.materials:
                raise ValueError(f"parts_material[{part!r}] -> unknown material {key!r}")
        if self.battery.material not in self.materials:
            raise ValueError(f"battery.material {self.battery.material!r} is not in materials")
        if self.parts_material.get("battery") != self.battery.material:
            raise ValueError("parts_material['battery'] must equal battery.material")

    def _check_geometry(self) -> None:
        arm_top = self.geometry.arms.thickness_m
        hub_z = self.motors[0].position_m[2]
        if abs(hub_z - (arm_top + self.prop_plane_height_m)) > 1e-5:
            raise ValueError(
                f"motor z = {hub_z} but arm top + prop plane height = "
                f"{arm_top + self.prop_plane_height_m:.6f}"
            )
        if self.geometry.arms.inner_radius_m >= self.hub_radius_m:
            raise ValueError("arms.inner_radius_m must be smaller than the hub radius")
        gap = 2.0 * abs(self.motors[0].position_m[0]) - self.prop.diameter_m
        if gap < self.geometry.min_prop_clearance_m:
            raise ValueError(
                f"adjacent props are {gap * 1e3:.1f} mm apart, below "
                f"min_prop_clearance_m = {self.geometry.min_prop_clearance_m * 1e3:.1f} mm"
            )
        st = self.motor.stator
        if st.inner_diameter_m >= st.outer_diameter_m:
            raise ValueError("motor.stator.inner_diameter_m must be below the outer diameter")
        if st.inner_diameter_m <= self.motor.shaft.diameter_m:
            raise ValueError("motor shaft must fit through the stator bore")
        if self.motor.stator.outer_diameter_m >= (
            self.motor.bell.outer_diameter_m
            - 2.0 * self.motor.bell.wall_m
            - 2.0 * self.motor.magnets.thickness_m
        ):
            raise ValueError("stator does not fit inside the bell + magnets")
        if self.geometry.standoffs.inner_diameter_m >= self.geometry.standoffs.outer_diameter_m:
            raise ValueError("standoff inner diameter must be below the outer diameter")


class PartMass(_Frozen):
    name: str
    material: str
    density_kg_m3: float
    volume_m3: float
    mass_kg: float
    count: int
    com_c_m: Vec3


class MassProperties(_Frozen):
    """Contents of ``mass_props.json`` (``docs/architecture.md`` §4)."""

    schema_version: int = MASS_PROPS_SCHEMA_VERSION
    drone: str
    model_hash: str
    cad_version: str
    units: dict[str, str] = {"length": "m", "mass": "kg", "inertia": "kg*m^2"}
    frame: str = (
        "axes: body FLU; com_c_m in CAD frame C; inertia about the CoM; "
        "principal_axes_quat_wxyz rotates principal axes into body axes"
    )
    mass_kg: float
    com_c_m: Vec3
    inertia_com_kg_m2: tuple[Vec3, Vec3, Vec3]
    principal_moments_kg_m2: Vec3
    principal_axes_quat_wxyz: tuple[float, float, float, float]
    parts: list[PartMass]
    motor_hubs_b_m: tuple[Vec3, Vec3, Vec3, Vec3]
    target_auw_kg: float
    auw_error_frac: float
    overrides_applied: list[str]
    source: Literal["cad", "cad+measured"]


def load_drone_config(name_or_path: str | Path) -> DroneConfig:
    return load_model(DroneConfig, name_or_path, "drone")


def model_hash(cfg: DroneConfig) -> str:
    """10-hex-character hash of the config and ``CAD_VERSION`` (the cache key)."""
    return stable_hash({"drone": cfg.model_dump(mode="json"), "cad_version": CAD_VERSION})
