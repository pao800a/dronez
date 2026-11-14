"""Parametric parts: pure functions of the validated :class:`DroneConfig`.

Every builder returns a list of placed solids (one per physical instance) in the CAD frame C
(FLU axes, xy origin at the hub centroid, z = 0 at the top of the bottom plate). build123d works in
millimetres: lengths are converted from the config's metres exactly once, in :func:`mm`; the
inverse conversion (mm -> m) lives in ``mass_props`` and ``export``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from build123d import Align, Box, Cylinder, Pos, RectangleRounded, Rot, Shape, extrude

from dronez.cad.schema import DroneConfig

MM_PER_M = 1000.0

AnyShape = Shape[Any]

_MIN_Z = (Align.CENTER, Align.CENTER, Align.MIN)
_MIN_XZ = (Align.MIN, Align.CENTER, Align.MIN)
_MIN_X = (Align.MIN, Align.CENTER, Align.CENTER)


def mm(value_m: float) -> float:
    """Config metres -> build123d millimetres (the single boundary conversion)."""
    return value_m * MM_PER_M


def _pos(x: float, y: float, z: float) -> Pos:
    return Pos(mm(x), mm(y), mm(z))


def _plate(spec_size: tuple[float, float], thickness: float, radius: float) -> AnyShape:
    """Rounded plate occupying z in [0, thickness], centred in xy."""
    sx, sy = mm(spec_size[0]), mm(spec_size[1])
    if radius <= 0.0:
        return Box(sx, sy, mm(thickness), align=_MIN_Z)
    return extrude(RectangleRounded(sx, sy, mm(radius)), amount=mm(thickness))


def _cyl(diameter: float, height: float) -> AnyShape:
    """Cylinder on z in [0, height], axis z, centred in xy."""
    return Cylinder(mm(diameter) / 2.0, mm(height), align=_MIN_Z)


def _tube(d_out: float, d_in: float, height: float) -> AnyShape:
    return _cyl(d_out, height) - _cyl(d_in, height)


def bottom_plate(cfg: DroneConfig) -> list[AnyShape]:
    g = cfg.geometry.bottom_plate
    return [_pos(0, 0, -g.thickness_m) * _plate(g.size_m, g.thickness_m, g.corner_radius_m)]


def top_plate(cfg: DroneConfig) -> list[AnyShape]:
    g = cfg.geometry.top_plate
    z = cfg.geometry.standoffs.height_m
    return [_pos(0, 0, z) * _plate(g.size_m, g.thickness_m, g.corner_radius_m)]


def arms(cfg: DroneConfig) -> list[AnyShape]:
    """One arm per motor: a bar along the diagonal with a round tip under the motor base."""
    a = cfg.geometry.arms
    out: list[AnyShape] = []
    for x, y, _ in cfg.motor_positions_c_m:
        s_hub = math.hypot(x, y)
        length = s_hub - a.inner_radius_m
        bar = Box(mm(length), mm(a.width_m), mm(a.thickness_m), align=_MIN_XZ)  # from local x = 0
        tip = _pos(length, 0, 0) * _cyl(a.width_m, a.thickness_m)
        yaw = math.degrees(math.atan2(y, x))
        out.append(Rot(0, 0, yaw) * _pos(a.inner_radius_m, 0, 0) * (bar + tip))
    return out


def standoffs(cfg: DroneConfig) -> list[AnyShape]:
    s = cfg.geometry.standoffs
    out: list[AnyShape] = []
    for sx in (1, -1):
        for sy in (1, -1):
            out.append(
                _pos(sx * s.x_m, sy * s.y_m, 0)
                * _tube(s.outer_diameter_m, s.inner_diameter_m, s.height_m)
            )
    return out


def _motor_part(cfg: DroneConfig, build: Callable[[DroneConfig], AnyShape]) -> list[AnyShape]:
    """Place a motor sub-part (built in the motor frame, origin at the base bottom) per motor."""
    z0 = cfg.geometry.arms.thickness_m
    return [_pos(x, y, z0) * build(cfg) for x, y, _ in cfg.motor_positions_c_m]


def _bell_bottom(cfg: DroneConfig) -> float:
    return cfg.motor.base.height_m + cfg.motor.bell.gap_above_base_m


def _motor_base(cfg: DroneConfig) -> AnyShape:
    return _cyl(cfg.motor.base.diameter_m, cfg.motor.base.height_m)


def _motor_stator(cfg: DroneConfig) -> AnyShape:
    s = cfg.motor.stator
    return _pos(0, 0, cfg.motor.base.height_m) * _tube(
        s.outer_diameter_m, s.inner_diameter_m, s.height_m
    )


def _shaft_length(cfg: DroneConfig) -> float:
    top = _bell_bottom(cfg) + cfg.motor.bell.height_m + cfg.prop.hub_height_m
    return top + cfg.motor.shaft.protrusion_above_hub_m - cfg.motor.base.height_m


def _motor_shaft(cfg: DroneConfig) -> AnyShape:
    return _pos(0, 0, cfg.motor.base.height_m) * _cyl(
        cfg.motor.shaft.diameter_m, _shaft_length(cfg)
    )


def _motor_bell(cfg: DroneConfig) -> AnyShape:
    b = cfg.motor.bell
    zb = _bell_bottom(cfg)
    inner_d = b.outer_diameter_m - 2.0 * b.wall_m
    cup = _cyl(b.outer_diameter_m, b.height_m) - _cyl(inner_d, b.height_m - b.top_thickness_m)
    cup = cup - _cyl(cfg.motor.shaft.diameter_m, b.height_m)
    return _pos(0, 0, zb) * cup


def _motor_magnets(cfg: DroneConfig) -> AnyShape:
    b, m = cfg.motor.bell, cfg.motor.magnets
    d_out = b.outer_diameter_m - 2.0 * b.wall_m
    return _pos(0, 0, _bell_bottom(cfg)) * _tube(d_out, d_out - 2.0 * m.thickness_m, m.height_m)


def motor_base(cfg: DroneConfig) -> list[AnyShape]:
    return _motor_part(cfg, _motor_base)


def motor_stator(cfg: DroneConfig) -> list[AnyShape]:
    return _motor_part(cfg, _motor_stator)


def motor_bell(cfg: DroneConfig) -> list[AnyShape]:
    return _motor_part(cfg, _motor_bell)


def motor_magnets(cfg: DroneConfig) -> list[AnyShape]:
    return _motor_part(cfg, _motor_magnets)


def motor_shaft(cfg: DroneConfig) -> list[AnyShape]:
    return _motor_part(cfg, _motor_shaft)


def _prop(cfg: DroneConfig, spin_sign: int) -> AnyShape:
    """One prop in its own frame: mid-plane at z = 0, hub on the spin axis."""
    p = cfg.prop
    radius = p.diameter_m / 2.0
    pitch_angle = math.degrees(math.atan2(p.pitch_m, 2.0 * math.pi * 0.7 * radius))
    length = radius - p.blade_root_radius_m
    hub = Cylinder(mm(p.hub_diameter_m) / 2.0, mm(p.hub_height_m))
    blade = Box(mm(length), mm(p.blade_chord_m), mm(p.blade_thickness_m), align=_MIN_X)
    # Leading edge (+y for CCW) is raised so that the rotor pushes air down: tilt sign follows spin.
    blade = Rot(spin_sign * pitch_angle, 0, 0) * blade
    blade = _pos(p.blade_root_radius_m, 0, 0) * blade
    prop = hub
    for k in range(p.blade_count):
        prop = prop + Rot(0, 0, 360.0 * k / p.blade_count) * blade
    shaft = Cylinder(mm(cfg.motor.shaft.diameter_m) / 2.0, mm(p.hub_height_m))
    return prop - shaft


def props(cfg: DroneConfig) -> list[AnyShape]:
    return [
        _pos(x, y, z) * _prop(cfg, s)
        for (x, y, z), s in zip(cfg.motor_positions_c_m, cfg.spin_signs, strict=True)
    ]


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> AnyShape:
    return _pos(*center) * Box(mm(size[0]), mm(size[1]), mm(size[2]))


def _block_builder(name: str) -> Callable[[DroneConfig], list[AnyShape]]:
    def build(cfg: DroneConfig) -> list[AnyShape]:
        spec = cfg.geometry.blocks[name]
        return [_box(spec.size_m, spec.center_m)]

    return build


def camera(cfg: DroneConfig) -> list[AnyShape]:
    c = cfg.geometry.camera
    return [_box(c.size_m, c.center_m)]


def camera_mount(cfg: DroneConfig) -> list[AnyShape]:
    """TPU bracket: a base under the camera and two side cheeks."""
    c = cfg.geometry.camera
    m = c.mount
    cx, cy, cz = c.center_m
    sx, sy, sz = c.size_m
    z_cam_bottom = cz - sz / 2.0
    z_base_bottom = z_cam_bottom - m.base_thickness_m
    base = _pos(cx, cy, z_base_bottom) * Box(
        mm(sx), mm(sy + 2.0 * m.cheek_thickness_m), mm(m.base_thickness_m), align=_MIN_Z
    )
    cheek_h = m.base_thickness_m + sz + m.cheek_extra_height_m
    shape = base
    for side in (1, -1):
        y = cy + side * (sy / 2.0 + m.cheek_thickness_m / 2.0)
        shape = shape + _pos(cx, y, z_base_bottom) * Box(
            mm(sx), mm(m.cheek_thickness_m), mm(cheek_h), align=_MIN_Z
        )
    return [shape]


def battery(cfg: DroneConfig) -> list[AnyShape]:
    return [_box(cfg.battery.dimensions_m, cfg.battery.position_m)]


def battery_strap(cfg: DroneConfig) -> list[AnyShape]:
    """Straps looping around the top plate and the battery (rings in the y-z plane)."""
    s = cfg.geometry.strap
    top = cfg.geometry.top_plate
    bat = cfg.battery
    inner_y = max(top.size_m[1], bat.dimensions_m[1])
    z_lo = cfg.standoff_top_z_m
    z_hi = bat.position_m[2] + bat.dimensions_m[2] / 2.0
    inner_z = z_hi - z_lo
    t = s.thickness_m
    out: list[AnyShape] = []
    for x in s.x_positions_m:
        outer = Box(mm(s.width_m), mm(inner_y + 2 * t), mm(inner_z + 2 * t))
        inner = Box(mm(s.width_m), mm(inner_y), mm(inner_z))
        out.append(_pos(x, 0, (z_lo + z_hi) / 2.0) * (outer - inner))
    return out


_FIXED_BUILDERS: dict[str, Callable[[DroneConfig], list[AnyShape]]] = {
    "bottom_plate": bottom_plate,
    "top_plate": top_plate,
    "arms": arms,
    "standoffs": standoffs,
    "motor_base": motor_base,
    "motor_stator": motor_stator,
    "motor_bell": motor_bell,
    "motor_magnets": motor_magnets,
    "motor_shaft": motor_shaft,
    "props": props,
    "camera": camera,
    "camera_mount": camera_mount,
    "battery": battery,
    "battery_strap": battery_strap,
}


def part_builders(cfg: DroneConfig) -> dict[str, Callable[[DroneConfig], list[AnyShape]]]:
    """Part name -> builder, in a deterministic order (fixed parts, then config blocks)."""
    builders = dict(_FIXED_BUILDERS)
    for name in cfg.geometry.blocks:
        if name in builders:
            raise ValueError(f"block name {name!r} collides with a built-in part")
        builders[name] = _block_builder(name)
    return builders
