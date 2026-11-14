"""Mass properties from the solids: volume x density, CoM, inertia about the CoM (SI).

build123d reports volume in mm^3, centre of mass in mm and ``matrix_of_inertia`` in mm^5: the
inertia of the solid at *unit density* about its own CoM, in axes parallel to the world axes. The
conversion to SI happens once, here: length x1e-3, volume x1e-9, inertia x rho x1e-15.
"""

from __future__ import annotations

import itertools

import numpy as np
from build123d import CenterOf
from numpy.typing import NDArray

from dronez.cad.assembly import Assembly
from dronez.cad.parts import MM_PER_M, AnyShape
from dronez.cad.schema import (
    CAD_VERSION,
    DroneConfig,
    MassProperties,
    PartMass,
    model_hash,
)
from dronez.utils.rotations import quat_canonical, rotmat_to_quat

F64 = NDArray[np.float64]

_VOLUME_FROM_MM3 = MM_PER_M**-3
_INERTIA_UNIT_DENSITY_FROM_MM5 = MM_PER_M**-5  # (mm^5 -> m^5), multiplied by rho in kg/m^3


def solid_properties(shape: AnyShape, density_kg_m3: float) -> tuple[float, F64, F64]:
    """(mass [kg], CoM [m], inertia about its own CoM [kg*m^2]) of one solid, world axes."""
    volume_m3 = shape.volume * _VOLUME_FROM_MM3
    com = shape.center(CenterOf.MASS)
    com_m = np.array([com.X, com.Y, com.Z], dtype=np.float64) / MM_PER_M
    inertia = (
        np.array(shape.matrix_of_inertia, dtype=np.float64)
        * density_kg_m3
        * _INERTIA_UNIT_DENSITY_FROM_MM5
    )
    return density_kg_m3 * volume_m3, com_m, inertia


def parallel_axis(mass: float, d: F64) -> F64:
    """Inertia tensor of a point mass at offset ``d`` (to add to the inertia about its own CoM)."""
    return mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def principal(inertia: F64) -> tuple[F64, F64]:
    """Principal moments and a right-handed matrix whose columns are the principal axes.

    The axes are ordered by the body axis each is closest to (column ``i`` is the principal axis
    most aligned with body axis ``i``, with a non-negative ``i``-th component for ``i = 0, 1``),
    so ``moments[i]`` is the moment about (nearly) body axis ``i`` whatever the ordering of the
    moments; the rotation is proper (``det = +1``).
    """
    w, v = np.linalg.eigh(0.5 * (inertia + inertia.T))
    best = max(
        itertools.permutations(range(3)), key=lambda p: sum(abs(v[i, p[i]]) for i in range(3))
    )
    order = list(best)
    w, v = w[order], v[:, order]
    for j in range(2):
        if v[j, j] < 0.0:
            v[:, j] = -v[:, j]
    v[:, 2] = np.cross(v[:, 0], v[:, 1])
    return w, v


def aggregate(
    items: list[tuple[float, F64, F64]],
) -> tuple[float, F64, F64]:
    """Total mass, CoM and inertia about the total CoM from per-solid (m, com, I_com)."""
    mass = sum(m for m, _, _ in items)
    com = np.sum([m * c for m, c, _ in items], axis=0) / mass
    inertia = np.zeros((3, 3))
    for m, c, i_own in items:
        inertia += i_own + parallel_axis(m, c - com)
    return float(mass), np.asarray(com, dtype=np.float64), inertia


def _tuple3(v: F64) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def compute_mass_properties(asm: Assembly, cfg: DroneConfig) -> MassProperties:
    """CAD mass properties with ``measured_overrides`` applied last."""
    all_items: list[tuple[float, F64, F64]] = []
    part_rows: list[PartMass] = []
    for part in asm.parts.values():
        items = [solid_properties(s, part.density_kg_m3) for s in part.solids]
        all_items.extend(items)
        p_mass = sum(m for m, _, _ in items)
        p_com = np.sum([m * c for m, c, _ in items], axis=0) / p_mass
        part_rows.append(
            PartMass(
                name=part.name,
                material=part.material,
                density_kg_m3=part.density_kg_m3,
                volume_m3=p_mass / part.density_kg_m3,
                mass_kg=float(p_mass),
                count=part.count,
                com_c_m=_tuple3(p_com),
            )
        )
    mass, com, inertia = aggregate(all_items)

    ov = cfg.measured_overrides
    applied = ov.set_fields()
    if ov.mass_kg is not None:
        if ov.inertia_com_kg_m2 is None:
            inertia = inertia * (ov.mass_kg / mass)
        mass = ov.mass_kg
    if ov.com_c_m is not None:
        com = np.array(ov.com_c_m, dtype=np.float64)
    if ov.inertia_com_kg_m2 is not None:
        inertia = np.array(ov.inertia_com_kg_m2, dtype=np.float64)

    moments, axes = principal(inertia)
    quat = quat_canonical(rotmat_to_quat(axes))
    hubs = [_tuple3(np.array(p) - com) for p in cfg.motor_positions_c_m]
    return MassProperties(
        drone=cfg.name,
        model_hash=model_hash(cfg),
        cad_version=CAD_VERSION,
        mass_kg=mass,
        com_c_m=_tuple3(com),
        inertia_com_kg_m2=(_tuple3(inertia[0]), _tuple3(inertia[1]), _tuple3(inertia[2])),
        principal_moments_kg_m2=_tuple3(moments),
        principal_axes_quat_wxyz=(
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ),
        parts=part_rows,
        motor_hubs_b_m=(hubs[0], hubs[1], hubs[2], hubs[3]),
        target_auw_kg=cfg.target_auw_kg,
        auw_error_frac=(mass - cfg.target_auw_kg) / cfg.target_auw_kg,
        overrides_applied=applied,
        source="cad+measured" if applied else "cad",
    )
