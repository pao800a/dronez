"""Build the MuJoCo model through ``mujoco.MjSpec`` (never hand-written XML).

MuJoCo 4.0.1 facts verified by introspection while writing this module (see PROGRESS / report):

* ``body.explicitinertial = True`` plus ``compiler.inertiafromgeom = FALSE`` makes the explicit
  ``mass`` / ``inertia`` (principal moments) / ``iquat`` / ``ipos`` the only inertial source.
* ``body.simple = 0`` is **required** because runtime edits of ``body_ipos`` on a body whose
  inertial frame equals the body frame make ``mj_setConst`` raise ("compiled as simple").
* ``mj_setConst`` **is required** after editing ``body_mass`` / ``body_inertia`` / ``body_ipos``
  (``body_subtreemass``, ``body_invweight0``... are stale otherwise; trajectories then differ from
  a fresh compile, with ``mj_setConst`` they match to the last bit).
* Site actuators: ``trntype = mjTRN_SITE``, ``gear = [fx fy fz tx ty tz]`` in the site frame; the
  defaults are gain 1, bias 0, no dynamics; ``ctrllimited`` is set to FALSE explicitly.
* ``xfrc_applied[body]`` is a world-frame force applied **at the body CoM** (``xipos``) and a
  world-frame torque. The free-joint ``qvel[3:6]`` is the angular velocity in the body frame and
  ``qvel[0:3]`` the linear velocity of the body *origin* in the world frame.
* ``accelerometer`` reports specific force: ~0 in free fall, ``+g`` (up) at rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dronez.cad.artifacts import ModelArtifacts, load_artifacts
from dronez.cad.schema import DroneConfig
from dronez.sim.config import SimConfig

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

BODY_NAME = "drone"
IMU_SITE = "imu"
GROUND_NAME = "ground"

_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
}

# visual colour by material key (cosmetic only)
_RGBA: dict[str, tuple[float, float, float, float]] = {
    "carbon_fibre": (0.12, 0.12, 0.14, 1.0),
    "aluminium": (0.7, 0.7, 0.75, 1.0),
    "stator_cu_steel": (0.55, 0.35, 0.2, 1.0),
    "steel": (0.6, 0.6, 0.6, 1.0),
    "ndfeb": (0.4, 0.4, 0.45, 1.0),
    "polycarbonate": (0.85, 0.3, 0.1, 1.0),
    "pcb_assembly": (0.1, 0.5, 0.2, 1.0),
    "lipo": (0.2, 0.3, 0.8, 1.0),
}
_RGBA_DEFAULT = (0.5, 0.5, 0.5, 1.0)


@dataclass(frozen=True)
class NominalInertial:
    """Nominal inertial snapshot written by ``DroneSim.set_params`` (never drifts across episodes).

    ``inertia`` are the principal moments in the inertial frame ``iquat`` (w, x, y, z) relative to
    the body frame; ``ipos`` is the CoM in the body frame (zero: the body frame is at the CoM).
    """

    mass: float
    inertia: F64
    iquat: F64
    ipos: F64


@dataclass(frozen=True)
class ModelIndex:
    """Ids and constants needed to drive the compiled model."""

    body_id: int
    imu_site_id: int
    motor_site_ids: I64
    actuator_ids: I64
    sensor_slices: dict[str, slice]
    spin: F64  # (4,), s_i = +1 CCW, -1 CW
    motor_pos_b: F64  # (4, 3), hubs in the body frame
    nominal: NominalInertial
    k_m_over_k_f: float  # nominal yaw-reaction ratio baked into the gear
    joint_id: int = 0
    rotor_radius_m: float = 0.0
    com_c_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # CoM in the CAD frame C


def motor_gear(spin: F64, k_m_over_k_f: float | F64) -> F64:
    """Site-actuator gear rows ``[0, 0, 1, 0, 0, -s_i k_m/k_f]``; shape ``(n, 6)``.

    ``ctrl`` is the thrust in N; MuJoCo applies the force along the site z and the yaw reaction
    torque ``-s_i (k_m/k_f) T``. ``spin`` is ``DroneConfig.spin_signs`` (from the YAML).
    """
    spin = np.asarray(spin, dtype=np.float64)
    gear = np.zeros((spin.shape[0], 6))
    gear[:, 2] = 1.0
    gear[:, 5] = -spin * np.asarray(k_m_over_k_f, dtype=np.float64)
    return gear


def _vec(v: Any) -> list[float]:
    return [float(x) for x in v]


def build_spec(cfg: DroneConfig, art: ModelArtifacts, sim: SimConfig) -> mujoco.MjSpec:
    """Assemble the ``MjSpec``. Frame B = frame C translated by the CoM (``com_c_m``)."""
    mp = art.mass_props
    com = np.asarray(mp.com_c_m, dtype=np.float64)

    spec = mujoco.MjSpec()
    spec.modelname = cfg.name
    spec.option.timestep = sim.physics_dt_s
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = _INTEGRATORS[sim.integrator]
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE

    if sim.ground_plane:
        spec.worldbody.add_geom(
            name=GROUND_NAME,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            rgba=[0.25, 0.27, 0.3, 1.0],
            friction=list(sim.friction),
            contype=1,
            conaffinity=1,
        )

    # -- body with explicit inertial ----------------------------------------------------------
    body = spec.worldbody.add_body(name=BODY_NAME, pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name="root")
    body.mass = mp.mass_kg
    body.inertia = _vec(mp.principal_moments_kg_m2)
    body.iquat = _vec(mp.principal_axes_quat_wxyz)
    body.ipos = [0.0, 0.0, 0.0]
    body.explicitinertial = True
    body.simple = 0  # see module docstring: needed for runtime ipos edits

    # -- visual meshes (no collision, no mass) ------------------------------------------------
    for part in mp.parts:
        path = art.meshes.get(part.name)
        if path is None:
            continue
        spec.add_mesh(name=part.name, file=str(path))
        body.add_geom(
            name=f"vis_{part.name}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=part.name,
            pos=_vec(-com),
            contype=0,
            conaffinity=0,
            group=1,
            density=0.0,
            rgba=list(_RGBA.get(part.material, _RGBA_DEFAULT)),
        )

    # -- primitive collision geoms (group 3) --------------------------------------------------
    _add_collision_geoms(body, cfg, sim, com)

    # -- sites, actuators ---------------------------------------------------------------------
    hubs_b = np.asarray(mp.motor_hubs_b_m, dtype=np.float64)
    body.add_site(
        name=IMU_SITE,
        pos=_vec(np.asarray(cfg.geometry.imu_position_m) - com),
        size=[0.002, 0.002, 0.002],
        rgba=[0.1, 0.9, 0.1, 1.0],
    )
    spin = np.asarray(cfg.spin_signs, dtype=np.float64)
    gear = motor_gear(spin, cfg.propulsion.k_m / cfg.propulsion.k_f)
    for i, m in enumerate(cfg.motors):
        site_name = f"m{i}"
        body.add_site(
            name=site_name,
            pos=_vec(hubs_b[i]),
            size=[0.003, 0.003, 0.003],
            rgba=[1.0, 0.2, 0.2, 1.0] if spin[i] > 0 else [0.2, 0.4, 1.0, 1.0],
        )
        act = spec.add_actuator(
            name=f"thrust_{m.name.lower()}",
            target=site_name,
            trntype=mujoco.mjtTrn.mjTRN_SITE,
        )
        act.gear = _vec(gear[i])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE

    # -- sensors (ground truth; MuJoCo sensor noise is not used) ------------------------------
    site = mujoco.mjtObj.mjOBJ_SITE
    obj_body = mujoco.mjtObj.mjOBJ_BODY
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=site, objname=IMU_SITE)
    spec.add_sensor(
        name="accelerometer",
        type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        objtype=site,
        objname=IMU_SITE,
    )
    spec.add_sensor(
        name="framequat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=site, objname=IMU_SITE
    )
    spec.add_sensor(
        name="framepos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=obj_body, objname=BODY_NAME
    )
    spec.add_sensor(
        name="framelinvel",
        type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
        objtype=obj_body,
        objname=BODY_NAME,
    )
    return spec


def _add_collision_geoms(body: mujoco.MjsBody, cfg: DroneConfig, sim: SimConfig, com: F64) -> None:
    """Boxes for plates / stack / battery / camera, one box per arm, cylinders for motors."""
    g = cfg.geometry
    friction = list(sim.friction)
    common: dict[str, Any] = {
        "contype": 1,
        "conaffinity": 1,
        "group": 3,
        "density": 0.0,
        "friction": friction,
        "rgba": [1.0, 0.0, 1.0, 0.3],
    }

    def box(
        name: str, size: tuple[float, float, float], center: tuple[float, float, float]
    ) -> None:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[s / 2.0 for s in size],
            pos=_vec(np.asarray(center) - com),
            **common,
        )

    bp, tp, st = g.bottom_plate, g.top_plate, g.standoffs
    box("col_bottom_plate", (*bp.size_m, bp.thickness_m), (0.0, 0.0, -bp.thickness_m / 2.0))
    box(
        "col_top_plate",
        (*tp.size_m, tp.thickness_m),
        (0.0, 0.0, st.height_m + tp.thickness_m / 2.0),
    )
    bat = cfg.battery
    box("col_battery", bat.dimensions_m, bat.position_m)
    box("col_camera", g.camera.size_m, g.camera.center_m)

    # arms: one box per motor along the diagonal, from inner radius to just past the hub
    r_hub = cfg.hub_radius_m
    r_in = g.arms.inner_radius_m
    length = r_hub + g.arms.width_m / 2.0 - r_in
    r_mid = r_in + length / 2.0
    for i, (x, y, _) in enumerate(cfg.motor_positions_c_m):
        yaw = math.atan2(y, x)
        c, s = math.cos(yaw), math.sin(yaw)
        body.add_geom(
            name=f"col_arm{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2.0, g.arms.width_m / 2.0, g.arms.thickness_m / 2.0],
            pos=_vec(np.array([r_mid * c, r_mid * s, g.arms.thickness_m / 2.0]) - com),
            quat=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            **common,
        )

    # motors: bell cylinder from the arm top to the bell top
    m = cfg.motor
    z0 = g.arms.thickness_m
    h = m.base.height_m + m.bell.gap_above_base_m + m.bell.height_m
    for i, (x, y, _) in enumerate(cfg.motor_positions_c_m):
        body.add_geom(
            name=f"col_motor{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[m.bell.outer_diameter_m / 2.0, h / 2.0, 0.0],
            pos=_vec(np.array([x, y, z0 + h / 2.0]) - com),
            **common,
        )

    # prop disks (toggle): thin cylinders at the hub plane
    if sim.prop_disk_collision:
        for i, (x, y, z) in enumerate(cfg.motor_positions_c_m):
            body.add_geom(
                name=f"col_prop{i}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[cfg.prop.diameter_m / 2.0, 0.0005, 0.0],
                pos=_vec(np.array([x, y, z]) - com),
                **common,
            )


def _sensor_slices(model: mujoco.MjModel) -> dict[str, slice]:
    out: dict[str, slice] = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr = int(model.sensor_adr[i])
        out[name] = slice(adr, adr + int(model.sensor_dim[i]))
    return out


def build_index(model: mujoco.MjModel, cfg: DroneConfig, art: ModelArtifacts) -> ModelIndex:
    """Look up ids in a compiled model and snapshot the nominal inertial."""
    mp = art.mass_props
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
    n = len(cfg.motors)
    site_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"m{i}") for i in range(n)],
        dtype=np.int64,
    )
    act_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrust_{m.name.lower()}")
            for m in cfg.motors
        ],
        dtype=np.int64,
    )
    nominal = NominalInertial(
        mass=float(model.body_mass[body_id]),
        inertia=model.body_inertia[body_id].copy(),
        iquat=model.body_iquat[body_id].copy(),
        ipos=model.body_ipos[body_id].copy(),
    )
    return ModelIndex(
        body_id=body_id,
        imu_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, IMU_SITE),
        motor_site_ids=site_ids,
        actuator_ids=act_ids,
        sensor_slices=_sensor_slices(model),
        spin=np.asarray(cfg.spin_signs, dtype=np.float64),
        motor_pos_b=np.asarray(mp.motor_hubs_b_m, dtype=np.float64),
        nominal=nominal,
        k_m_over_k_f=cfg.propulsion.k_m / cfg.propulsion.k_f,
        joint_id=int(model.body_jntadr[body_id]),
        rotor_radius_m=cfg.prop.diameter_m / 2.0,
        com_c_m=(float(mp.com_c_m[0]), float(mp.com_c_m[1]), float(mp.com_c_m[2])),
    )


def build_model(cfg: DroneConfig, sim: SimConfig) -> tuple[mujoco.MjModel, ModelIndex]:
    """Compile the model from the cached CAD artifacts. No CAD kernel is imported.

    Raises ``MissingModelError`` (a ``FileNotFoundError``) with the command to run if the cache
    for ``cfg`` has not been built.
    """
    art = load_artifacts(cfg)
    return build_model_from_artifacts(cfg, art, sim)


def build_model_from_artifacts(
    cfg: DroneConfig, art: ModelArtifacts, sim: SimConfig
) -> tuple[mujoco.MjModel, ModelIndex]:
    spec = build_spec(cfg, art, sim)
    model = spec.compile()
    return model, build_index(model, cfg, art)


def write_inspection_xml(spec: mujoco.MjSpec, path: Path) -> Path:
    """Write ``spec.to_xml()`` (inspection only, never loaded as a source). Call after compile."""
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path
