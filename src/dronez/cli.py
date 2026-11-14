"""Console entry points. Each command lazily imports its implementation module."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronez.cad.artifacts import ModelArtifacts
    from dronez.cad.schema import DroneConfig


def _not_yet(name: str) -> int:
    print(f"{name}: not implemented yet (see docs/PROGRESS.md)", file=sys.stderr)
    return 1


def build_model(argv: list[str] | None = None) -> None:
    """``dronez-build-model``: drone YAML -> CAD -> mass properties, meshes, STEP, renders.

    Then builds the MuJoCo model (MjSpec) from the cache and prints hover command and T/W.
    """
    parser = argparse.ArgumentParser(prog="dronez-build-model", description=build_model.__doc__)
    parser.add_argument("--drone", default="default_5in", help="name in configs/drone/ or a path")
    parser.add_argument("--force", action="store_true", help="ignore the cache and rebuild")
    parser.add_argument("--sim", default="default", help="sim config name in configs/sim/ or path")
    parser.add_argument(
        "--save-xml",
        action="store_true",
        help="write the MJCF (inspection only, generated) into the model cache directory",
    )
    args = parser.parse_args(argv)

    import numpy as np  # noqa: PLC0415

    from dronez.cad.artifacts import is_valid_cache, model_dir  # noqa: PLC0415
    from dronez.cad.export import build_artifacts  # noqa: PLC0415
    from dronez.cad.schema import load_drone_config  # noqa: PLC0415

    cfg = load_drone_config(args.drone)
    cached = not args.force and is_valid_cache(model_dir(cfg), cfg)
    art = build_artifacts(cfg, force=args.force)
    mp = art.mass_props
    to_g = 1e3
    com_mm = np.array(mp.com_c_m) * 1e3
    print(f"drone      : {cfg.name} (hash {mp.model_hash}, cad_version {mp.cad_version})")
    print(f"build      : {'cache hit' if cached else 'built'}")
    print(
        f"mass       : {mp.mass_kg * to_g:.1f} g "
        f"(target {mp.target_auw_kg * to_g:.0f} g, {mp.auw_error_frac * 100:+.1f} %)"
    )
    print(f"CoM (C)    : [{com_mm[0]:+.2f}, {com_mm[1]:+.2f}, {com_mm[2]:+.2f}] mm")
    pm = np.array(mp.principal_moments_kg_m2)
    print(f"principal I: [{pm[0]:.4e}, {pm[1]:.4e}, {pm[2]:.4e}] kg*m^2")
    print(f"source     : {mp.source}")
    print(f"output dir : {art.root}")
    _report_sim(cfg, art, args.sim, args.save_xml)


def _report_sim(cfg: DroneConfig, art: ModelArtifacts, sim_name: str, save_xml: bool) -> None:
    """Build the MuJoCo model from ``art`` and print the propulsion summary."""
    import numpy as np  # noqa: PLC0415

    from dronez.sim import (  # noqa: PLC0415
        MotorModel,
        MotorParams,
        build_index,
        build_spec,
        hover_command,
        hover_thrust_per_motor,
        load_sim_config,
    )
    from dronez.sim.builder import write_inspection_xml  # noqa: PLC0415
    from dronez.utils.config import stable_hash  # noqa: PLC0415

    sim = load_sim_config(sim_name)
    spec = build_spec(cfg, art, sim)
    model = spec.compile()
    index = build_index(model, cfg, art)
    params = MotorParams.from_drone(cfg)
    mass = art.mass_props.mass_kg
    u_h = hover_command(params, mass)
    t_h = hover_thrust_per_motor(mass)
    t_max = float(MotorModel(params).params.k_f[0] * params.omega_max[0] ** 2)
    w_h = float(params.omega_min[0] + u_h[0] * (params.omega_max[0] - params.omega_min[0]))
    print("--- MuJoCo model ---")
    print(
        f"sim        : dt {sim.physics_dt_s * 1e3:.2f} ms, decimation {sim.decimation} "
        f"(policy {sim.policy_rate_hz:.0f} Hz), integrator {sim.integrator}"
    )
    print(f"model      : nq {model.nq}, nv {model.nv}, nu {model.nu}, ngeom {model.ngeom}")
    print(f"gear (yaw) : {np.array2string(model.actuator_gear[:, 5], precision=5)}")
    print(f"hover cmd  : {u_h[0]:.4f} (omega {w_h:.0f} rad/s, {w_h * 60 / (2 * np.pi):.0f} rpm)")
    print(f"hover T    : {t_h:.3f} N per motor ({4 * t_h:.3f} N total = weight)")
    print(f"T_max      : {t_max:.2f} N per motor, T/W = {4 * t_max / (mass * 9.81):.2f}")
    print(f"actuators  : {[int(i) for i in index.actuator_ids]}")
    print(f"sensors    : {sorted(index.sensor_slices)}")
    if save_xml:
        path = art.root / f"model_{stable_hash(sim)}.xml"
        write_inspection_xml(spec, path)
        print(f"xml        : {path}")


def train() -> None:
    sys.exit(_not_yet("dronez-train"))


def evaluate() -> None:
    sys.exit(_not_yet("dronez-eval"))


def export() -> None:
    sys.exit(_not_yet("dronez-export"))
