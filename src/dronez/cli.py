"""Console entry points. Each command lazily imports its implementation module."""

from __future__ import annotations

import argparse
import sys


def _not_yet(name: str) -> int:
    print(f"{name}: not implemented yet (see docs/PROGRESS.md)", file=sys.stderr)
    return 1


def build_model(argv: list[str] | None = None) -> None:
    """``dronez-build-model``: drone YAML -> CAD -> mass properties, meshes, STEP, renders.

    Only the CAD stage exists so far; the MJCF stage is added with the sim package.
    """
    parser = argparse.ArgumentParser(prog="dronez-build-model", description=build_model.__doc__)
    parser.add_argument("--drone", default="default_5in", help="name in configs/drone/ or a path")
    parser.add_argument("--force", action="store_true", help="ignore the cache and rebuild")
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


def train() -> None:
    sys.exit(_not_yet("dronez-train"))


def evaluate() -> None:
    sys.exit(_not_yet("dronez-eval"))


def export() -> None:
    sys.exit(_not_yet("dronez-export"))
