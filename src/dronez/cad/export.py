"""Exports and the cache-aware, atomic ``build_artifacts`` entry point.

Layout of ``$DRONEZ_DATA_DIR/models/<name>-<hash>/``: ``manifest.json`` (written last),
``drone_config.resolved.yaml``, ``mass_props.json``, ``meshes/<part>.stl|obj`` (metres, frame C,
decimated), ``assembly.step`` (millimetres, STEP carries its units) and ``render_*.png``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from build123d import export_step
from numpy.typing import NDArray

from dronez.cad import render
from dronez.cad.artifacts import (
    MANIFEST_NAME,
    MASS_PROPS_NAME,
    MESHES_DIR,
    ModelArtifacts,
    is_valid_cache,
    model_dir,
    read_artifacts,
)
from dronez.cad.assembly import Assembly, Part, build_assembly
from dronez.cad.mass_props import compute_mass_properties
from dronez.cad.parts import MM_PER_M
from dronez.cad.schema import CAD_VERSION, DroneConfig, MassProperties, model_hash
from dronez.utils.paths import repo_root

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Mesh = tuple[F64, I64]

_WELD_DECIMALS_MM = 6  # nanometre grid


def tessellate_part(part: Part, tolerance_mm: float, angular_tol_rad: float) -> Mesh:
    """Welded triangle mesh of all instances of a part, in metres, frame C."""
    verts: list[F64] = []
    faces: list[I64] = []
    offset = 0
    for solid in part.solids:
        v, t = solid.tessellate(tolerance_mm, angular_tol_rad)
        verts.append(np.array([[p.X, p.Y, p.Z] for p in v], dtype=np.float64))
        faces.append(np.array(t, dtype=np.int64) + offset)
        offset += len(v)
    all_v = np.concatenate(verts)
    all_f = np.concatenate(faces)
    uniq, inverse = np.unique(np.round(all_v, _WELD_DECIMALS_MM), axis=0, return_inverse=True)
    all_f = inverse.reshape(-1)[all_f]
    keep = (
        (all_f[:, 0] != all_f[:, 1]) & (all_f[:, 1] != all_f[:, 2]) & (all_f[:, 0] != all_f[:, 2])
    )
    return uniq / MM_PER_M, all_f[keep]


def decimated_mesh(part: Part, cfg: DroneConfig) -> Mesh:
    """Tessellate, coarsening the tolerance until the triangle cap is met."""
    tol_mm = cfg.cad.mesh_tolerance_m * MM_PER_M
    ang = cfg.cad.mesh_angular_tolerance_rad
    mesh = tessellate_part(part, tol_mm, ang)
    for _ in range(8):
        if len(mesh[1]) <= cfg.cad.max_triangles_per_part:
            break
        tol_mm *= 2.0
        ang *= 1.5
        mesh = tessellate_part(part, tol_mm, ang)
    return mesh


def mesh_volume(mesh: Mesh) -> float:
    """Signed volume of a closed mesh (positive for outward-facing triangles), m^3."""
    v, f = mesh
    t = v[f]
    return float(np.einsum("ij,ij->", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0)


def write_stl(mesh: Mesh, path: Path, name: str) -> None:
    v, f = mesh
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, t
    header = f"dronez {name} (metres, frame C)".encode().ljust(80, b" ")[:80]
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def write_obj(mesh: Mesh, path: Path, name: str) -> None:
    v, f = mesh
    lines = [f"# dronez {name} (metres, frame C)", f"o {name}"]
    lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_meshes(asm: Assembly, cfg: DroneConfig, out_dir: Path) -> dict[str, Mesh]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, Mesh] = {}
    for name, part in asm.parts.items():
        mesh = decimated_mesh(part, cfg)
        write_stl(mesh, out_dir / f"{name}.stl", name)
        write_obj(mesh, out_dir / f"{name}.obj", name)
        meshes[name] = mesh
    return meshes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return r.stdout.strip()

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _manifest(cfg: DroneConfig, root: Path) -> dict[str, Any]:
    files = {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST_NAME
    }
    return {
        "schema_version": 1,
        "drone": cfg.name,
        "model_hash": model_hash(cfg),
        "cad_version": CAD_VERSION,
        "hash_inputs": ["drone_config", "cad_version"],
        "git": _git_info(),
        "versions": {
            "python": platform.python_version(),
            "build123d": _pkg_version("build123d"),
            "numpy": np.__version__,
        },
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }


def _write_into(cfg: DroneConfig, root: Path) -> MassProperties:
    asm = build_assembly(cfg)
    mass_props = compute_mass_properties(asm, cfg)
    (root / MASS_PROPS_NAME).write_text(mass_props.model_dump_json(indent=2) + "\n", "utf-8")
    (root / "drone_config.resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    meshes = export_meshes(asm, cfg, root / MESHES_DIR)
    export_step(asm.compound(), root / "assembly.step")
    render.render_views(
        meshes,
        root,
        mass_props.com_c_m,
        cfg.motor_positions_c_m,
        title=f"{cfg.name} ({mass_props.mass_kg * 1e3:.0f} g)",
    )
    (root / MANIFEST_NAME).write_text(
        json.dumps(_manifest(cfg, root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return mass_props


def build_artifacts(cfg: DroneConfig, force: bool = False) -> ModelArtifacts:
    """Build (or load from the cache) the CAD artifacts of ``cfg``.

    The build runs in ``models/.tmp-<hash>-<pid>/`` and is renamed into place, so a crash never
    leaves a half-built cache. A directory is valid only when its manifest carries the hash.
    """
    final = model_dir(cfg)
    if not force and is_valid_cache(final, cfg):
        return read_artifacts(final)
    tmp = final.parent / f".tmp-{model_hash(cfg)}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _write_into(cfg, tmp)
        if final.exists():
            old = final.parent / f".old-{model_hash(cfg)}-{os.getpid()}"
            final.rename(old)
            tmp.rename(final)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return read_artifacts(final)
