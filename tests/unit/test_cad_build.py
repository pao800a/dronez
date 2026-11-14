"""Full CAD build: solids, mass properties, exports, cache. Needs build123d (``-m cad``)."""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from build123d import Box, CenterOf, Pos

from dronez.cad import export
from dronez.cad.artifacts import MANIFEST_NAME, ModelArtifacts, is_valid_cache, model_dir
from dronez.cad.assembly import Assembly, build_assembly
from dronez.cad.mass_props import aggregate, compute_mass_properties, principal, solid_properties
from dronez.cad.schema import DroneConfig, MassProperties, load_drone_config, model_hash

pytestmark = pytest.mark.cad


@pytest.fixture(scope="module")
def cfg() -> DroneConfig:
    return load_drone_config("default_5in")


@pytest.fixture(scope="module")
def asm(cfg: DroneConfig) -> Assembly:
    return build_assembly(cfg)


@pytest.fixture(scope="module")
def mp(asm: Assembly, cfg: DroneConfig) -> MassProperties:
    return compute_mass_properties(asm, cfg)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    path = tmp_path_factory.mktemp("dronez_data")
    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setenv("DRONEZ_DATA_DIR", str(path))
        yield path


@pytest.fixture(scope="module")
def built(cfg: DroneConfig, data_dir: Path) -> ModelArtifacts:
    return export.build_artifacts(cfg)


def _inertia(mp: MassProperties) -> np.ndarray:
    return np.array(mp.inertia_com_kg_m2)


# -- units and the parallel-axis theorem ---------------------------------------------------------


def test_unit_conversion_known_box() -> None:
    # 10 x 20 x 30 mm box at 1000 kg/m^3: m = 6e-3 kg, Ixx = m (b^2 + c^2) / 12 = 6.5e-7 kg m^2.
    m, com, inertia = solid_properties(Box(10, 20, 30), 1000.0)
    assert m == pytest.approx(6e-3, rel=1e-9)
    assert com == pytest.approx([0, 0, 0], abs=1e-12)
    assert inertia[0, 0] == pytest.approx(6e-3 / 12 * (0.02**2 + 0.03**2), rel=1e-9)
    assert inertia[1, 1] == pytest.approx(6e-3 / 12 * (0.01**2 + 0.03**2), rel=1e-9)
    assert inertia[2, 2] == pytest.approx(6e-3 / 12 * (0.01**2 + 0.02**2), rel=1e-9)
    # translated copy: CoM in metres, inertia about its own CoM unchanged
    m2, com2, inertia2 = solid_properties(Pos(100, -50, 25) * Box(10, 20, 30), 1000.0)
    assert com2 == pytest.approx([0.1, -0.05, 0.025], rel=1e-9)
    assert inertia2 == pytest.approx(inertia, abs=1e-15)
    assert m2 == pytest.approx(m)


def test_parallel_axis_two_boxes() -> None:
    a, b = Pos(-30, 0, 0) * Box(10, 20, 30), Pos(30, 10, 5) * Box(20, 20, 20)
    items = [solid_properties(a, 2000.0), solid_properties(b, 500.0)]
    mass, com, inertia = aggregate(items)
    # brute-force: integrate point masses on a fine grid
    rng = np.random.default_rng(0)
    pts, ms = [], []
    for solid, rho in ((a, 2000.0), (b, 500.0)):
        bb = solid.bounding_box()
        lo = np.array([bb.min.X, bb.min.Y, bb.min.Z]) * 1e-3
        hi = np.array([bb.max.X, bb.max.Y, bb.max.Z]) * 1e-3
        n = 200_000
        pts.append(rng.uniform(lo, hi, size=(n, 3)))
        ms.append(np.full(n, rho * float(np.prod(hi - lo)) / n))
    p, w = np.concatenate(pts), np.concatenate(ms)
    assert w.sum() == pytest.approx(mass, rel=1e-9)
    com_mc = (p * w[:, None]).sum(0) / w.sum()
    assert com == pytest.approx(com_mc, abs=5e-4)
    d = p - com
    i_mc = np.einsum(
        "n,nij->ij", w, (d**2).sum(1)[:, None, None] * np.eye(3) - d[:, :, None] * d[:, None, :]
    )
    assert inertia == pytest.approx(i_mc, rel=0.01)


def test_assembly_inertia_matches_occ_compound(asm: Assembly) -> None:
    """Parallel-axis aggregation vs OCC's direct integral over the whole compound (uniform rho)."""
    rho = 1000.0
    items = [solid_properties(s, rho) for p in asm.parts.values() for s in p.solids]
    mass, com, inertia = aggregate(items)
    comp = asm.compound()
    occ_com = comp.center(CenterOf.MASS)
    assert com * 1e3 == pytest.approx([occ_com.X, occ_com.Y, occ_com.Z], abs=1e-6)
    occ_i = np.array(comp.matrix_of_inertia) * rho * 1e-15
    assert inertia == pytest.approx(occ_i, rel=1e-6, abs=1e-12)
    solids_volume = sum(s.volume for p in asm.parts.values() for s in p.solids)
    assert mass == pytest.approx(rho * solids_volume * 1e-9, rel=1e-9)


# -- mass properties (gate G1) ---------------------------------------------------------------------


def test_mass_within_ten_percent_of_target(mp: MassProperties, cfg: DroneConfig) -> None:
    assert abs(mp.mass_kg - cfg.target_auw_kg) / cfg.target_auw_kg < 0.10
    assert mp.auw_error_frac == pytest.approx((mp.mass_kg - cfg.target_auw_kg) / cfg.target_auw_kg)
    assert sum(p.mass_kg for p in mp.parts) == pytest.approx(mp.mass_kg, rel=1e-12)


def test_com_xy_near_centre(mp: MassProperties) -> None:
    assert abs(mp.com_c_m[0]) < 3e-3
    assert abs(mp.com_c_m[1]) < 3e-3
    assert 0.0 < mp.com_c_m[2] < 0.06  # above the bottom plate, below the battery top


def test_inertia_spd_and_triangle_inequalities(mp: MassProperties) -> None:
    inertia = _inertia(mp)
    assert inertia == pytest.approx(inertia.T, abs=1e-15)
    w = np.linalg.eigvalsh(inertia)
    assert np.all(w > 0.0)
    assert w == pytest.approx(sorted(mp.principal_moments_kg_m2), rel=1e-9)
    for a, b, c in itertools.permutations(w):
        assert a + b >= c * (1.0 - 1e-12)
    ixx, iyy, izz = np.diag(inertia)
    assert ixx + iyy >= izz
    assert iyy + izz >= ixx
    assert ixx + izz >= iyy
    # a flat-ish quad: yaw inertia is the largest, roll/pitch are of the same order
    assert izz > max(ixx, iyy)
    assert 5e-4 < ixx < 1e-2
    assert 5e-4 < iyy < 1e-2


def test_principal_axes_quaternion_consistent(mp: MassProperties) -> None:
    from dronez.utils.rotations import quat_to_rotmat

    r = quat_to_rotmat(mp.principal_axes_quat_wxyz)
    assert np.linalg.det(r) == pytest.approx(1.0)
    assert r @ np.diag(mp.principal_moments_kg_m2) @ r.T == pytest.approx(_inertia(mp), abs=1e-12)
    w, _ = principal(_inertia(mp))
    assert w == pytest.approx(mp.principal_moments_kg_m2)


def test_motor_hubs_match_conventions(mp: MassProperties, cfg: DroneConfig) -> None:
    com = np.array(mp.com_c_m)
    hubs = np.array(mp.motor_hubs_b_m)
    assert hubs + com == pytest.approx(np.array(cfg.motor_positions_c_m), abs=1e-12)
    # PX4 Quad-X order, FLU: M0 FR (+,-), M1 RL (-,+), M2 FL (+,+), M3 RR (-,-)
    signs = np.sign(hubs[:, :2] + com[:2])
    assert signs.tolist() == [[1, -1], [-1, 1], [1, 1], [-1, -1]]
    assert cfg.spin_signs == [1, 1, -1, -1]
    # the prop meshes carry the spin: blade tilt flips between CCW and CW props
    props = asm_props_tilt_signs(cfg)
    assert props == cfg.spin_signs


def asm_props_tilt_signs(cfg: DroneConfig) -> list[int]:
    """Sign of the z-offset of a blade's +y edge relative to its -y edge, per prop."""
    from dronez.cad.parts import _prop

    out = []
    for s in cfg.spin_signs:
        bb = _prop(cfg, s)
        # take material of the blade pointing along +x at x in [30, 60] mm
        verts = np.array([[p.X, p.Y, p.Z] for p in bb.vertices()])
        sel = verts[(verts[:, 0] > 30.0)]
        lead = sel[sel[:, 1] > 0.0, 2].mean()
        trail = sel[sel[:, 1] < 0.0, 2].mean()
        out.append(1 if lead > trail else -1)
    return out


def test_no_part_overlaps(asm: Assembly) -> None:
    bad = []
    for a, b in itertools.combinations(asm.parts, 2):
        for sa in asm.parts[a].solids:
            for sb in asm.parts[b].solids:
                if not sa.bounding_box().overlaps(sb.bounding_box()):
                    continue
                common = sa & sb
                if common is not None and common.volume > 1e-3:  # mm^3
                    bad.append((a, b, common.volume))
    assert not bad, f"overlapping parts: {bad}"


def test_every_part_has_a_material(asm: Assembly, cfg: DroneConfig) -> None:
    assert set(asm.materials) == set(cfg.parts_material)
    assert all(p.count >= 1 for p in asm.parts.values())
    assert asm.parts["props"].count == 4
    assert asm.parts["arms"].count == 4


# -- measured overrides --------------------------------------------------------------------


def _with_overrides(cfg: DroneConfig, **ov: Any) -> DroneConfig:
    data = cfg.model_dump(mode="json")
    data["measured_overrides"] = {"source": "test", **ov}
    return DroneConfig.model_validate(data)


def test_mass_only_override_scales_inertia(
    asm: Assembly, cfg: DroneConfig, mp: MassProperties
) -> None:
    over = compute_mass_properties(asm, _with_overrides(cfg, mass_kg=0.6))
    k = 0.6 / mp.mass_kg
    assert over.mass_kg == 0.6
    assert _inertia(over) == pytest.approx(_inertia(mp) * k, rel=1e-12)
    assert over.com_c_m == pytest.approx(mp.com_c_m)
    assert over.overrides_applied == ["mass_kg"]
    assert over.source == "cad+measured"


def test_full_override_replaces_everything(asm: Assembly, cfg: DroneConfig) -> None:
    inertia = [[2e-3, 0, 0], [0, 2.5e-3, 0], [0, 0, 4e-3]]
    over = compute_mass_properties(
        asm,
        _with_overrides(cfg, mass_kg=0.6, com_c_m=[0.001, 0.0, 0.03], inertia_com_kg_m2=inertia),
    )
    assert _inertia(over) == pytest.approx(np.array(inertia))
    assert over.com_c_m == pytest.approx((0.001, 0.0, 0.03))
    assert over.principal_moments_kg_m2 == pytest.approx((2e-3, 2.5e-3, 4e-3))
    assert np.array(over.motor_hubs_b_m)[0] == pytest.approx(
        np.array(cfg.motor_positions_c_m[0]) - np.array([0.001, 0.0, 0.03])
    )


# -- exports and cache ---------------------------------------------------------------------


def test_artifacts_layout(built: ModelArtifacts, cfg: DroneConfig, asm: Assembly) -> None:
    root = built.root
    assert root == model_dir(cfg)
    assert root.name == f"{cfg.name}-{model_hash(cfg)}"
    for name in (
        "manifest.json",
        "drone_config.resolved.yaml",
        "mass_props.json",
        "assembly.step",
        "render_top.png",
        "render_side.png",
        "render_iso.png",
    ):
        assert (root / name).is_file(), name
    assert set(built.meshes) == set(asm.parts)
    for name in asm.parts:
        assert (root / "meshes" / f"{name}.obj").is_file()
    assert not list(root.parent.glob(".tmp-*"))


def test_mass_props_json_roundtrip(built: ModelArtifacts, mp: MassProperties) -> None:
    on_disk = json.loads((built.root / "mass_props.json").read_text())
    assert on_disk["schema_version"] == 1
    assert on_disk["units"] == {"length": "m", "mass": "kg", "inertia": "kg*m^2"}
    assert built.mass_props.mass_kg == pytest.approx(mp.mass_kg, rel=1e-12)
    assert built.mass_props.model_hash == model_hash(load_drone_config("default_5in"))
    assert built.mass_props.overrides_applied == []
    assert built.mass_props.source == "cad"


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    (n,) = struct.unpack("<I", raw[80:84])
    rec = np.frombuffer(
        raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")], count=n
    )
    assert len(raw) == 84 + 50 * n
    return rec["n"].astype(np.float64), rec["v"].astype(np.float64)


def test_meshes_are_metric_light_and_outward(
    built: ModelArtifacts, asm: Assembly, cfg: DroneConfig
) -> None:
    total = 0
    for name, path in built.meshes.items():
        _, tri = _read_stl(path)
        total += len(tri)
        assert len(tri) <= cfg.cad.max_triangles_per_part
        assert np.abs(tri).max() < 0.5, f"{name}: vertices not in metres"
        vol = np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
        solid_vol = sum(s.volume for s in asm.parts[name].solids) * 1e-9
        assert vol == pytest.approx(solid_vol, rel=0.02), name
    assert total < 25_000


def test_step_and_renders_nonempty(built: ModelArtifacts) -> None:
    assert b"ISO-10303" in (built.root / "assembly.step").read_bytes()[:64]
    for view in ("top", "side", "iso"):
        png = (built.root / f"render_{view}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png) > 10_000


def test_manifest_hashes_and_versions(built: ModelArtifacts, cfg: DroneConfig) -> None:
    m = built.manifest
    assert m["model_hash"] == model_hash(cfg)
    assert m["drone"] == cfg.name
    assert {"python", "build123d", "numpy"} <= set(m["versions"])
    assert "sha" in m["git"]
    assert MANIFEST_NAME not in m["files"]
    assert "mass_props.json" in m["files"]
    for rel, digest in m["files"].items():
        assert hashlib.sha256((built.root / rel).read_bytes()).hexdigest() == digest, rel


def test_second_build_is_a_cache_hit(
    built: ModelArtifacts, cfg: DroneConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("cache miss: _write_into was called")

    monkeypatch.setattr(export, "_write_into", boom)
    assert is_valid_cache(built.root, cfg)
    again = export.build_artifacts(cfg)
    assert again.root == built.root
    assert again.manifest == built.manifest


def test_force_rebuilds_atomically(
    built: ModelArtifacts, cfg: DroneConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = export._write_into

    def counting(c: DroneConfig, root: Path) -> MassProperties:
        calls.append(1)
        return real(c, root)

    monkeypatch.setattr(export, "_write_into", counting)
    art = export.build_artifacts(cfg, force=True)
    assert calls == [1]
    assert is_valid_cache(art.root, cfg)
    assert not list(art.root.parent.glob(".tmp-*"))
    assert not list(art.root.parent.glob(".old-*"))
    assert art.mass_props.mass_kg == pytest.approx(built.mass_props.mass_kg, rel=1e-12)


def test_failed_build_leaves_no_partial_cache(
    cfg: DroneConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRONEZ_DATA_DIR", str(tmp_path))

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(export, "_write_into", boom)
    with pytest.raises(RuntimeError):
        export.build_artifacts(cfg)
    assert not model_dir(cfg).exists()
    assert not list(model_dir(cfg).parent.glob(".tmp-*"))


def test_changed_config_gets_new_directory(cfg: DroneConfig) -> None:
    data = cfg.model_dump(mode="json")
    data["target_auw_kg"] = 0.56
    other = DroneConfig.model_validate(data)
    assert model_dir(other) != model_dir(cfg)


def test_principal_orders_axes_by_closest_body_axis() -> None:
    """moments[i] belongs to body axis i even when the moments are not ascending (Iyy > Izz)."""
    inertia = np.diag([2.0e-3, 5.0e-3, 3.0e-3])
    inertia[0, 2] = inertia[2, 0] = 1.0e-4  # small product of inertia
    w, v = principal(inertia)
    assert np.linalg.det(v) == pytest.approx(1.0)
    assert v @ np.diag(w) @ v.T == pytest.approx(inertia, abs=1e-15)
    assert w[1] == pytest.approx(5.0e-3)  # y stays the middle axis, not sorted to the end
    assert w[0] < w[2] < w[1]
    np.testing.assert_allclose(np.abs(np.diag(v)), 1.0, atol=1e-2)
    assert v[0, 0] > 0.0
    assert v[1, 1] > 0.0
