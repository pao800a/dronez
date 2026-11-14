"""Drone config schema, hashing and artifact-cache lookup. No CAD kernel work (fast)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dronez.cad import artifacts, schema
from dronez.cad.schema import DroneConfig, load_drone_config, model_hash
from dronez.utils.config import load_yaml, resolve_config_path

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    return load_yaml(resolve_config_path("default_5in", "drone"))


@pytest.fixture(scope="module")
def cfg() -> DroneConfig:
    return load_drone_config("default_5in")


def _modified(raw: dict[str, Any], **edits: Any) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for dotted, value in edits.items():
        node = out
        *path, leaf = dotted.split("__")
        for key in path:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(leaf)] = value
        else:
            node[leaf] = value
    return out


def test_default_config_loads(cfg: DroneConfig) -> None:
    assert cfg.name == "default_5in"
    assert cfg.measured_overrides.set_fields() == []
    assert 0.5 <= cfg.target_auw_kg <= 0.7


def test_motor_layout_matches_conventions(cfg: DroneConfig) -> None:
    # conventions.md: M0 FR ccw (+a,-a), M1 RL ccw (-a,+a), M2 FL cw (+a,+a), M3 RR cw (-a,-a)
    a = 0.25 / (2 * 2**0.5)
    expected = [("FR", 1, -1, 1), ("RL", -1, 1, 1), ("FL", 1, 1, -1), ("RR", -1, -1, -1)]
    for m, s, (name, sx, sy, spin) in zip(cfg.motors, cfg.spin_signs, expected, strict=True):
        assert m.name == name
        assert m.position_m[0] == pytest.approx(sx * a, abs=1e-4)
        assert m.position_m[1] == pytest.approx(sy * a, abs=1e-4)
        assert s == spin
    diag = 2 * cfg.hub_radius_m
    assert diag == pytest.approx(0.250, abs=1e-3)
    centroid = [sum(p[i] for p in cfg.motor_positions_c_m) for i in range(2)]
    assert centroid == pytest.approx([0.0, 0.0], abs=1e-9)


def test_hash_is_stable_and_short(cfg: DroneConfig) -> None:
    h = model_hash(cfg)
    assert len(h) == 10
    assert h == model_hash(load_drone_config("default_5in"))
    int(h, 16)


def test_hash_changes_with_config_and_cad_version(
    raw: dict[str, Any], cfg: DroneConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = model_hash(cfg)
    heavier = DroneConfig.model_validate(_modified(raw, target_auw_kg=0.6))
    assert model_hash(heavier) != base
    monkeypatch.setattr(schema, "CAD_VERSION", "999")
    assert model_hash(cfg) != base


@pytest.mark.parametrize(
    ("edits", "match"),
    [
        ({"motors__0__spin": "cw"}, "PX4 Quad-X"),
        ({"motors__0__position_m": [-0.0884, -0.0884, 0.025]}, "wrong quadrant"),
        ({"motors__0__position_m": [0.0884, -0.07, 0.025]}, "\\|x\\| != \\|y\\|"),
        ({"motors__0__position_m": [0.0884, -0.0884, 0.030]}, "same z"),
        ({"parts_material__arms": "unobtainium"}, "unknown material"),
        ({"propulsion__omega_min_rad_s": 5000.0}, "omega_min"),
        ({"geometry__min_prop_clearance_m": 0.2}, "adjacent props"),
        ({"measured_overrides": {"mass_kg": 0.5}}, "source"),
        ({"layout": "hex"}, "quad_x"),
        ({"bogus_field": 1}, "Extra inputs"),
    ],
)
def test_validator_rejects(raw: dict[str, Any], edits: dict[str, Any], match: str) -> None:
    bad = _modified(raw, **edits)
    with pytest.raises((ValidationError, ValueError), match=match):
        DroneConfig.model_validate(bad)


def test_motor_z_must_match_arm_and_prop_plane(raw: dict[str, Any]) -> None:
    bad = _modified(raw, geometry__arms__thickness_m=0.010)
    with pytest.raises(ValidationError, match="prop plane height"):
        DroneConfig.model_validate(bad)


def test_measured_overrides_with_source_ok(raw: dict[str, Any]) -> None:
    ok = _modified(raw, measured_overrides={"mass_kg": 0.5, "source": "kitchen scale"})
    cfg = DroneConfig.model_validate(ok)
    assert cfg.measured_overrides.set_fields() == ["mass_kg"]


def test_load_artifacts_missing_raises(
    cfg: DroneConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRONEZ_DATA_DIR", str(tmp_path))
    with pytest.raises(artifacts.MissingModelError, match="dronez-build-model"):
        artifacts.load_artifacts(cfg)


def test_cache_validity_requires_matching_manifest(
    cfg: DroneConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRONEZ_DATA_DIR", str(tmp_path))
    root = artifacts.model_dir(cfg)
    assert root.name == f"{cfg.name}-{model_hash(cfg)}"
    root.mkdir(parents=True)
    assert not artifacts.is_valid_cache(root, cfg)
    (root / artifacts.MANIFEST_NAME).write_text("not json")
    assert not artifacts.is_valid_cache(root, cfg)
    (root / artifacts.MANIFEST_NAME).write_text(json.dumps({"model_hash": "0000000000"}))
    assert not artifacts.is_valid_cache(root, cfg)
    (root / artifacts.MANIFEST_NAME).write_text(json.dumps({"model_hash": model_hash(cfg)}))
    assert artifacts.is_valid_cache(root, cfg)


def test_schema_and_artifacts_do_not_import_build123d() -> None:
    code = (
        "import sys; import dronez.cad, dronez.cad.schema, dronez.cad.artifacts; "
        "bad = [m for m in sys.modules if m.split('.')[0] in {'build123d', 'OCP', 'matplotlib'}]; "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
