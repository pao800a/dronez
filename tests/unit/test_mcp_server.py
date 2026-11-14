"""dronez-sim MCP server tool functions (no MCP transport). Test data is fabricated."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dronez.utils import paths

sys.path.insert(0, str(paths.repo_root()))
from tools.mcp_server import core

pytestmark = pytest.mark.fast


def _size(d: dict[str, Any]) -> int:
    return len(json.dumps(d).encode())


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DRONEZ_DATA_DIR", str(tmp_path))
    run = tmp_path / "runs" / "r1"
    (run / "tb").mkdir(parents=True)
    (run / "checkpoints" / "step_000000000100").mkdir(parents=True)
    (run / "checkpoints" / "step_000000000100" / "ckpt.json").write_text('{"timesteps": 100}')
    (run / "checkpoints" / "step_000000000100" / "model.zip").write_bytes(b"x")
    (run / "status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "timesteps_done": 50,
                "timesteps_total": 200,
                "eta_s": 12.0,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    ev = tmp_path / "eval" / "r1" / "step_000000000100"
    ev.mkdir(parents=True)
    scen = {
        f"s{i}": {"kind": "recovery", "aggregate": {"n_episodes": 3, "success_rate": 1.0}}
        for i in range(3)
    }
    (ev / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "checkpoint": "step_000000000100",
                "scenarios": scen,
                "gates": {
                    "G4": {
                        "pass": True,
                        "criteria": [{"name": f"c{i}", "pass": True} for i in range(40)],
                    }
                },
            }
        )
    )
    return tmp_path


def test_list_runs_and_status(data: Path) -> None:
    r = core.list_runs()
    assert r["runs"][0]["run_id"] == "r1"
    assert r["runs"][0]["timesteps_done"] == 50
    assert r["source"]
    s = core.training_status("r1")
    assert s["progress_frac"] == 0.25
    assert s["source"].endswith("status.json")
    assert _size(s) <= core.MAX_BYTES


def test_errors_are_json(data: Path) -> None:
    for out in (
        core.training_status("missing"),
        core.training_status("../evil"),
        core.eval_summary("r1", "nope"),
        core.gate_check("r1", "latest", "G9"),
        core.tb_scalars("r1"),
        core.sim_rollout(duration_s=99.0),
    ):
        assert set(out) == {"error"}
        assert "Traceback" not in out["error"]


def test_checkpoints(data: Path) -> None:
    r = core.list_checkpoints("r1")
    assert r["checkpoints"][0]["name"] == "step_000000000100"
    assert r["checkpoints"][0]["complete"] is True
    assert r["checkpoints"][0]["meta"] == {"timesteps": 100}
    assert r["checkpoints"][0]["source"].endswith("ckpt.json")


def test_eval_and_gate_truncation(data: Path) -> None:
    e = core.eval_summary("r1", "latest")
    assert e["source"].endswith("metrics.json")
    assert set(e["scenarios"]) == {"s0", "s1", "s2"}
    assert e["gates_pass"] == {"G4": True}
    g = core.gate_check("r1", "step_000000000100", "g4")
    assert g["gate"] == "G4"
    assert g["pass"] is True
    assert len(g["criteria"]) == core.MAX_ARRAY
    assert g["truncated"] is True
    assert g["source"].endswith("metrics.json")
    assert _size(g) <= core.MAX_BYTES


def test_tb_scalars(data: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(str(data / "runs" / "r1" / "tb"))
    for i in range(100):
        w.add_scalar("a/b", float(i), i)
    w.close()
    r = core.tb_scalars("r1", ["a/b", "zzz"], points=10)
    t = r["scalars"]["a/b"]
    assert t["last"] == 99.0
    assert t["min"] == 0.0
    assert t["max"] == 99.0
    assert len(t["values"]) <= 10
    assert r["scalars"]["zzz"] == {"error": "unknown tag"}
    assert r["source"]
    assert core.tb_scalars("r1")["available_tags"] == ["a/b"]


def test_output_cap_and_array_truncation() -> None:
    out = core.finalize({"x": list(range(100)), "y": {"z": list(range(5))}})
    assert len(out["x"]) == core.MAX_ARRAY
    assert out["truncated"] is True
    assert out["y"]["z"] == list(range(5))
    big = core.finalize({f"k{i}": "v" * 200 for i in range(100)})
    assert _size(big) <= core.MAX_BYTES
    assert "error" in big
    shrunk = core.finalize({"x": ["w" * 200] * 32})
    assert _size(shrunk) <= core.MAX_BYTES
    assert shrunk["truncated"] is True
    assert core.finalize({"v": float("nan")})["v"] is None


def test_sim_rollout_and_model_info(model_cache: None) -> None:
    info = core.model_info("default_5in")
    assert "error" not in info
    assert info["mass_kg"] > 0
    assert info["thrust_to_weight"] > 1
    assert info["source"].endswith("mass_props.json")
    r = core.sim_rollout("pid", "hover_hold", seed=1, duration_s=0.5)
    assert "error" not in r, r
    assert r["steps"] > 0
    assert r["crashed"] is False
    assert _size(r) <= core.MAX_BYTES
