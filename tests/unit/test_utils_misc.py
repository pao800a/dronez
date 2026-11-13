"""Tests for paths, config and device helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from dronez.utils import config, device, paths

pytestmark = pytest.mark.fast


def test_data_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "d"))
    assert paths.data_dir() == tmp_path / "d"
    assert paths.runs_dir("abc").is_dir()
    assert paths.models_dir("m", create=False) == tmp_path / "d" / "models" / "m"


def test_default_data_dir_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.default_data_dir() == tmp_path / "dronez"


def test_default_data_dir_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.default_data_dir() == tmp_path / "Dronez" / "data"


def test_repo_root_has_pyproject() -> None:
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_yaml_merge_hash(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("a: 1\nb: {c: 2, d: 3}\n", encoding="utf-8")
    data = config.load_yaml(f)
    merged = config.deep_merge(data, {"b": {"d": 4}})
    assert merged == {"a": 1, "b": {"c": 2, "d": 4}}
    assert config.stable_hash(merged) == config.stable_hash({"b": {"d": 4, "c": 2}, "a": 1})
    assert config.stable_hash(merged) != config.stable_hash(data)


def test_load_model(tmp_path: Path) -> None:
    class Cfg(BaseModel):
        a: int

    f = tmp_path / "x.yaml"
    f.write_text("a: 5\n", encoding="utf-8")
    assert config.load_model(Cfg, f, "drone").a == 5
    with pytest.raises(FileNotFoundError):
        config.resolve_config_path("does-not-exist", "drone")


def test_resolve_device() -> None:
    assert device.resolve_device("CPU") == "cpu"
    with pytest.raises(ValueError, match="unsupported"):
        device.resolve_device("cuda")
    if not device.xpu_available():
        with pytest.raises(RuntimeError):
            device.resolve_device("xpu")
