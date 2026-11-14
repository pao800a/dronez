"""Guard against a stale model cache: CAD code changes must come with a ``CAD_VERSION`` bump.

``tests/unit/cad_fingerprint.json`` records a fingerprint of ``src/dronez/cad/*.py`` (an AST dump
without docstrings, so comments, docstrings and formatting do not count) next to the
``CAD_VERSION`` it was recorded with.

Workflow when this test fails:

1. You changed the CAD code: bump ``CAD_VERSION`` in ``src/dronez/cad/schema.py``.
2. Re-record: ``DRONEZ_UPDATE_CAD_FINGERPRINT=1 uv run pytest tests/unit/test_cad_version.py``.
   (Refused unless ``CAD_VERSION`` differs from the recorded one: no hiding a forgotten bump.)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from dronez.cad.schema import CAD_VERSION

pytestmark = pytest.mark.fast

CAD_DIR = Path(__file__).resolve().parents[2] / "src" / "dronez" / "cad"
RECORD = Path(__file__).with_name("cad_fingerprint.json")
UPDATE_ENV = "DRONEZ_UPDATE_CAD_FINGERPRINT"


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(CAD_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _strip_docstrings(tree)
        h.update(path.name.encode())
        h.update(ast.dump(tree, include_attributes=False).encode())
    return h.hexdigest()


def test_cad_version_tracks_cad_sources() -> None:
    current = fingerprint()
    recorded = json.loads(RECORD.read_text()) if RECORD.exists() else None

    if os.environ.get(UPDATE_ENV):
        if recorded is not None and recorded["cad_version"] == CAD_VERSION:
            pytest.fail(
                f"refusing to re-record: CAD_VERSION is still {CAD_VERSION!r}. "
                "Bump it in src/dronez/cad/schema.py first."
            )
        RECORD.write_text(
            json.dumps({"cad_version": CAD_VERSION, "fingerprint": current}, indent=2) + "\n"
        )
        return

    assert recorded is not None, f"{RECORD.name} missing; run with {UPDATE_ENV}=1"
    if current == recorded["fingerprint"]:
        assert recorded["cad_version"] == CAD_VERSION
    elif recorded["cad_version"] == CAD_VERSION:
        pytest.fail(
            "src/dronez/cad/*.py changed but CAD_VERSION was not bumped "
            f"(still {CAD_VERSION!r}); a stale model cache would be reused. Bump CAD_VERSION in "
            f"src/dronez/cad/schema.py, then run {UPDATE_ENV}=1 uv run pytest "
            "tests/unit/test_cad_version.py"
        )
    else:
        pytest.fail(
            f"CAD_VERSION was bumped ({recorded['cad_version']!r} -> {CAD_VERSION!r}) but "
            f"tests/unit/cad_fingerprint.json is stale; run {UPDATE_ENV}=1 uv run pytest "
            "tests/unit/test_cad_version.py"
        )


def test_fingerprint_ignores_comments_docstrings_and_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = tmp_path / "m.py"
    monkeypatch.setattr(sys.modules[__name__], "CAD_DIR", tmp_path)
    mod.write_text('"""Doc."""\n\n\ndef f(x):\n    """Doc."""\n    return x+1  # c\n')
    base = fingerprint()
    mod.write_text('"""Other doc."""\n# comment\n\ndef f( x ):\n    return (x + 1)\n')
    assert fingerprint() == base
    mod.write_text("def f(x):\n    return x + 2\n")
    assert fingerprint() != base
