"""Console entry points. Each command lazily imports its implementation module."""

from __future__ import annotations

import sys


def _not_yet(name: str) -> int:
    print(f"{name}: not implemented yet (see docs/PROGRESS.md)", file=sys.stderr)
    return 1


def build_model() -> None:
    sys.exit(_not_yet("dronez-build-model"))


def train() -> None:
    sys.exit(_not_yet("dronez-train"))


def evaluate() -> None:
    sys.exit(_not_yet("dronez-eval"))


def export() -> None:
    sys.exit(_not_yet("dronez-export"))
