"""Torch device selection: CPU by default, Intel XPU strictly opt-in.

See docs/adr for the CPU-vs-XPU benchmark. For small MLP policies the env
stepping dominates and the CPU is expected to win. CUDA is not supported.
"""

from __future__ import annotations

import logging
from typing import Literal

DeviceName = Literal["cpu", "xpu"]

log = logging.getLogger(__name__)


def xpu_available() -> bool:
    """True if this torch build exposes a usable Intel XPU device."""
    try:
        import torch  # noqa: PLC0415 - torch is heavy; import lazily
    except ImportError:
        return False
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except Exception:  # pragma: no cover - driver-specific failures
        return False


def resolve_device(requested: str = "cpu") -> DeviceName:
    """Validate a requested device and return it; logs the choice.

    Raises ``ValueError`` for unknown devices and ``RuntimeError`` when
    ``xpu`` is requested but unavailable (no silent fallback: a run that
    claims XPU must actually use it).
    """
    name = requested.strip().lower()
    if name == "cpu":
        log.info("torch device: cpu")
        return "cpu"
    if name == "xpu":
        if not xpu_available():
            raise RuntimeError("--device xpu requested but torch.xpu.is_available() is False")
        log.info("torch device: xpu")
        return "xpu"
    raise ValueError(f"unsupported device {requested!r}; use 'cpu' or 'xpu'")


def describe_devices() -> dict[str, object]:
    """Small summary for status blocks and run metadata."""
    info: dict[str, object] = {"cpu": True, "xpu": xpu_available()}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["threads"] = torch.get_num_threads()
    except ImportError:
        info["torch"] = None
    return info
