"""Compact summary of TensorBoard scalars for a run directory (no raw arrays).

Usage:
    uv run python .claude/skills/training-diagnostics/scripts/tb_summary.py RUN [--tags a,b]

RUN is a run directory or a run id under DRONEZ_DATA_DIR/runs.

For every scalar tag: last value, min, max, and the trend over the last 20 %
of points. Searches the run dir recursively for event files (SB3 nests them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dronez.utils.paths import runs_dir


def summarize(run: Path, tags: list[str] | None = None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ev_dir in sorted({p.parent for p in run.rglob("events.out.tfevents.*")}):
        acc = EventAccumulator(str(ev_dir), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if tags and not any(t in tag for t in tags):
                continue
            events = acc.Scalars(tag)
            vals = np.array([e.value for e in events], dtype=float)
            tail = vals[int(len(vals) * 0.8) :] if len(vals) > 5 else vals
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 2 else 0.0
            out[tag] = {
                "last": round(float(vals[-1]), 5),
                "min": round(float(vals.min()), 5),
                "max": round(float(vals.max()), 5),
                "step": int(events[-1].step),
                "n": len(vals),
                "tail_slope": round(slope, 6),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tags", default="")
    args = ap.parse_args()
    run = Path(args.run)
    if not run.exists():
        run = runs_dir() / args.run
    tags = [t for t in args.tags.split(",") if t] or None
    print(json.dumps(summarize(run, tags), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
