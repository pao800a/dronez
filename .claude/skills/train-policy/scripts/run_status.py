"""Print a short status of a Dronez training run (latest run if no id given)."""

from __future__ import annotations

import json
import sys

from dronez.utils.paths import runs_dir

KEYS = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/entropy_loss",
    "train/explained_variance",
    "time/fps",
)


def main() -> int:
    root = runs_dir()
    if len(sys.argv) > 1:
        run = root / sys.argv[1]
    else:
        runs = [p for p in root.iterdir() if p.is_dir()]
        if not runs:
            print("no runs")
            return 1
        run = max(runs, key=lambda p: p.stat().st_mtime)
    status = run / "status.json"
    print(f"run: {run.name}")
    if status.exists():
        print(json.dumps(json.loads(status.read_text(encoding="utf-8")), indent=1)[:1500])
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
            EventAccumulator,
        )

        for ev_dir in sorted({p.parent for p in run.rglob("events.out.tfevents.*")}):
            acc = EventAccumulator(str(ev_dir), size_guidance={"scalars": 0})
            acc.Reload()
            for key in KEYS:
                if key in acc.Tags().get("scalars", []):
                    ev = acc.Scalars(key)[-1]
                    print(f"{key}: {ev.value:.4g} @ {ev.step}")
    except Exception as exc:  # tensorboard optional for status
        print(f"(no scalars: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
