---
name: evaluate-policy
description: Run the Dronez evaluation suite on a checkpoint or baseline controller — seeded scenarios (hover hold, step attitude, recovery bins <30°/30–90°/90–180°, impulses, wind, DR-extreme robustness sweep, latency stress) — generate the Markdown/HTML report with plots and MP4s, and compare against PID/geometric baselines. Use for gate checks G3–G6, after training, or when comparing runs.
---
# Evaluate a policy

```bash
uv run dronez-eval --run <run_id> [--checkpoint best|latest|<path>] --suite configs/eval/full.yaml
uv run dronez-eval --controller pid --suite configs/eval/full.yaml        # baseline rows
uv run dronez-eval --controller geometric --suite configs/eval/full.yaml
uv run dronez-eval --run <run_id> --suite configs/eval/quick.yaml        # 1–2 min sanity eval
```
Reports go to `$DRONEZ_DATA_DIR/eval/<run_id>/<checkpoint>/report.{md,html}` with `metrics.json`, plots and `videos/*.mp4`. Rendering needs an OpenGL backend: `MUJOCO_GL=egl` (or `osmesa`) on Linux/WSL.

## Metrics (all per scenario, seeded)
success rate · time to recover below 5° · RMS attitude error · overshoot · motor saturation % · mean |Δa| (smoothness) · action-spectrum energy above 20 Hz.

## Gates
| Gate | Criterion |
|---|---|
| G3 | PID and geometric stabilize from 30°, hover ±5 cm |
| G4 | 1,000 eps, nominal + sensor noise: hover RMS tilt < 2°; < 90° recovery ≥ 95 % success, < 1.5 s to < 5° |
| G5 | Including inverted starts and 5 rad/s rates: ≥ 90 %, < 2.5 s; no ground contact from ≥ 3 m |
| G6 | ≥ 90 % across the DR-extreme sweep + 20 ms latency; smoothness and HF energy ≤ PID baseline |

Report gate results with the path to `metrics.json`. Never quote numbers that aren't in a produced file. Look at at least one MP4 frame or plot with Read before claiming a gate.
