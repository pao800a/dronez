# Dronez progress

> Resume protocol: a new session continues from `CLAUDE.md` plus this file. Update it at the end of every phase, before every long-running job, and before compaction.

## Current phase
**Phase 2: Sim** (branch `phase-2-sim`). Started 2026-11-14.

Phase 1 complete: PR #2 merged as `1c00465` (2026-11-14); CI run 43570176840 green (lint, ubuntu, windows; same 514.3 g mass on both).

Phase 0 complete: PR #1 merged as `d6aea3e` (2026-11-13); CI run 43518516684 green (lint, ubuntu, windows).

## Done
- Environment checked (2026-11-13): WSL2 Ubuntu 24.04, i7-1255U (WSL capped at 4 vCPU / 6 GB by `.wslconfig`), Iris Xe (no `torch.xpu`: `torch.xpu.is_available() == False` on torch 2.15.0+cpu), Python 3.12, uv 0.13.6, gh logged in as `pao800a`.
- Owner decisions → ADR-0001…0004 (WSL2 primary; public `pao800a/dronez`; GitHub MCP via `gh auth token`; Context7 keyless).
- `pyproject.toml` + `uv.lock`: mujoco 4.0.1, torch 2.15.0+cpu (CPU index), gymnasium 1.4.0, stable-baselines3 2.9.2, build123d 0.13.0, onnx 1.23.2, onnxruntime 1.32.0, numpy 2.5.4, mcp.
- `src/dronez/utils/`: `rotations.py` (wxyz, cross-checked against `mju_*`), `paths.py`, `config.py`, `device.py`, with unit tests.
- `.claude/`: 12 agents, 7 skills, 8 hook scripts wired in `settings.json` with permissions; `tests/hooks/` (29 tests). The `pip install` block was verified live.
- `CLAUDE.md`, `docs/conventions.md` (motor diagram, sign table), `.mcp.json`, pre-commit, CI (ubuntu + windows), setup scripts.
- `docs/architecture.md` + ADR-0005 (architect, Opus): layering utils←cad←sim←control←envs←rl←eval←deploy; model cache `models/<name>-<hash>/` keyed by config + `CAD_VERSION`; MjSpec built in-process; one `act(obs, sensed)` controller interface; open questions in §Open questions of architecture.md.

- Phase 1 CAD (cad-engineer, 2026-11-14): `configs/drone/default_5in.yaml`, `src/dronez/cad/` (schema, artifacts, materials, parts, assembly, mass_props, export, render), `dronez-build-model` (CAD stage). G1 results from `$DRONEZ_DATA_DIR/models/default_5in-92c403696b/mass_props.json`: mass 0.5143 kg (target 0.55, -6.5 %), CoM (C frame) [0.06, 0.00, 24.7] mm, principal inertias 1.709e-3 / 1.894e-3 / 3.210e-3 kg·m². Renders checked (motor order and spin layout correct). `pytest -m "fast or cad"`: 92 passed. `CAD_VERSION` fingerprint test in `tests/unit/test_cad_version.py`.

- Phase 2 Sim (sim-engineer + test-engineer, 2026-11-14): `src/dronez/sim/` (SimConfig 1 kHz / decimation 10 / rk4, MjSpec builder, MotorModel with exact lag, rotor reaction + gyroscopic torque, ImuNoiseModel, DisturbanceModel, ActionDelay prefilled at reset, DroneSim), `configs/sim/default.yaml`, `tests/physics/` (30 tests). MuJoCo 4.0.1 facts verified by script: `mj_setConst` needed after mass/inertia/ipos edits (requires `body.simple = 0`); `xfrc_applied` is world-frame at the CoM; accelerometer = specific force; `mj_step` sensors are pre-step, so `true_state()` runs `mj_forward`. rk4 chosen: torque-free tumble energy drift 1.7e-8 vs 1.3 % (euler/implicitfast) over 10 s (`tests/unit/test_sim_plant.py::test_default_integrator_beats_euler_on_a_tumble`). Hover cmd 0.2864 (1.261 N/motor), T/W 9.53. G2: physics-auditor (Opus) found no HIGH; all 3 MEDIUM + 4 LOW findings fixed and re-checked by reading the diff. `CAD_VERSION` = 2 (principal-axis ordering).

## Next
1. Merge the Phase 2 PR on green CI.
2. Phase 3: `controls-engineer` builds `control/` (mixer from config, cascaded PID, geometric SO(3)), first `eval/` skeleton + `video.py`; gate G3 (both stabilize from 30°, hover ±5 cm). Use `DroneSim.trim_command()` semantics: equal-split hover drifts because the CAD CoM is 0.06 mm off the hub centroid (real physics; controllers must trim).

## Open problems / risks
- Propulsion constants in `default_5in.yaml` are **estimates** for a 2306 ~2450 KV motor on 4S (k_f 1.48e-6, k_m/k_f 0.016 m, ω_max 2850 rad/s, tau_m 30 ms → T_max 12.0 N/motor, T/W 9.53 per `dronez-build-model`). Replace with thrust-stand data before any sim-to-real claim.
- CAD mass is 6.5 % under target (no screws, wires or antenna). Use `measured_overrides` once the real AUW is known.
- MuJoCo 4: setting `MUJOCO_GL` makes `import mujoco` initialize that GL backend. `egl` without libEGL raises at import; on Windows `egl` is invalid. Set it only in render code or render jobs, never globally (CI run 43517234038).
- Project subagents created mid-session became available after a short delay (hot reload). If one is ever missing, delegate via `general-purpose` with an explicit `model` and the instruction to follow `.claude/agents/<name>.md`.
- MuJoCo is **4.0.1** (new major). Check MjSpec, actuator and sensor API changes via Context7 before the sim phase.
- `GITHUB_PAT` is not exported yet, so the GitHub MCP is unauthenticated until the owner runs the one-time step (see the user action items). `gh` works meanwhile.
- WSL capped at 4 vCPU: the training throughput sizing waits for the Phase 4 SPS benchmark.

## Runs
_None yet._

## Opus usage log
| Date | Phase | Agent | Reason |
|---|---|---|---|
| 2026-11-13 | 0 | (main session) | The initial plan ran on Opus 5.5 (session default). |
| 2026-11-13 | 0 | architect (via general-purpose, model=opus) | `docs/architecture.md` + ADR-0005 (planned call 1 of ~3). |
| 2026-11-14 | 2 | physics-auditor | G2 review of `src/dronez/sim/`, `cad/mass_props.py`, `tests/physics/` (Phase 2 call 1 of ~3). |

## User action items
- [ ] `echo 'export GITHUB_PAT="$(gh auth token)"' >> ~/.bashrc`, then restart Claude Code (loads the GitHub MCP).
- [ ] Switch the main session model to `opusplan` or Sonnet (`/model`).
- [ ] Optional, before Phase 4: raise `.wslconfig` to `processors=10`, `memory=10GB`, then `wsl --shutdown`.
