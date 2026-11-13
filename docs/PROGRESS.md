# Dronez progress

> Resume protocol: a new session continues from `CLAUDE.md` plus this file. Update it at the end of every phase, before every long-running job, and before compaction.

## Current phase
**Phase 0: Bootstrap** (branch `phase-0-bootstrap`). Started 2026-11-13.

## Done
- Environment checked (2026-11-13): WSL2 Ubuntu 24.04, i7-1255U (WSL capped at 4 vCPU / 6 GB by `.wslconfig`), Iris Xe (no `torch.xpu`: `torch.xpu.is_available() == False` on torch 2.15.0+cpu), Python 3.12, uv 0.13.6, gh logged in as `pao800a`.
- Owner decisions → ADR-0001…0004 (WSL2 primary; public `pao800a/dronez`; GitHub MCP via `gh auth token`; Context7 keyless).
- `pyproject.toml` + `uv.lock`: mujoco 4.0.1, torch 2.15.0+cpu (CPU index), gymnasium 1.4.0, stable-baselines3 2.9.2, build123d 0.13.0, onnx 1.23.2, onnxruntime 1.32.0, numpy 2.5.4, mcp.
- `src/dronez/utils/`: `rotations.py` (wxyz, cross-checked against `mju_*`), `paths.py`, `config.py`, `device.py`, with unit tests.
- `.claude/`: 12 agents, 7 skills, 8 hook scripts wired in `settings.json` with permissions; `tests/hooks/` (29 tests). The `pip install` block was verified live.
- `CLAUDE.md`, `docs/conventions.md` (motor diagram, sign table), `.mcp.json`, pre-commit, CI (ubuntu + windows), setup scripts.

## Next
1. Push the skeleton, open the Phase 0 PR, and get CI green on ubuntu and windows.
2. `architect` (Opus): `docs/architecture.md` + ADR-0005.
3. Phase 1: `cad-engineer` builds `configs/drone/default_5in.yaml` and `src/dronez/cad/`, then gate G1.

## Open problems / risks
- MuJoCo is **4.0.1** (new major). Check MjSpec, actuator and sensor API changes via Context7 before the sim phase.
- `GITHUB_PAT` is not exported yet, so the GitHub MCP is unauthenticated until the owner runs the one-time step (see the user action items). `gh` works meanwhile.
- WSL capped at 4 vCPU: the training throughput sizing waits for the Phase 4 SPS benchmark.

## Runs
_None yet._

## Opus usage log
| Date | Phase | Agent | Reason |
|---|---|---|---|
| 2026-11-13 | 0 | (main session) | The initial plan ran on Opus 5.5 (session default). |

## User action items
- [ ] `echo 'export GITHUB_PAT="$(gh auth token)"' >> ~/.bashrc`, then restart Claude Code (loads the GitHub MCP).
- [ ] Switch the main session model to `opusplan` or Sonnet (`/model`).
- [ ] Optional, before Phase 4: raise `.wslconfig` to `processors=10`, `memory=10GB`, then `wsl --shutdown`.
