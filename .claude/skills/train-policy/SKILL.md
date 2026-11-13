---
name: train-policy
description: Launch Dronez RL training — smoke runs and multi-hour background runs with SB3 PPO — choose n_envs from the SPS benchmark, estimate wall-clock time, locate logs/checkpoints/TensorBoard events, check status, and resume a run. Use whenever training needs to be started, checked, resumed or sized.
---
# Train a policy

Runs live in `$DRONEZ_DATA_DIR/runs/<run_id>/`: `config.yaml` (resolved), `status.json`, `train.log`, `tb/`, `checkpoints/` (model zip **plus** VecNormalize stats, saved together).

## 1. Size the run
```bash
uv run python scripts/benchmark_sps.py --env DronezAttitude-v0 --n-envs 1,2,3   # writes $DRONEZ_DATA_DIR/bench/sps_*.json
```
Pick the `n_envs` with the best total SPS (usually physical cores − 2; in WSL limited by `.wslconfig`). Expected wall-clock = `total_timesteps / SPS_total`. **State that estimate before launching.** A stage must fit in ≤ 4–6 h.

## 2. Smoke run (foreground is fine, ≤ 200k steps)
```bash
uv run dronez-train --config configs/train/smoke.yaml
```

## 3. Full run: always in the background
Use the Bash tool with `run_in_background: true` (the PreToolUse hook asks for confirmation for > 200k foreground steps):
```bash
uv run dronez-train --config configs/train/attitude_stage1.yaml --run-id att-s1-001 > "$DRONEZ_DATA_DIR/runs/att-s1-001.stdout" 2>&1
```
Update `docs/PROGRESS.md` (run ID, config, expected end time) **before** launching.

## 4. Monitor without flooding context
- `uv run python .claude/skills/train-policy/scripts/run_status.py [run_id]` prints `status.json` and the latest key scalars (short).
- From Phase 4 on: the `dronez-sim` MCP tool `training_status(run_id)`.
- Diagnosis goes to `training-analyst` with the training-diagnostics skill. Long logs go to `log-summarizer`.

## 5. Resume
```bash
uv run dronez-train --config <same config> --run-id <run_id> --resume   # latest checkpoint + matching VecNormalize stats
```

## Device
CPU by default. `--device xpu` only if `torch.xpu.is_available()` and the ADR benchmark showed a speed-up.
