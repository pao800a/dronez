---
name: training-diagnostics
description: Checklist for diagnosing Dronez PPO training — which TensorBoard scalars to read, healthy ranges, symptoms of reward hacking, entropy collapse, value divergence, action saturation and oscillatory policies, and the standard fixes. Use when a run plateaus, diverges, looks too good to be true, oscillates, or before deciding the next training change.
---
# Training diagnostics (SB3 PPO, Dronez)

Get a compact summary first (never read raw event files):
```bash
uv run python .claude/skills/training-diagnostics/scripts/tb_summary.py <run_id> --tags rollout,train,reward,curriculum,action
```

## Healthy ranges (starting heuristics, not laws)

| Scalar | Healthy | Red flag |
|---|---|---|
| `train/approx_kl` | 0.005–0.03 | > 0.05 sustained (lr too high / too many epochs); ~0 (not learning) |
| `train/clip_fraction` | 0.05–0.25 | > 0.4 |
| `train/entropy_loss` / `train/std` | slow decline | std collapses to ~0.05 early (premature determinism) or grows (reward too flat) |
| `train/explained_variance` | rises toward 0.8–0.99 | < 0 after the first ~1M steps (value can't fit; check reward scale and normalization) |
| `train/value_loss` | bounded, slowly falling | exponential growth (reward scale, missing `VecNormalize` on reward, termination bonus spikes) |
| `rollout/ep_len_mean` | → max episode length for attitude hold | stuck short (early terminations: check the crash/rate limits) |
| `reward/*` terms | attitude term dominates early; penalties a small fraction | one penalty dominates (the agent learns to do nothing) |
| `action/saturation_frac` | < 10 % at hover | > 30 % (bang-bang policy; won't transfer) |
| `action/mean_abs_delta` | falling | rising (oscillation) |
| `curriculum/level` | advances | stuck for > 20 % of the budget |

## Symptom → likely cause → fix
- **Return rises but the attitude error doesn't fall:** reward hacking through the alive bonus. Reduce the alive bonus, or make it conditional on error < threshold.
- **Policy hovers, refuses to recover from large tilts:** the curriculum is too fast or the termination too strict. Slow the promotion and widen the rate-termination limit.
- **High-frequency motor chatter:** raise the action-rate penalty weight, add a low-pass filter / last-action obs, check the policy rate vs τ_m.
- **Learns fine nominally, fails under DR:** widen DR gradually, or use an asymmetric critic (ADR needed).
- **Value loss explodes:** normalize rewards (`VecNormalize(norm_reward=True)`), clip rewards, lower `vf_coef`, check for NaN observations (inverted states → `arccos` domain).
- **KL spikes then collapse:** lower lr, fewer epochs, larger batch, `target_kl=0.02`.
- **Entropy collapses early:** raise `ent_coef` (1e-3 → 5e-3), init log_std higher, or use gSDE.

## Protocol
1. Diagnose with evidence (scalar, step, value).
2. Change **one** thing per attempt, recorded in `docs/PROGRESS.md`.
3. After **two failed attempts** on the same problem, escalate to `architect` (Opus) with the analyst's evidence.
