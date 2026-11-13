---
name: export-policy
description: Export a trained Dronez policy to ONNX with observation normalization and action scaling baked in, run PyTorch-vs-onnxruntime parity (≤1e-5) and latency (<50 µs) checks, write policy_meta.json, update docs/deployment_contract.md and regenerate releases/<version>/. Use when a checkpoint passes its gates and needs to be packaged, or when the deployment contract changes.
---
# Export a policy

```bash
uv run dronez-export --run <run_id> --checkpoint best --version v0.1.0
uv run pytest -m integration -k "onnx or parity" -q
```
`dronez-export` writes to `releases/<version>/`: `policy.onnx`, `policy_meta.json` (obs layout, normalization, action semantics, motor order, rates, latency budget, git SHA, config hash), `parity_report.json`, `latency_report.json`, and a copy of the eval report. **Never hand-edit `releases/`** (the PreToolUse hook blocks it); rerun the export.

## Checklist
1. Parity: max |Δ| ≤ 1e-5 on 10k random **and** 10k recorded observations (from eval rollouts).
2. Latency: p50/p99 per single inference on CPU (onnxruntime, 1 thread), target < 50 µs.
3. Obs normalization is frozen: the VecNormalize mean/var come from the same checkpoint step.
4. `docs/deployment_contract.md` is in sync with `policy_meta.json` (layout, units, frames, FLU/ENU → FRD/NED, motor-order mapping, rates, safety).
5. Ask `physics-auditor` to review the frames section (Opus budget).
6. Commit `releases/<version>/` and tag the version.
