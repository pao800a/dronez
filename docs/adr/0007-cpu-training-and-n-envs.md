# ADR-0007: CPU training, 4 SubprocVecEnv workers, 1 torch thread

- **Status:** Accepted (2026-11-14)

## Context
BUILD_PROMPT asks for a one-off CPU-vs-XPU benchmark recorded in an ADR, and ADR-0006 §8 sets `n_envs` to the largest n whose marginal effective-SPS gain is ≥ 10 % in `scripts/benchmark_sps.py`. Machine: WSL2 on an i7-1255U with 4 vCPU and 6 GB (`.wslconfig`), and Iris Xe graphics. torch is 2.15.0+cpu.

Measured by `uv run python scripts/benchmark_sps.py` (defaults: n = 1–4, torch threads 1 and 2, 2 timed PPO rollouts of 16384 steps with the ADR-0006 hyperparameters, `info_level: minimal`). Source: `$DRONEZ_DATA_DIR/benchmarks/sps-20261114T131212Z.json`.

| Measurement | n = 1 | n = 2 | n = 3 | n = 4 |
|---|---|---|---|---|
| Raw single env, in-process | 2283 | | | |
| Collection (SubprocVecEnv, random actions) | 1392 | 2409 | 2288 | 2593 |
| PPO effective, 1 thread | 790 | 1472 | 1559 | **1809** |
| PPO effective, 2 threads | 837 | 1315 | 1443 | 1602 |

XPU was not measured. `torch.xpu.is_available()` is False on the CPU wheel, and the XPU wheels need the Intel GPU driver stack inside WSL2.

## Decision
- **Device: CPU.** `--device xpu` stays opt-in (`utils/device.py`), and nothing installs XPU wheels. The policy is a 64×64 actor with a 256×256 critic, and env stepping dominates the cost. An XPU gain would have to come from the update phase alone, which is not worth a driver stack for an unmeasured gain.
- **`n_envs = 4`** with `SubprocVecEnv`. The marginal gain is +86 % from n=1 to n=2, +6 % from 2 to 3, and +16 % from 3 to 4. n=3 also leaves a truncated 4095 minibatch.
- **Main-process torch threads: 1.** Two threads are slower at every n ≥ 2, because they compete with the env workers on 4 vCPU.

## Consequences
- The ADR-0006 effective-SPS estimate of about 3,000 at 4 vCPU was optimistic: the measurement is 1,809. Under the 4.5 h cap, `total_timesteps = min(budget, 4.5 h × 1809) ≈ 29.3M`. `att_p5` (40M budget) therefore needs about 6.1 h, or two chained runs, or more vCPUs.
- Collection is flat from n = 2 to n = 4 (2.4k–2.6k SPS): 4 workers plus the learner share 4 vCPU. The cheapest speed-up is the pending user action of raising `.wslconfig` to 10 processors. After that change, rerun the benchmark with `--n-envs 4 6 8` and amend this ADR.
- If SPS is still short after that, follow architecture §5.4 step 2 (profile, then a batched VecEnv).
- XPU would be revisited only if the policy grows (ADR-0006 escalation) and a profile shows the update phase dominating.
