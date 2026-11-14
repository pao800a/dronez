# ADR-0006: RL spec for `DronezAttitude-v0` and `DronezHover-v0`

- **Status:** Accepted (2026-11-14)
- **Supersedes:** the "tentative" rates in [architecture §11](../architecture.md#11-rates), the tentative `command_map` in architecture §5.2, and open questions [§13.4](../architecture.md#13-open-questions-for-implementers) (command map) and §13.7 (CTBR gyro).
- **Refines:** [ADR-0005](0005-architecture-and-layering.md). It also revises the definitions of gates G4–G6 in [`BUILD_PROMPT.md` §8](../BUILD_PROMPT.md) (the thresholds stay, the definitions become exact; see §9).
- **Conventions:** [`conventions.md`](../conventions.md) applies throughout: SI units, ENU world / FLU body, wxyz quaternions, geodesic error, no Euler angles in the obs or the reward, motor order from the YAML.
- **Owners:** rl-engineer (`envs/`, `rl/`, `eval/` metrics and gates), controls-engineer (`control/command_map.py`, baseline action mapping), test-engineer (§10 tests).

## Context

Phase 4 builds the Gymnasium envs and the PPO pipeline. Everything below is fixed now because it shapes the ONNX contract, the eval gates and the training budget.

Measured facts this ADR uses (anything else is labelled **estimate**):

| Fact | Value | Source |
|---|---|---|
| Physics | 1 kHz, rk4, decimation 10; rk4 tumble energy drift 1.7e-8 vs 1.3 % (euler) | `configs/sim/default.yaml`, `tests/unit/test_sim_plant.py` |
| Hover command / T/W | 0.2864 / 9.53, from **estimated** propulsion constants | `dronez-build-model`, `docs/PROGRESS.md` |
| Motor | τ_m 30 ms, ω ∈ [150, 2850] rad/s, k_f 1.48e-6, k_m/k_f 0.016 m, rotor inertia modelled (all estimates) | `configs/drone/default_5in.yaml` |
| Command map in the plant | `MotorModel.command_to_omega` is **linear in ω**: ω* = ω_min + u(ω_max − ω_min) | `src/dronez/sim/motor_model.py` |
| Baselines (G3) | 30° recovery worst case 0.15 s (pid) / 0.17 s (geometric); hover max error 0.44 / 0.45 cm; 0 % saturation | `$DRONEZ_DATA_DIR/eval/baseline-{pid,geometric}/c5762191dc-800e194166/metrics.json` |
| Baseline hover `hf_energy_frac_gt20hz` | ≈ 0.71 with \|Δu\| ≈ 0.003 (noise-driven) | `docs/PROGRESS.md` (open problems) |
| Raw throughput | ≈ 1750 policy steps/s, single process, `DroneSim` only, WSL 4 vCPU, ~57 µs per substep | `docs/PROGRESS.md` (ad-hoc script) |

Derived numbers below (envelope thrust, torques) are computed from these estimated constants and must be recomputed if the YAML changes.

## Decision

### 1. Rates and latency

| Item | Decision | Reason |
|---|---|---|
| Physics dt | **1 ms**, rk4 | Measured drift evidence. Contacts only trigger termination (§5), so contact stiffness is irrelevant |
| Policy rate | **100 Hz**, decimation 10 | Motor pole 1/(2π·30 ms) ≈ 5.3 Hz and the PID attitude bandwidth ≈ 5 Hz (`configs/control/pid.yaml`): 100 Hz is ~20× the bandwidth, and the 10 ms ZOH (5 ms mean delay) is well under τ_m. The cost per simulated second is set by the physics, so 200 Hz would double the samples per second of flight, need γ ≈ 0.995 and make credit assignment harder. Nyquist (50 Hz) stays above the 20 Hz HF cutoff |
| Latency | Action delay, whole **physics** steps (`ActionDelay`), configured in **seconds** and converted with `round(L / physics_dt)` | dt-agnostic config. Sensor-side delay is not modelled separately: for this loop it is equivalent to the same amount of action delay (open question O6) |
| `latency_budget_s` (policy_meta) | 0.020 s, the trained maximum | The deployment's sensor-to-motor latency must not exceed it. ≤ 10 ms is recommended |

Pre-authorized alternatives (no new ADR needed; record the evidence in `docs/PROGRESS.md` and add one line to this ADR's Status):
- **500 Hz physics** (dt 2 ms, decimation 5). Substeps dominate the cost, so the expected gain is ~1.7× (**estimate**). Switch only if all three hold:
  1. `pytest -m physics` passes with a `configs/sim/fast500.yaml`;
  2. the G3 baseline suite metrics stay within 10 % of the 1 kHz values;
  3. the benchmark shows a ≥ 1.4× effective-SPS gain.
- **200 Hz policy:** only if G4 fails on RMS tilt and the training analyst attributes it to bandwidth.

### 2. Observation

Obs are raw SI float32, laid out by `ObsLayout`. They are built **only from `Sensed`**, the target and the action history, never from the true state. Normalization is done by VecNormalize with **frozen fixed statistics** (below).

**Attitude layout, D = 29**

| Slice | Name | Dim | Units | Frame | Definition | Offset | Scale |
|---|---|---|---|---|---|---|---|
| 0:6 | `r_err_6d` | 6 | 1 | target ← body | `R_err = rotations.rotation_error_matrix(q̂, q_t)` (= R_tᵀR̂), columns 0 and 1, column-major: `[R00,R10,R20,R01,R11,R21]`. Identity error gives `[1,0,0,0,1,0]` | 0 | 1 |
| 6:9 | `gravity_b` | 3 | 1 | body FLU | `rotations.gravity_in_body(q̂)` | 0 | 1 |
| 9:12 | `gyro_b` | 3 | rad/s | body FLU | `Sensed.gyro_b` (noise + bias) | 0 | 5 |
| 12:13 | `vel_z_w` | 1 | m/s | world z (up) | `Sensed.vel_w[2]` (estimator abstraction) | 0 | 2 |
| 13:29 | `action_hist` | 16 | 1 | motor index | `[a_{t-1}, a_{t-2}, a_{t-3}, a_{t-4}]`, each a 4-vector in M0..M3 order: the **clipped, applied** actions, newest first. Prefilled with `a0` at reset | 0 | 1 |

**Hover layout, D = 34** = the Attitude layout 0:29, plus:

| Slice | Name | Dim | Units | Frame | Definition | Offset | Scale |
|---|---|---|---|---|---|---|---|
| 29:32 | `pos_err_h` | 3 | m | target heading frame | `quat_rotate_inverse(q_t, p̂ − p_t)` (q_t is yaw-only in Hover) | 0 | 1 |
| 32:34 | `vel_xy_h` | 2 | m/s | target heading frame | `quat_rotate_inverse(q_t, v̂)[:2]`. The z component is already in slot 12 | 0 | 2 |

Justification:
- **6D rotation error.** It is continuous over all of SO(3), with no double cover and no gimbal lock, so it is valid when inverted.
- **Heading invariance.** `r_err_6d` and `gravity_b` together determine the target's tilt (target z in body = `R_errᵀ e3`), so the BUILD_PROMPT "target attitude" input is present without the world yaw. The dynamics are yaw-invariant, so the policy never has to learn that the world heading doesn't matter.
- **K = 4.** 40 ms of action history covers the motor lag (30 ms, randomized up to 60 ms) plus latency (up to 20 ms) well enough to infer the rotor state, which is not observed.
- **`vel_z_w`.** It allows the "softly regulated" altitude of the Attitude env and G5's "no ground contact from 3 m". It needs a vertical-velocity estimate on hardware (baro + accel EKF); that goes into the deployment contract.
- **No accelerometer, no world position** in Attitude: they are noisy or biased, and an angle-mode deployment does not need them. They remain in `info`.
- **Prefix-compatible layouts.** Hover = Attitude + 5 dims, so an Attitude actor can seed a Hover actor by widening the first layer (optional, O10).

**Normalization: frozen fixed scales, not running statistics.**
- With a curriculum, VecNormalize running stats are fitted to level 0 (tiny rates, `gravity_b ≈ (0,0,-1)` almost constant). Later levels then come out at many σ and are clipped at ±10. Because the stats are count-weighted, they re-adapt only slowly after tens of millions of level-0 samples, and near-constant features amplify noise.
- Fixed physical scales avoid this. They are deterministic, identical across stages and seeds, and trivial to state in the contract.
- Implementation: `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10, gamma=γ)` with `obs_rms` replaced by `dronez.rl.normalization.FixedRunningMeanStd(mean=offset, var=scale²)`. Its `update`/`update_from_moments` are no-ops and `copy()` keeps the class. This was verified against SB3 2.x: `step_wait` and `reset` call `obs_rms.update` when `training`, `_normalize_obs` = `clip((x−mean)/sqrt(var+ε), ±clip_obs)`, and `save` pickles the whole object, so the class must stay importable.
- As a result, `vecnormalize.pkl` still carries μ, σ², ε and clip, and the ONNX graph of architecture §9 is unchanged.
- Reward normalization stays adaptive (`ret_rms`).

**Privileged obs.** `info["privileged_obs"]` is a raw SI float32 vector with a fixed layout (`ObsLayout.privileged`). It is computed **only if** `EnvConfig.privileged_obs: true` (default false, to save SPS). Order:
1. true `r_err_6d` (6), `gravity_b` (3), `angvel_b` (3), `linvel_w` (3), `pos_err_h` (3, zeros in Attitude);
2. `omega/omega_max_nom` (4);
3. `mass_scale` (1), `inertia_scale` (3), `com_offset_b_m` (3), `k_f_scale` per motor (4), `k_m_scale` (1), `tau_m_s` (1), `latency_s` (1), `battery_scale` (1);
4. disturbance `force_w_n` (3), `torque_w_nm` (3), `wind_w_m_s` (3).

D = 46. **Asymmetric critic: not in Phases 4–5.** Open one in Phase 6, through its own ADR, only if at d ≥ 0.75 `explained_variance` stays < 0.7 or the G6 failures correlate with unobservable parameters.

### 3. Actions

**Motors mode (default, gated).** `a ∈ [-1,1]^4` in motor index order maps to the normalized command:

```
u = clip(u_h + Δu · clip(a, -1, 1), 0, 1),   u_h = hover_command(nominal motor, nominal mass) (scalar, = 0.2864 now),
Δu = authority · min(u_h, 1 − u_h),   authority = 1.0 (config)   →  u ∈ [0, 0.573]
```

- **Resolves §13.4: linear in ω**, because u is affine and `MotorModel` is linear in ω.
  - It matches what a DShot/PWM throttle approximates (ω, not thrust).
  - It needs no sqrt in the ONNX graph.
  - It does not depend on the uncertain k_f.
  - A thrust-linear map would bake the estimated k_f into the contract.
  - The policy learns the ω² nonlinearity. dT/da is smooth and monotone (≈ 0.34 / 2.11 / 3.88 N at a = −1 / 0 / +1, **estimate**), with no kink at hover. A piecewise map would have an 8.7× gain jump at a = 0.
- **The symmetric envelope is deliberate.**
  - a = 0 is nominal hover, so the untrained policy's output does not shoot the drone upward. With `(a+1)/2` the initial T/W would be ≈ 2.6.
  - Nominal T/W at a = +1 is ≈ 3.38. The worst DR corner (mass +20 %, k_f −15 %, mismatch −5 %, battery 0.85) leaves ≈ 1.7, and its hover needs a ≈ +0.50 (**estimates**).
  - Torque authority is ample: roll ≈ 0.75 N·m (≈ 437 rad/s²), yaw ≈ 0.135 N·m (≈ 42 rad/s²), **estimates**.
  - Real motor saturation (u = 1) is never commanded, which is a deployment safety property. `authority` may be raised to 1.0 of `1 − u_h` above hover only by amending this ADR.
- **`CommandMap`** lives in `src/dronez/control/command_map.py` (owner controls-engineer). It is built from the **nominal** `MotorParams` and the nominal mass, so it is never randomized and always equals the ONNX constants.
  - API: `action_to_command(a)`, `command_to_action(u)` (clipped to [−1, 1]), `thrust_limits(motor) -> (T_lo, T_hi)` = thrust at a = ∓1.
  - `policy_meta.action.command_map = {type: "affine_omega", u_hover, delta_u, authority}`.
- **Baselines** return actions in the same space:
  - `control.base.thrusts_to_command` becomes `thrusts_to_action(thrusts, motor, cmap) = cmap.command_to_action(motor.thrust_to_command(thrusts))`.
  - The baseline `Mixer` limits become `cmap.thrust_limits(nominal)`, so saturation handling respects the envelope instead of being clipped after the fact.
  - Baseline rows are **regenerated** under this envelope. G3 stays passed as recorded.

**CTBR mode (specified; implementation optional in Phase 4; not gated).**
- `a = [a_T, a_p, a_q, a_r]`. Collective `T = W_nom (1 + a_T)` ∈ [0, 2 W_nom]; rate setpoint `ω_sp = (a_p, a_q, a_r) ⊙ (10, 10, 6)` rad/s, body FLU.
- Inside the substep loop (1 kHz): `control.RateController` (gains from `configs/control/pid.yaml`) and the nominal `Mixer` with the envelope limits, then `thrust_to_command`, then the motors.
- **Resolves §13.7:** the inner loop uses the **noisy gyro sampled at the physics rate**, from the same `ImuNoiseModel` state (one sensor, one bias). The obs `gyro_b` is the mean of the `decimation` samples of the step. This has the same noise statistics as one sample at the policy dt.
- The ONNX output 2 becomes `[T_norm, p, q, r]` (FLU); the contract adds `q_FRD = −q`, `r_FRD = −r`.

**Controller.act** is unchanged (architecture §7). The env clips again (SB3 clips Gaussian samples, but baselines and eval bypass SB3). The clipped action goes into the history, the reward and the trajectory. Trajectories record both `a` and `u = cmap(a)`.

### 4. Reward

`r = Σ terms`, computed from the **true** state. `info["reward_terms"]` holds the **weighted** contributions and sums to `r` exactly (test). Defaults below; the weights marked † must be tuned empirically.

| Term key | Contribution | Default | Per-step range | ≈ at good hover (estimate) |
|---|---|---|---|---|
| `alive` | +w | 1.0 | +1 | +1 |
| `att` | −w · e/π, e = geodesic(q, q_t) | 0.5 | [−0.5, 0] | −0.006 (e = 2°) |
| `att_fine` | +w · exp(−(e/σ_e)²), σ_e = 0.10 rad† | 0.5 | [0, 0.5] | +0.44 |
| `rate` | −w · min((‖ω_b‖/5 rad/s)², 1) | 0.1† | [−0.1, 0] | ≈ 0 |
| `vz` (Attitude) | −w · min((v_z/1 m/s)², 1) | 0.1† | [−0.1, 0] | ≈ 0 |
| `act_mag` | −w · mean_i a_i² | 0.05† | [−0.05, 0] | −0.002 |
| `act_smooth` | −w · mean_i (a_i − a_{t−1,i})² | 0.25† | [−1, 0] | ≈ −3e-5 |
| `act_sat` | −w · mean_i 1[\|a_i\| ≥ 0.99] | 0.1 | [−0.1, 0] | 0 |
| `penalty_clip` | `max(P, −1.0) − P`, P = sum of the penalty rows (`att`, `rate`, `vz`/`vel`, `pos`, `act_*`) | — | ≥ 0 | 0 |
| `crash` | −w on a `terminated` step (any reason in §5) | 10† | {−10, 0} | 0 |

Hover replaces `vz` and re-weights:

| Term key | Contribution | Default |
|---|---|---|
| `att` / `att_fine` | as above | 0.2 / 0.2, σ_e = 0.2 rad (tilt is needed to translate) |
| `pos` | −w · min(‖e_p‖/2 m, 1) | 0.5 |
| `pos_fine` | +w · exp(−(‖e_p‖/0.10 m)²) | 0.5† |
| `vel` | −w · min(‖v_w‖²/(1 m/s)², 1) | 0.1† |

Everything else is identical. The maximum step reward is 1.5 (Attitude) / 1.7 (Hover).

Anti-hacking:

| Exploit | Countermeasure |
|---|---|
| Early suicide | The penalty sum is clipped at −1, so `r ≥ 0` on every non-terminal step, and `crash` adds −10. Termination therefore always loses value (test) |
| Time-limit exploitation | `truncated` (time limit only) is bootstrapped by SB3 (`TimeLimit.truncated` + `terminal_observation`, verified) |
| Bang-bang or mean drift past ±1 | `act_sat`, `act_smooth`. Eval gates HF on the deterministic action |
| Flying away in Attitude | `vz` plus out-of-bounds termination |
| Ground exploits | Termination on any contact |
| Noise tracking (chatter) | `act_smooth`. The G6 amplitude-aware HF gate (§9) |
| Obs/reward mismatch | The reward uses the true state while the obs are sensed, so noise cannot be "gamed" |

TensorBoard tags (rollout means of the info values):
- `reward/<term>`;
- `episode/success_rate`, `episode/termination/<reason>`;
- `curriculum/level`, `curriculum/dr_scale`;
- `action/saturation_frac`, `action/mean_abs_delta`;
- `obs/clip_frac` (fraction of normalized entries at ±10; a scale alarm when > 1e-3);
- `metrics/att_err_rad`.

### 5. Termination and truncation

Checks run after **every substep**; they are cheap attribute reads. `info["termination"]` uses the architecture §6 enum.

| `termination` | Condition | Attitude | Hover |
|---|---|---|---|
| `ground_contact` | `data.ncon > 0` (the only collidable pair is drone–ground) or CoM z < 0.05 m | yes | yes |
| `rate_limit` | ‖ω_b,true‖ > 25 rad/s | yes | yes |
| `out_of_bounds` | Attitude: horizontal distance from the start > 20 m or z > z0 + 10 m. Hover: ‖p − p_t‖ > 3 m | yes | yes |
| `crash` | a non-finite qpos/qvel/obs. The returned obs is the last finite one | yes | yes |
| `time_limit` (`truncated`) | step count = `episode_len_s / policy_dt` | 5.0 s (500 steps) | 8.0 s (800 steps) |

There is no tilt-based termination: inverted is a valid state. `eval/runner.py`'s `CRASH_TILT_RAD` rule must go; in eval, crashed means `terminated`.

### 6. Curriculum

One integer level `k` per env, set through `VecEnv.env_method("set_curriculum_level", k)`. It takes effect at each env's next reset.

**Promotion** (`CurriculumCallback`, `rl/callbacks.py`):
- Every `eval_every = 1M` steps, run a **deterministic** mini-eval at the current level: 100 episodes, fixed seeds per round, on a separate `SubprocVecEnv` with `n_envs` workers that is idle during training.
- Promote when the eval success rate is ≥ the threshold in **2 consecutive** evals and ≥ `min_steps` have been spent at the level.
- No demotion. If the level's `max_steps` is exceeded, log `curriculum/stalled = 1` and continue; the training analyst then decides.
- Success in training and the mini-eval means: not terminated, and e < 5° at every step of the final 1.0 s.
- Stochastic training episodes are not used for promotion: exploration noise would gate on σ instead of on skill.

**Attitude levels**

| k | Run | Initial error δ (cos-uniform cap) | ω0 (\|ω0\| ~ U, random direction) | v0 ball | a0 ~ U(±) | Target mix | DR scale d | Disturbances | Promote at SR ≥ | min / max steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `att_p5` | ≤ 15° | ≤ 0.5 rad/s | 0 | 0 | heading hold | 0 | off | 0.90 | 1M / 6M |
| 1 | `att_p5` | ≤ 45° | ≤ 1.5 | 0 | 0.2 | + tilted 20 %, step 20 % | 0.10 | off | 0.90 | 2M / 8M |
| 2 | `att_p5` | ≤ 90° | ≤ 3.0 | 1 m/s | 0.2 | same | 0.25 | off | 0.90 | 2M / 12M |
| 3 | `att_p5` | ≤ 180° (all of SO(3)) | ≤ 5.0 | 1 m/s | 0.2 | same | 0.25 | off | — (end of Phase 5) | — |
| 4 | `att_p6_dr` | ≤ 180° | ≤ 5.0 | 1 m/s | 0.2 | same | 0.50 | × d | 0.85 | 3M / 12M |
| 5 | `att_p6_dr` | ≤ 180° | ≤ 5.0 | 1 m/s | 0.2 | same | 0.75 | × d | 0.85 | 3M / 12M |
| 6 | `att_p6_dr` | ≤ 180° | ≤ 5.0 | 1 m/s | 0.2 | same | 1.00 | × d | — (G6 candidate) | — |

**How a level maps to the init distribution** (all from `rng_init`, in this order):
1. With p = 0.2 (k ≥ 1), sample the task parameters from a uniformly chosen lower level. This prevents forgetting and keeps near-level samples for G4. DR stays at the current d.
2. `z0 ~ U(3, 5)` m, xy = 0.
3. Target:
   - heading hold `q_t = quat_from_yaw(ψ)`, ψ ~ U(−π, π);
   - or, if tilted, `q_t = q_tilt(cos-uniform ≤ 20°, random horizontal axis) ⊗ quat_from_yaw(ψ)`.
4. Initial attitude `q0 = q_δ ⊗ q_t`, where `q_δ` rotates about a random horizontal **world** axis by δ, with cos δ ~ U(cos δ_max, 1). Then geodesic(q0, q_t) = δ, the same construction as `eval/scenarios.py`.
5. ω0, then v0.
6. `a0`: the history is prefilled with `a0` and `ω0_rotor = command_to_omega(cmap(a0))` under the drawn params.
7. A target step (if drawn): at `t_s ~ U(1, 3)` s to a new target within 30° of the old one.

**Hover levels** (`hover_p7`, Phase 7):

| k | Offset ball | v0 | δ | ω0 | Initial heading vs ψ_t | d |
|---|---|---|---|---|---|---|
| H0 | 0.2 m | 0.2 m/s | 10° | 0.5 rad/s | ±15° | 0 |
| H1 | 0.5 m | 0.5 m/s | 30° | 1.5 rad/s | ±45° | 0.25 |
| H2 | 1.0 m | 1.0 m/s | 60° | 3 rad/s | ±180° | 0.5 |
| H3 | 1.0 m | 1.0 m/s | 60° | 3 rad/s | ±180° | 1.0 |

The target is `p_t = (0, 0, z_t)`, `z_t ~ U(1.5, 3)` m. Promotion uses SR ≥ 0.9, where success = no termination and ‖e_p‖ < 0.10 m over the final 1 s.

### 7. Domain randomization

**Sampling.**
- `rng_dr` draws a **fixed-length** block `U = rng_dr.random(N_DR)` once per reset, independent of d. A fixed index table in `envs/randomization.py` maps each parameter to its uniforms; new parameters are appended, never reordered.
- Each parameter uses a unit variate z ∈ [−1, 1] (`2U−1`). With probability `p_extreme = 0.15` (its own uniform), z is replaced by its sign (±1), so DR corners are actually visited.
- A seed therefore perturbs "in the same direction" at every d, and d = 0 reproduces nominal exactly (test).

**Full widths (d = 1)**, applied as `nominal × (1 + d·h·z)` unless stated otherwise:

| Parameter | Rule at width d | d = 1 range | Applied by |
|---|---|---|---|
| mass | h = 0.20 | ±20 % | `body_mass` |
| inertia, common | f_c: h = 0.20 | ±20 % | principal moments × f_c(1 + δ_k) |
| inertia, anisotropy δ_k (3) | h = 0.05 per axis | ±5 % | With the 12 % nominal triangle margin (1.709 + 1.894 vs 3.210 e-3), ±5 % can never break `I_a + I_b ≥ I_c`. Still check `inertia_triangle_ok`; if it fails, resample δ (≤ 10 tries), then `project_inertia_triangle` |
| CoM offset | `d · z · (3, 3, 5)` mm | x, y ±3 mm; z ±5 mm | `body_ipos` |
| k_f, common | h = 0.15 | ±15 % | `MotorParams.k_f` and the gear |
| k_f, per-motor mismatch (4) | h = 0.05 | ±5 % | same |
| k_m | h = 0.20 | ±20 % | gear `−s_i k_m/k_f` |
| τ_m | log-uniform: 30 ms · 2^(d·z) (common), × (1 + 0.1·d·z_i) per motor | 15–60 ms | `MotorParams.tau_m_s` |
| latency | `round(d · 20 ms · U)` (edge: U = 1) | 0–20 ms | `ActionDelay` |
| battery (ω_max scale) | U(1 − 0.15d, 1 + 0.05d) | 0.85–1.05 | `PhysParams.battery_scale` |
| sensor noise scale (gyro/accel/quat/vel densities and bias std) | 2^(d·z) | ×0.5–×2 | `ImuNoiseConfig` per episode |
| wind (levels ≥ 4) | mean U(0, 3d) m/s horizontal, random direction; gust std U(0, 1d) m/s | ≤ 3 m/s, ≤ 1 m/s | `DisturbanceModel` (`rng_dist`) |
| impulses (levels ≥ 4) | count U{0..2} per episode, 20 ms, force U(0, 2d) N, torque U(0, 0.12d) N·m, random directions | the eval `impulse_small` size | `DisturbanceModel` (`rng_dist`) |

Rotor drag and ground effect stay **off**, which keeps the open-loop motor shortcut of architecture §5.4. `DomainRandomizer.apply` always writes from the nominal snapshot (ADR-0005). At d = 0, sensor noise is the nominal `ImuNoiseConfig()` (**on**), which is the G4 condition.

**Widening schedule:** d is tied to the curriculum level (§6 table), so DR widens only after the task is mastered. That gives 0 → 0.10 → 0.25 in Phase 5 and 0.5 → 0.75 → 1.0 in Phase 6.

### 8. PPO

| Parameter | Value | Reason |
|---|---|---|
| Policy | `MlpPolicy`, `net_arch = dict(pi=[64, 64], vf=[256, 256])` | Spec. The small actor exports cheaply and to C. The critic must absorb curriculum and DR variance. Escalation: pi [128, 128] |
| `activation_fn` | `nn.Tanh` (SB3 default, verified) | Smooth, bounded features, which helps the HF metric |
| Action distribution | DiagGaussian, `log_std_init = −1.0` (σ ≈ 0.37)†, `ortho_init=True`, `use_sde=False` | Simplest correct path. `squash_output` requires gSDE in SB3 (verified assert). The deterministic action is `clip(mean)`, the same in the ONNX graph. **Fallback** if the G6 smoothness fails: `use_sde=True, sde_sample_freq=4` |
| `n_envs` | 4 (4 vCPU) / 8 (10 vCPU). Final value = the largest n whose marginal effective-SPS gain is ≥ 10 % in `scripts/benchmark_sps.py` | Architecture §5.4 |
| `n_steps` | `16384 // n_envs` (4096 / 2048) | ≥ 4 episodes per env per rollout, for GAE |
| `batch_size` | 4096 (4 minibatches) | Low-variance gradients, cheap on CPU |
| `n_epochs` | 5 | Bounds update time and KL |
| `learning_rate` | 3e-4 constant (`att_p5`); 1e-4 constant (`att_p6_dr`, `hover_p7` fine-tune) | A linear decay would starve late curriculum levels |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | Horizon ≈ 1 s at 100 Hz, matching the 1–2.5 s recovery scale |
| `clip_range` / `clip_range_vf` | 0.2 / None | Standard |
| `ent_coef` | 0.0 | A state-independent log-std is learned; an entropy bonus inflates σ |
| `vf_coef` / `max_grad_norm` | 0.5 / 0.5 | Standard |
| `target_kl` | 0.03 | Guards against destructive updates (SB3 stops at 1.5 × target_kl) |
| VecNormalize | §2: frozen obs, `norm_reward=True`, `gamma=0.99`, `clip_reward=10` | — |
| torch threads (main process) | 1 (the benchmark compares 1 vs 2) | Workers: `OMP/MKL_NUM_THREADS=1` (architecture §5.1) |
| Stage hand-off | `TrainConfig.init_from: {run_id, checkpoint}` loads `model.zip` + `vecnormalize.pkl` (keeping `ret_rms`) and `curriculum.start_level` | Each stage is its own run, ≤ 6 h |

**Budgets.**

The effective SPS (collection + updates + mini-evals) is an **estimate**: ≈ 3,000 at 4 vCPU (1750 raw × ~0.8 env overhead × ~3 effective workers × ~0.75 for updates and IPC) and ≈ 7,000 at 10 vCPU / 8 envs. Replace both with the `benchmark_sps.py` result. Set `total_timesteps = min(budget, 4.5 h × measured SPS)` and state the ETA before every launch.

| Config | Levels | Budget (steps) | Wall-clock @ 3k / 7k SPS (estimate) |
|---|---|---|---|
| `configs/train/smoke.yaml` | 0 | 8,192 | < 1 min including startup (CI target ≤ 2 min) |
| `configs/train/att_p5.yaml` | 0→3 | 40M (level shares ≈ 4/6/10/20M, estimate) | 3.7 h / 1.6 h |
| `configs/train/att_p6_dr.yaml` | 4→6, init from the `att_p5` best | 30M | 2.8 h / 1.2 h |
| `configs/train/hover_p7.yaml` | H0→H3 | 30M | 2.8 h / 1.2 h |

If the measured SPS < 2,500, first cut `info_level` and mini-eval overhead, then try the 500 Hz switch (§1), then the batched VecEnv (architecture §5.4). Do not cut `att_p5` below 30M without a training-analyst review.

**Smoke config (CI):**
- Settings: `DummyVecEnv`, `n_envs 2`, `n_steps 1024`, `batch_size 256`, `n_epochs 2`, `total_timesteps 8192`, `episode_len_s 2.0`, level 0, mini-eval of 4 episodes at the end.
- Asserts:
  1. the run reaches `state: finished`;
  2. `step_*/` holds `model.zip` + `vecnormalize.pkl` + `ckpt.json`;
  3. TensorBoard has every `reward/<term>` tag;
  4. the reloaded checkpoint gives a finite deterministic action;
  5. `obs_rms.mean/var` still equal the ObsLayout offsets/scales².
- No learning-curve asserts.

### 9. Gates G4–G6 (exact definitions)

These are computed by `eval/metrics.py` and checked by `eval.check_gate(metrics, gate, baseline=None)`. The thresholds live in `configs/eval/gates.yaml` (pydantic) and change only by ADR. The suites are `configs/eval/{g4,g5,g6}.yaml`.

**Common conditions:**
- deterministic policy;
- paired seeds `SeedSequence([suite_seed, scenario_idx, episode_idx])`;
- unless stated otherwise: nominal plant (d = 0), latency 0, nominal sensor noise on, no disturbances;
- start z0 = 3.0 m at rest, `a0 = 0`;
- heading-hold target `q_t = quat_from_yaw(ψ)`, ψ ~ U(−π, π), and `q0 = q_δ ⊗ q_t`, so the initial geodesic error equals the tilt δ;
- crashed = env `terminated`;
- `e` is the true geodesic error, which includes yaw.

New metric functions:
- `success_recovery(e, dt, t_max, crashed) := not crashed and time_to_recover(e, dt) ≤ t_max`. The existing `time_to_recover` already requires the error to stay below 5° until the end.
- `hf_rms(u, rate_hz, f_cut=20, t_start=1.0)`: per motor, take u from `t_start` on, remove a linear detrend, `rfft`, zero the bins with f ≤ f_cut, `irfft`; return the RMS over time and motors. Units: normalized motor command u.
- `mean_abs_delta(u)` on u from `t_start = 1.0`.
- `hf_energy_fraction` is kept but **report-only**, and NaN when the detrended AC RMS is < 1e-3. Noise-driven tiny actions make a scale-free fraction meaningless: this is the PID's 0.71.

| Gate | Scenario (n episodes, duration) | Criterion key | Threshold |
|---|---|---|---|
| **G4** | `hover_hold` (200, 10 s): δ = 0, ω0 = 0 | `hover_rms_tilt_rad` = sqrt(mean of `tilt_angle`² pooled over episodes and samples, t ≥ 1 s) | < 0.0349 (2°) |
| | | `hover_crash_count` | = 0 |
| | `recovery_lt90` (800, 4 s): δ cos-uniform in [0°, 90°), ω0 = 0 | `recovery_lt90_success_rate`, success = `success_recovery(t_max = 1.5 s)` | ≥ 0.95 |
| **G5** | `recovery_full` (1000, 5 s): q0 uniform over SO(3) (cos δ ~ U(−1, 1)), \|ω0\| ~ U(0, 5) rad/s with random direction, z0 = 3.0 m | `recovery_full_success_rate`, success = `success_recovery(t_max = 2.5 s)` | ≥ 0.90 |
| | | `ground_contact_count` (`termination == ground_contact`) | = 0 |
| | | `g4_pass` (the same checkpoint) | true |
| **G6** | `sweep`: the `recovery_lt90` task (4 s), 50 episodes per fixed config × 19 configs, plus `dr_random` (200 episodes, a fresh d = 1 draw each, latency 20 ms) | `sweep_success_rate_pooled`, success = `success_recovery(t_max = 2.5 s)` | ≥ 0.90 |
| | | `sweep_success_rate_min_config` | ≥ 0.80 |
| | | `latency20_success_rate` (the `latency_20ms` config) | ≥ 0.90 |
| | `hover_hold` (G4 conditions) and `hover_hold_dr` (100, 10 s, d = 1 draws, latency U(0, 20 ms)); PID run on the same env, seeds and noise | `hf_rms_u` for each scenario | ≤ PID + 0.001 |
| | | `mean_abs_delta_u` for each scenario | ≤ PID + 0.001 |

**G6 fixed configs (19).** All other parameters are nominal and latency is 0 unless listed.

One at a time (17):

| Parameter | Configs |
|---|---|
| mass | 0.8, 1.2 |
| inertia common | 0.8, 1.2 |
| CoM | (+3, +3, +5) mm and (−3, −3, −5) mm |
| k_f | 0.85, 1.15 |
| k_m | 0.8, 1.2 |
| τ_m | 15 ms, 60 ms |
| mismatch | `roll` (M0, M3 −5 %; M1, M2 +5 %), `yaw` (M0, M1 −5 %; M2, M3 +5 %) |
| latency | `latency_20ms` |
| battery | 0.85 |
| noise | ×2 |

Corners (2), both at latency 20 ms:
- `heavy_sluggish`: mass 1.2, inertia 1.2, k_f 0.85, τ_m 60 ms, battery 0.85, `roll` mismatch, CoM +;
- `light_twitchy`: mass 0.8, inertia 0.8, k_f 1.15, τ_m 15 ms, noise ×2.

**Smoothness rule.**
- It is compared only in regulation (hover_hold). In transients a faster policy legitimately has a larger |Δa|; transients are report-only.
- In the Attitude env the PID gets `Setpoint(quat_wxyz=q_t, pos_w=p0, hold_xy=False)`.
- Only episodes where **neither** controller terminated count.
- Why the 0.001 absolute floor: 0.001 u ≈ 2.7 rad/s of ω command (0.3 % of hover ω), which τ_m attenuates further above 20 Hz. That is thermally irrelevant, and it keeps "≤ PID" from being decided by noise-floor differences. Revisit once PID `hf_rms_u` is measured; if it turns out ≫ 0.001, the floor is moot.

**Release candidate:** one checkpoint must pass G4, G5 and G6 together before G7. Saturation % and the G3-style scenarios (steps, impulses, wind) are reported, not gated.

**Hover target (Phase 7, not a §8 gate):** the max ‖e_p‖ for t ≥ 2 s is ≤ 0.05 m in ≥ 95 % of 200 nominal episodes, mirroring G3's ±5 cm.

Scenario schema additions (`eval/scenarios.py`):
- `tilt_min_deg`/`tilt_max_deg`;
- `tilt_dist: fixed | cap_uniform | so3_uniform`;
- `rate_max_rad_s`;
- `dr: {mode: nominal | fixed | sample, overrides: {...}, scale: 1.0}`;
- `latency_s` (fixed or a range).

The env receives them through `reset(options=...)` as plain data: `qpos0`, `qvel0`, `a0`, `target_schedule: [(t_s, quat_wxyz)]`, `pos_target`, `phys_params: PhysParams`, `disturbance: DisturbanceConfig`, `noise: ImuNoiseConfig`. `envs` cannot import `eval`, so these keys must not be eval types.

The baseline cache key adds the env-config hash, since the envelope changes the baseline behaviour.

### 10. Seeding, determinism and the required tests

**RNG streams** (architecture §6, unchanged): `SeedSequence(episode_seed).spawn(4)` gives:

| Stream | Draws, in order |
|---|---|
| `rng_init` | the lower-level mix draw; z0; target (tilted?, ψ, tilt); δ and its axis; ω0; v0; a0; the target step (flag, t_s, new target); hover target/offset |
| `rng_dr` | one fixed-length block (§7) |
| `rng_noise` | `ImuNoiseModel.reset`, then one `apply` per step (CTBR: per substep) |
| `rng_dist` | impulse/wind magnitudes, then `DisturbanceModel.reset` |

Training ranks are seeded `seed + rank` via `PPO(seed=...)` (SB3 calls `env.seed`). `make_env` must not reseed.

**Determinism requirements:**
- **D1.** The same configs, model hash, `episode_seed` and action sequence give **bitwise-identical** obs, rewards, flags, `reward_terms` and `true` on the same machine and library versions.
- **D2.** Changing d, the noise config or the disturbances never changes `rng_init` draws. Changing d never changes which U index a parameter uses.
- **D3.** `reset(seed=s)` twice gives the same first obs.
- **D4.** PPO learning curves need **not** be reproducible (torch).

**Required tests** (fast unless marked):

| # | Test |
|---|---|
| T1 | `gymnasium.utils.env_checker.check_env` and SB3 `check_env` pass for both env ids |
| T2 | `ObsLayout`: contiguous slices covering D, unique names, units/frames set, matches `observation_space`; `to_meta()` round trip |
| T3 | Obs content with noise off: at q = q_t → `r_err_6d = [1,0,0,0,1,0]`, `gravity_b = (0,0,−1)`; a known 90° roll gives the expected columns; **heading invariance**: rotating q, q_t (and p, p_t) by a world yaw leaves the obs unchanged; Hover `pos_err_h` for ψ_t = 90°, p − p_t = (1,0,0) gives (0, −1, 0) |
| T4 | Reward: `sum(reward_terms) == r` (1e-9); each term within its table range over 10k random steps; `r ≥ 0` on non-terminal steps; perfect state gives r = 1.5 (Attitude); `crash` only on terminated steps; `att` is monotone in e |
| T5 | Command map: a = 0 → u = u_h exactly; a = ±1 → u_h ± Δu; `command_to_action ∘ action_to_command` = id on [−1, 1]; the baseline `thrusts_to_action` stays in [−1, 1]; the Mixer limits equal `cmap.thrust_limits` |
| T6 (physics) | Zero action, nominal, noise off, level start: after 0.5 s \|v_z\| < 0.05 m/s and tilt < 2° (the 0.06 mm nominal CoM offset gives ≈ 1° by then, **estimate**); a + δ on M0, M3 gives ω_x < 0 (conventions sign table, through the env) |
| T7 | Termination: drop with a = −1 → `ground_contact`; qvel 30 rad/s → `rate_limit`; out-of-bounds; injected NaN → `crash` with a finite obs; truncation exactly at step 500/800 with `terminated = False` |
| T8 | Determinism D1–D3; distinct seeds differ |
| T9 | DR: d = 0 → `PhysParams` == nominal; 10k draws at d = 1 inside the §7 ranges, triangle OK, edges hit at ≈ p_extreme; apply A → B → A leaves an identical model (no drift) |
| T10 | Curriculum: `env_method("set_curriculum_level")` bounds the observed δ and ω0 by the level; the promotion logic on synthetic eval streams (min_steps, 2 consecutive) |
| T11 | `info_level: minimal` keys (+ `episode_success`); `privileged_obs` only when enabled with D = 46; the harness never reads `true`/`params`/`privileged_obs` (extends the existing test) |
| T12 | Frozen normalization: after 2 rollouts `obs_rms` is unchanged; `vecnormalize.pkl` reloads to `FixedRunningMeanStd` |
| T13 | Out-of-range actions are clipped, and the clipped value appears in the history, reward and trajectory |
| T14 | Metrics: `hf_rms` = 0 for constant and pure < 20 Hz sinusoids, ≈ A/√2 for a 30 Hz sinusoid of amplitude A; `success_recovery` edge cases |
| T15 (integration) | The smoke run (§8); `run_suite` on the smoke checkpoint and on PID through the env |

## Consequences

- The contract gains two required inputs beyond the IMU: an attitude estimate (already implied) and a **vertical-velocity estimate** (`vel_z_w`). An FC without baro/EKF cannot run the Attitude policy as is.
- Fixed obs scales keep the frozen normalization identical across stages. A badly chosen scale shows up as `obs/clip_frac`, not as silent drift.
- The ±Δu envelope caps T/W at ≈ 3.4 nominal (estimate). This is enough for G5 by the torque estimates, but it is a hard limit; raising it means amending this ADR and the contract.
- The baseline comparison rows change, because baselines now fly inside the same envelope and env. The G3 records stay as they are.
- A memoryless policy carries steady-state attitude error under CoM offsets and motor mismatch (≈ 1° at 3 mm CoM, estimate). G6's 5° threshold tolerates it; G4 is nominal.
- Deterministic mini-evals cost ≈ 4 % of training time (estimate) but make promotion independent of exploration noise.
- G4–G6 are now exact and computable. They differ from the BUILD_PROMPT text in these interpretations:
  - "< 1.5 s / < 2.5 s" are per-episode success conditions;
  - G6 adds a per-config floor of 0.80;
  - smoothness is compared in regulation, with a 0.001 u floor.

## Open questions

| # | Question | Default until resolved |
|---|---|---|
| O1 | Attitude-estimate **bias** (tilt bias of a real EKF, 0.5–1°) is not in `ImuNoiseModel`; it needs a sim change (`quat_bias_std_rad`) | None. Recommended for Phase 6 |
| O2 | A leaky integral of the body-frame error as an extra obs block, if G6 fails on CoM/mismatch configs from steady-state error | Not included |
| O3 | Asymmetric critic (§2 trigger) | Not used |
| O4 | 500 Hz physics switch (§1 criteria) | 1 kHz |
| O5 | 200 Hz policy fallback | 100 Hz |
| O6 | Estimator lag on `vel_z_w` and the attitude estimate is not modelled | Covered only by the action delay |
| O7 | Hover steady-state altitude offset under mass/k_f DR (no integrator) | Measure in Phase 7; O2-style integrator if > 5 cm |
| O8 | Propulsion constants are estimates; the envelope numbers scale with them | Recompute after thrust-stand data |
| O9 | The reward weights marked † | Tune only through training-analyst evidence; record changes in `docs/PROGRESS.md` |
| O10 | Seeding Hover from the Attitude actor (first-layer widening) | Train Hover from scratch |
