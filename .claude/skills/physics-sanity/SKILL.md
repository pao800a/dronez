---
name: physics-sanity
description: Run and interpret the Dronez physics sanity suite (hover thrust = weight, free fall = g, single-motor roll/pitch signs in FLU, yaw-torque sign, inertia validity, motor-lag step response, determinism). Use after any change to the sim, motor model, mass properties, mixer or frames code, when a physics test fails, or when a policy behaves as if the physics is wrong.
---
# Physics sanity suite

Needs the built model (`uv run dronez-build-model --drone default_5in`); without it the suite skips with that
message. CI runs it on ubuntu and windows right after "CAD tests".

```bash
uv run pytest -m physics -q                      # whole suite (tests/physics/)
uv run pytest -m physics -q -k "sign or wrench"  # roll/pitch/yaw signs + steady wrench
uv run pytest -m physics -q -k hover             # hover (default rate and 500 Hz)
uv run pytest -m physics -q -k "free_fall or ground"
uv run pytest -m physics -q -k motor             # lag step response, saturation
uv run pytest -m physics -q --lf                 # re-run last failures
```

All `dronez.sim` imports and constructor assumptions are isolated in `tests/physics/conftest.py`
(`_make_drone_sim`, `_nominal_params`); a renamed API is a fix there. Tests use the `rig` / `rig_rate`
fixtures and `rig.new_plant()`.

## What each test proves, and what a failure usually means

| Test (file) | Pass criterion | Typical cause of failure |
|---|---|---|
| `test_hover_holds_position[default-rate\|500hz]` (test_hover) | hover command holds position within 2 cm over 2 s, tilt < 0.5 deg | wrong mass in model (CAD vs override), `k_f` units (rpm vs rad/s), thrust applied on wrong axis, `hover_command` inverse wrong |
| `test_hover_command_gives_weight_thrust` | `k_f ω² = mg/4` per motor (rel 1e-6) | `command_to_omega` / `hover_command` mismatch |
| `test_free_fall_is_gravity` | a = −g (rel 1e-3), accelerometer ≈ 0 (< 1e-3 m/s²) | gravity/timestep options, ω_min idle thrust, damping on the free joint, accelerometer not specific force |
| `test_at_rest_on_ground_accelerometer_reads_plus_g` | settled on ground: accel ≈ (0, 0, g) within 3 % | missing ground plane, contact params, collision geoms |
| `test_differential_sign[right-up-rolls-left]` | M0+M3 up → `ω_x < 0`, magnitude within 8 % of the analytic value | motor positions mirrored (y sign), FRD/FLU confusion |
| `test_differential_sign[front-up-nose-up]` | M0+M2 up → `ω_y < 0`, within 8 % | x sign of positions, pitch-axis sign in the mixer |
| `test_differential_sign[cw-up-yaws-left]` | M2+M3 (CW) up → `ω_z > 0`, within 40 % | gear sign `−s·k_m/k_f` inverted, spin enum mapping |
| `test_steady_wrench_matches_mixer_convention` | unequal constant ω: body accel = I⁻¹τ (2 %), vertical accel = ΣT/m − g | yaw torque magnitude, inertia frame, thrust axis |
| `test_model_matches_mass_properties` | mass, principal moments, hub positions, spin signs equal CAD/YAML | parallel-axis error, mm→m (1e-6), body frame not at CoM |
| `test_inertia_triangle_inequality_under_scaling` | SPD, triangle inequalities kept (or rejected) under scaling | DR scaling of principal moments without the guard |
| `test_motor_step_response_at_tau[dt]` / `_in_plant` | 63.2 % (±5 %) of a step at t = τ_m, exact (1e-6) for any dt | discretization (use `1−exp(−dt/τ)`), lag stepped at the policy rate instead of the physics rate |
| `test_motor_saturates` | output clipped to [k_f ω_min², k_f ω_max²] | clip before lag, or missing |
| `test_two_rollouts_bit_identical` | identical trajectories (same sim re-reset, fresh sim) | unseeded RNG, global `np.random`, wall-clock in the loop, `mj_resetData` not full |
| `test_set_params_nominal_is_idempotent` | nominal write restores model arrays (1e-12), 20 perturb/nominal cycles do not drift | `set_params` writing relative to current values instead of the nominal snapshot |

## When something fails
1. Read the assertion values (they print expected vs actual and units).
2. Check the sign conventions in `docs/conventions.md` and the frames-and-units skill.
3. Fix the **source** (YAML / builder / motor model), never the test tolerance, unless an ADR justifies it.
4. After a sim change passes, ask `physics-auditor` for review before merging (Opus budget permitting).
