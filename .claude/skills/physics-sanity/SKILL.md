---
name: physics-sanity
description: Run and interpret the Dronez physics sanity suite (hover thrust = weight, free fall = g, single-motor roll/pitch signs in FLU, yaw-torque sign, inertia validity, motor-lag step response, determinism). Use after any change to the sim, motor model, mass properties, mixer or frames code, when a physics test fails, or when a policy behaves as if the physics is wrong.
---
# Physics sanity suite

```bash
uv run pytest -m physics -q                 # whole suite
uv run pytest -m physics -q -k yaw          # one check
uv run pytest -m physics -q --lf            # re-run last failures
```

## What each test proves, and what a failure usually means

| Test | Pass criterion | Typical cause of failure |
|---|---|---|
| hover hold | analytic hover throttle holds altitude ±2 cm over 2 s | wrong mass in model (CAD vs override), `k_f` units (rpm vs rad/s), thrust applied on wrong axis, motor lag not settled before t0 |
| free fall | z(t) matches −½gt² (no thrust) | gravity/timestep options, ground contact at t0, damping on the free joint |
| roll sign | right motors (M0, M3) up → `ω_x < 0`, roll to the left side | motor positions mirrored (y sign), FRD/FLU confusion |
| pitch sign | front motors (M0, M2) up → `ω_y < 0`, nose up | x sign of positions, pitch-axis sign in the mixer |
| yaw sign | CW pair (M2, M3) faster → `ω_z > 0` (CCW from above) | gear sign `−s·k_m/k_f` inverted, spin enum mapping |
| inertia | SPD, triangle inequalities, matches `mass_props.json` | parallel-axis error, mm→m conversion squared wrongly (1e-6 factor) |
| motor lag | 63.2 % of a step at t = τ_m (±5 %) | discretization (use `1−exp(−dt/τ)`), lag stepped at the policy rate instead of the physics rate |
| determinism | identical trajectories for the same seed | unseeded RNG, global `np.random`, wall-clock in the loop |

## When something fails
1. Read the assertion values (they print expected vs actual and units).
2. Check the sign conventions in `docs/conventions.md` and the frames-and-units skill.
3. Fix the **source** (YAML / builder / motor model), never the test tolerance, unless an ADR justifies it.
4. After a sim change passes, ask `physics-auditor` for review before merging (Opus budget permitting).
