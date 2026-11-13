# ADR-0005: Architecture, layering and artifact layout

- **Status:** Accepted (2026-11-13)
- **Details:** [`docs/architecture.md`](../architecture.md). Conventions: [`docs/conventions.md`](../conventions.md).

## Context
Dronez chains build123d CAD → mass properties → a MuJoCo model → Gymnasium envs → SB3 PPO → ONNX. Several subagents own the modules (`cad`, `sim`, `control`, `envs`/`rl`, `eval`, `deploy`), so the boundaries must be explicit, or dependencies tangle and ownership blurs.

Training runs on CPU only (WSL2, 4 vCPU now, possibly 10 later), with `SubprocVecEnv` workers. Those workers must start fast and stay small: no CAD kernel and no torch in them. Generated artifacts must never be hand-edited or go stale silently. Baselines and learned policies have to be compared on identical terms, and the exported policy has to match training exactly. These decisions are expensive to reverse once code exists, so they are fixed now, before Phase 1.

## Decision
1. **Layering.**
   - `utils ← cad ← sim ← control ← envs ← rl ← eval ← deploy`. A module imports only from its left. The entry layer (`dronez/cli.py`, `tools/mcp_server/`, `scripts/`) may import anything, and nothing in `src/dronez` imports it.
   - `control` sits below `envs` because the CTBR mode runs the rate loop and mixer inside the env.
   - Heavy dependencies are confined: build123d only in `cad` (not in `cad.schema`/`cad.artifacts`); torch and SB3 only in `rl`, `eval` (lazily) and `deploy`; onnx and onnxruntime only in `deploy`.
   - An `ast` import-scan test enforces all of this.
2. **Generated artifacts and hashing.**
   - CAD outputs go to `$DRONEZ_DATA_DIR/models/<name>-<model_hash>/`: `manifest.json`, `drone_config.resolved.yaml`, `mass_props.json`, `meshes/*.stl` in metres, `assembly.step`, `render_{top,side,iso}.png`, and an inspection-only MJCF.
   - `model_hash = stable_hash({drone: DroneConfig, cad_version: CAD_VERSION})`.
   - The build is atomic (tmp directory + rename, manifest written last). A YAML change or a `CAD_VERSION` bump yields a new directory. Existing directories are never mutated.
3. **Mass properties.**
   - `mass_props.json` (`schema_version` 1, SI) gives mass, CoM in the CAD frame C, and the full inertia tensor about the CoM in body FLU axes. It also records per-part provenance and any applied `measured_overrides`.
   - Body frame B = C translated to the nominal CoM.
4. **MjSpec-only model construction.**
   - The MjModel is always built in-process through `MjSpec` from the cached artifacts plus `SimConfig`. It is not cached and never loaded from hand-written XML.
   - The free-joint body carries the explicit inertial. Visual meshes have no collision or mass; collision uses primitives.
5. **Motor model in the env layer.**
   - Normalized command → ω_cmd → exact first-order lag (τ_m) → saturation → `T = k_f ω²`, in numpy, stepped at the physics rate inside the decimation loop. No MuJoCo actuator dynamics and no `sensor_noise`.
   - Thrust is applied through site actuators at each motor with `gear = [0, 0, 1, 0, 0, −s_i·k_m/k_f]` and `ctrl = T`.
   - `s_i` comes from the YAML motor list (PX4 Quad-X order M0 FR CCW, M1 RL CCW, M2 FL CW, M3 RR CW).
   - Domain randomization of k_m and k_f rewrites the gear entry per env.
6. **Deterministic plant, randomness in the env.**
   - `sim.DroneSim` (MjModel + MjData + motor model + action delay) has no RNG.
   - The env owns four seeded streams (init, DR, noise, disturbances) spawned from a per-episode seed, and mutates its own per-process MjModel (`body_mass`, `body_inertia`, `body_ipos`, `actuator_gear`) from a nominal snapshot at reset.
   - There is one MjModel per env process. `SubprocVecEnv` comes first; a batched single-process VecEnv is added only if the measured SPS is insufficient.
7. **One controller interface.** `Controller.act(obs, sensed) -> action` serves PID, geometric, SB3 and ONNX controllers alike. Policies read only the raw `obs` and normalize it themselves; baselines read only `sensed` and use nominal parameters. Eval treats all four identically, with paired seeds.
8. **Run directory.**
   - `$DRONEZ_DATA_DIR/runs/<run_id>/` holds `config.yaml` (fully resolved, with hashes, seed and git SHA), `status.json` (written atomically, schema in architecture §6), `train.log`, `tb/`, and `checkpoints/step_<N>/` and `checkpoints/best/`.
   - Each checkpoint directory holds `model.zip`, `vecnormalize.pkl` and `ckpt.json`, always saved together.
9. **Export boundary.**
   - The ONNX graph takes raw obs in → normalization → MLP → action in [−1, 1] (fed back as history) and motor commands in [0, 1] in motor index order.
   - `policy_meta.json` records the layout, normalization, rates, motor mapping, hashes and versions.
10. **Rates.** Physics 1 kHz, policy 100 Hz and decimation 10 are tentative. The obs, action, reward and rate spec is deferred to the Phase 4 ADR.

## Consequences
- Env workers import only numpy, mujoco and pydantic, so startup is fast and memory per process stays low. If the cache is missing, the workers fail loudly instead of running CAD N times.
- Cache staleness depends on bumping `CAD_VERSION`. A forgotten bump reuses old meshes and mass properties. A source-fingerprint test is recommended.
- Rebuilding the MjSpec in every process costs a few milliseconds and removes a whole class of stale-MJCF bugs.
- Mutating MjModel fields at runtime bypasses the compiler checks. The randomizer must itself enforce positive mass and the inertia triangle inequalities, and may need `mj_setConst` (verify for MuJoCo 4.x).
- The open-loop motor model per policy step allows a later `mujoco.rollout`-based batched path. Enabling state-dependent rotor drag or ground effect removes that option.
- Baselines cannot cheat with privileged state, so comparison rows are fair. A unit test guards the harness.
- MuJoCo 4.0.1 API details (MjSpec names, actuator defaults, `xfrc_applied` frame, accelerometer semantics) must be verified with Context7 before Phase 2.
