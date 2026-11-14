# Dronez architecture

Phase 0 baseline (2026-11-13). The structural decisions are recorded in [ADR-0005](adr/0005-architecture-and-layering.md). [`conventions.md`](conventions.md) wins on any conflict about frames, units, signs or motor order. The observation layout, action scaling, reward, termination, final rates, curriculum and DR ranges are **not** specified here. They are decided by the Phase 4 RL-spec ADR. Anything marked **(verify)** is an API detail to check with Context7 at implementation time.

## 1. Layering

```
utils ← cad ← sim ← control ← envs ← rl ← eval ← deploy          ("A ← B": B may import A)
entry layer (imports anything, imported by nothing in src/): dronez/cli.py, tools/mcp_server/, scripts/
```

1. A module imports only from modules to its left (and from itself). Nothing imports to its right.
2. `control` sits below `envs` because the CTBR action mode reuses the rate loop and mixer inside the env.
3. Heavy dependencies stay confined:
   - `build123d` is used only in `cad` (never in `cad.schema` or `cad.artifacts`).
   - `mujoco` is used from `sim` upward.
   - `torch` and SB3 are used only in `rl`, `eval` (lazily, inside `PolicyController`) and `deploy`.
   - `onnx` and `onnxruntime` are used only in `deploy`.
   - `matplotlib` and `imageio` are used only in `cad.render` and `eval`.
   - Env worker processes must never import torch or build123d.
4. `dronez/cad/__init__.py` imports nothing heavy. `sim` reads CAD outputs through `cad.schema` and `cad.artifacts` only.
5. The rule is enforced by `tests/unit/test_layering.py`, an `ast` import scan of `src/dronez` that also checks the heavy-dependency rules (owner: test-engineer).

## 2. Module map

| Module | Owner | Responsibility |
|---|---|---|
| `utils` | orchestrator / test-engineer (rotation changes reviewed by physics-auditor) | rotations (the only quaternion math), paths, YAML loading, hashing, device selection. Already exists |
| `cad` | cad-engineer | `DroneConfig`, parts, assembly, materials, mass properties, mesh/STEP/render export, artifact cache |
| `sim` | sim-engineer | `SimConfig`, the MjSpec builder, motor model, delay line, sensor and noise models, disturbance generators, the `DroneSim` plant |
| `control` | controls-engineer | the `Controller` protocol, mixer, rate loop, cascaded PID, geometric SO(3) controller |
| `envs` | rl-engineer | Gymnasium envs, observation layout, reward hooks, domain randomization, curriculum, env factories |
| `rl` | rl-engineer | `TrainConfig`, PPO training, callbacks, run directory, checkpoint I/O |
| `eval` | rl-engineer (controls-engineer seeds it in Phase 3) | scenarios, metrics, gates, reports, video, `PolicyController` |
| `deploy` | deploy-engineer | ONNX export, `OnnxController`, parity and latency checks, `policy_meta.json`, C export |
| `tools/mcp_server` | rl-engineer (tests: test-engineer) | the `dronez-sim` MCP server (Phase 4) |

Public API sketches (names are binding, signatures indicative):

```python
# cad/schema.py (pydantic only)
CAD_VERSION: str                                   # bump whenever CAD code changes any artifact
class DroneConfig(BaseModel): ...                  # §3
class MassProperties(BaseModel): ...               # §4, the contents of mass_props.json
def load_drone_config(name_or_path: str | Path) -> DroneConfig   # utils.config.load_model(..., "drone")
def model_hash(cfg: DroneConfig) -> str            # stable_hash({"drone": cfg, "cad_version": CAD_VERSION})
# cad/artifacts.py (no build123d)
@dataclass(frozen=True)
class ModelArtifacts: root: Path; mass_props: MassProperties; meshes: dict[str, Path]; manifest: dict
def load_artifacts(cfg: DroneConfig) -> ModelArtifacts   # raises MissingModelError("run dronez-build-model")
# cad/assembly.py, mass_props.py, export.py, render.py (build123d)
def build_assembly(cfg: DroneConfig) -> Assembly         # parts: dict[str, Part(solid, material, count)]
def compute_mass_properties(asm: Assembly, cfg: DroneConfig) -> MassProperties  # overrides applied last
def build_artifacts(cfg: DroneConfig, force: bool = False) -> ModelArtifacts    # cache-aware, atomic

# sim/
class SimConfig(BaseModel): physics_dt_s; decimation; integrator; ground_plane; prop_disk_collision;
                            rotor_drag: bool = False; ground_effect: bool = False
def build_spec(cfg: DroneConfig, art: ModelArtifacts, sim: SimConfig) -> mujoco.MjSpec
def build_model(cfg: DroneConfig, sim: SimConfig) -> tuple[mujoco.MjModel, ModelIndex]
@dataclass(frozen=True)
class ModelIndex: body_id; imu_site_id; motor_site_ids; actuator_ids; sensor_slices: dict[str, slice];
                  spin: NDArray (4,); motor_pos_b: NDArray (4, 3); nominal: NominalInertial  # mass, inertia, iquat, ipos
@dataclass
class MotorParams: k_f; k_m; tau_m_s; omega_min; omega_max   # arrays (..., n_motors), broadcastable
class MotorModel:   # pure numpy, vectorized over leading batch dims, no RNG
    def reset(self, omega0: NDArray) -> None
    def command_to_omega(self, u: NDArray) -> NDArray        # u ∈ [0, 1] → ω_cmd [rad/s]
    def step(self, omega_cmd: NDArray, dt: float) -> NDArray # exact first-order lag, then saturation; returns T [N]
    def thrust_to_command(self, thrust: NDArray) -> NDArray  # inverse map for mixers / hover
def hover_command(params: MotorParams, mass_kg: float) -> NDArray
class ActionDelay: (n_steps: int) push(u) -> u_delayed     # ring buffer in physics steps
@dataclass
class Sensed: quat_wxyz; gyro_b; accel_b                   # what a flight controller would have
class ImuNoiseModel: reset(rng); apply(true: TrueState, dt: float, rng) -> Sensed
class DisturbanceModel: reset(rng, cfg); wrench(t: float, true: TrueState) -> tuple[force_w, torque_w]
@dataclass
class PhysParams: mass_kg; com_offset_b_m; inertia_diag_kg_m2; motor: MotorParams; latency_steps; battery_scale
class DroneSim:     # the deterministic plant: one MjModel + one MjData + MotorModel + ActionDelay
    def reset(self, qpos, qvel, omega0) -> None
    def set_params(self, p: PhysParams) -> None              # writes model fields from the nominal snapshot, §5.3
    def set_wrench(self, force_w, torque_w) -> None
    def substep(self, u: NDArray) -> None                    # one physics step
    def true_state(self) -> TrueState                        # pos_w, quat_wxyz, linvel_w, angvel_b, omega, thrust

# control/
class Controller(Protocol):                                  # §7
    action_mode: Literal["motors", "ctbr"]
    def reset(self, seed: int | None = None) -> None
    def act(self, obs: NDArray[np.float32], sensed: Sensed) -> NDArray[np.float32]
class Mixer: (cfg: DroneConfig, k_m_over_k_f) A: (4, 4); allocate(T_total, tau_b) -> T (4,)
class RateController; class CascadedPID(Controller); class GeometricSO3(Controller)

# envs/
class DronezBaseEnv(gym.Env): (env_cfg: EnvConfig, sim_cfg: SimConfig, drone: str, render_mode=None)
    def set_curriculum_level(self, level: float) -> None     # called through VecEnv.env_method
    # subclass hooks: _sample_task(rng), _task_obs(), _reward() -> (r, terms), _terminated()
class AttitudeEnv(DronezBaseEnv)   # "DronezAttitude-v0";  class HoverEnv  # "DronezHover-v0"
class ObsLayout: names, slices, units, frames            # single source for policy_meta.json
class DomainRandomizer: sample(rng, level) -> PhysParams; apply(sim: DroneSim, p: PhysParams)
class Curriculum: level; update(success_rate: float) -> float
def make_env(env_id: str, cfg: EnvConfig, rank: int, seed: int) -> Callable[[], gym.Env]

# rl/
class TrainConfig(BaseModel): run (stage, seed, total_timesteps, n_envs, device, vec_env), env, sim,
                              drone, ppo, policy, normalize, checkpoint_every, eval_every
def train(cfg: TrainConfig, run_id: str | None = None, resume: bool = False) -> Path
class RunDir: paths; write_status(**fields)                  # atomic (tmp file + os.replace)
def save_checkpoint(model, vecnorm, path) -> None; def load_checkpoint(path) -> tuple[PPO, VecNormalize]

# eval/
class EvalSuite(BaseModel); class Scenario(BaseModel)
class PolicyController(Controller): from_checkpoint(run_id, ckpt)   # applies frozen VecNormalize itself
def run_suite(make_controller: Callable[[], Controller], suite: EvalSuite, out: Path, n_workers: int) -> SuiteResult
def episode_metrics(traj: Trajectory, dt: float) -> dict[str, float]
def check_gate(metrics: dict, gate: str) -> GateResult; def write_report(result, baselines, out) -> None

# deploy/
def export_onnx(run_id: str, ckpt: str, version: str) -> Path
class OnnxController(Controller); def parity(...) -> ParityReport; def latency(...) -> LatencyReport
```

The CLI (`dronez/cli.py`) composes these. `dronez-build-model` calls `cad.build_artifacts`, then `sim.build_model`, and prints hover throttle and T/W. `dronez-eval --onnx <path>` wraps `deploy.OnnxController` and passes it to `eval.run_suite`.

## 3. DroneConfig (`configs/drone/<name>.yaml`)

The exact geometry fields are defined by cad-engineer in Phase 1. Every length is in m, given in the **CAD frame C**: FLU axes, with the xy origin at the centroid of the motor hubs and z = 0 at the arm plane (the top face of the bottom plate). Constants come from datasheets or thrust-stand data, cited in the `source` fields. No numbers are invented.

| Group | Fields (units) |
|---|---|
| identity | `name`, `layout: quad_x` (the validator checks 4 motors in PX4 Quad-X order and their spins) |
| `geometry` | frame plates, arms, standoffs, stack (`fc_position_m`, `imu_position_m`), camera, prop clearance (all m) |
| `materials` | `{key: {density_kg_m3}}` for carbon fibre, aluminium, copper/steel, LiPo, PLA, TPU. `parts_material: {part: key}` must cover every part |
| `motors` | list in index order: `{name, position_m: [x, y, z] (C, hub/prop plane), spin: ccw \| cw}`. M0 FR ccw, M1 RL ccw, M2 FL cw, M3 RR cw. The validator checks sign quadrants and the order. `s_i`, the mixer and the gear are derived from this list |
| `motor` | stator and bell dimensions (m) and materials, for CAD mass |
| `prop` | `diameter_m`, `pitch_m`, `blade_count`, material |
| `propulsion` | `k_f` (N/(rad/s)²), `k_m` (N·m/(rad/s)²), `omega_min_rad_s`, `omega_max_rad_s` (or the `_rpm` twins), `tau_m_s`, optional `rotor_inertia_kg_m2`, `source: str` (required), optional per-motor overrides |
| `battery` | `cells`, `cell_voltage_nominal_v`, `dimensions_m`, `position_m`, material (`lipo`) |
| `target_auw_kg` | the G1 check: CAD mass within ±10 % |
| `measured_overrides` | optional `mass_kg`, `com_c_m`, `inertia_com_kg_m2` (3×3), `source: str`. Each set field replaces the CAD value. If only the mass is overridden, the inertia is scaled by `m_meas/m_cad` |
| `cad` | export tuning (mesh tolerance, decimation target). Part of the hash |

## 4. Model build pipeline and artifacts

```
configs/drone/<name>.yaml ─► DroneConfig ─► build123d parts + assembly (part → material)
   ─► MassProperties ─► mass_props.json ─┐
   ─► meshes/*.stl, assembly.step, renders ─┴─► models/<name>-<hash>/ (cache)
configs/sim/<name>.yaml ─► SimConfig ─┐
   cache + DroneConfig ───────────────┴─► sim.build_spec (MjSpec) ─► spec.compile() ─► MjModel + ModelIndex
```

- **Hash and cache:** `model_hash = stable_hash({"drone": DroneConfig.model_dump(mode="json"), "cad_version": CAD_VERSION})` (10 hex characters).
  - A cache directory is valid only if its `manifest.json` exists and its hash matches. The manifest is written last.
  - The build runs in `models/.tmp-<hash>-<pid>/` and is renamed into place, so a crash never leaves a half-built cache.
  - `--force` rebuilds. Any change to the YAML or to `CAD_VERSION` produces a new directory. Old directories are never mutated.
- **Directory `$DRONEZ_DATA_DIR/models/<name>-<model_hash>/`:**
  - `manifest.json`: hash inputs, `cad_version`, git SHA and dirty flag, library versions, UTC timestamp, file list with sha256;
  - `drone_config.resolved.yaml`;
  - `mass_props.json`;
  - `meshes/<part>.stl`: visual meshes, decimated, already in **metres**;
  - `assembly.step` (STEP carries its own units);
  - `render_top.png`, `render_side.png`, `render_iso.png`;
  - `model_<sim_hash>.xml`: an inspection-only MJCF from `spec.to_xml()` (verify). It is never loaded as a source.
- **Units at the CAD boundary:** build123d works in mm. Convert once in `cad.mass_props`: length ×1e-3, volume ×1e-9, unit-density inertia (mm⁵) × ρ × 1e-15 → kg·m². The build123d mass-property API names are (verify).
- **`mass_props.json`** (`MassProperties`, `schema_version: 1`):

```json
{"schema_version": 1, "drone": "<name>", "model_hash": "<hex>", "cad_version": "<str>",
 "units": {"length": "m", "mass": "kg", "inertia": "kg*m^2"},
 "frame": "axes: body FLU; com_c_m in CAD frame C; inertia about the CoM",
 "mass_kg": 0.0, "com_c_m": [0, 0, 0],
 "inertia_com_kg_m2": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
 "principal_moments_kg_m2": [0, 0, 0], "principal_axes_quat_wxyz": [1, 0, 0, 0],
 "parts": [{"name": "", "material": "", "density_kg_m3": 0, "volume_m3": 0, "mass_kg": 0, "count": 1, "com_c_m": [0, 0, 0]}],
 "motor_hubs_b_m": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
 "target_auw_kg": 0.0, "auw_error_frac": 0.0, "overrides_applied": [], "source": "cad | cad+measured"}
```
  (The zeros are type placeholders, not values.) The inertia is the true tensor: off-diagonals are `−∫xy dm`, the form MjSpec `fullinertia` expects (verify the element order).
- **MjSpec builder rules.** Models are only ever built through MjSpec; no hand-written XML.
  - **Body frame B = C translated by `com_c_m`.** The free-joint body origin is the nominal CoM, `ipos = 0`, and every geometry position is `p_B = p_C − com_c_m`.
  - The inertial is explicit (`mass`, `fullinertia`). Inertia from geoms is disabled (verify the 4.x flag).
  - Visual mesh geoms have `contype = conaffinity = 0` and contribute no mass. Collision uses primitives (boxes and cylinders; prop disks behind a toggle) plus a ground plane.
  - The motor sites `m0…m3` sit at `motor_hubs_b_m`, parallel to B. There is one IMU site.
  - **The actuators are site actuators** on `m_i`:
    - `gear = [0, 0, 1, 0, 0, −s_i·k_m/k_f]`, `ctrl = thrust [N]`;
    - gain 1, no bias, no MuJoCo actuator dynamics (verify the defaults);
    - `ctrllimited = false`, because clamping happens in the motor model and DR can raise the thrust above nominal.
  - Sensors on the IMU site: `gyro`, `accelerometer` (verify that it reports specific force), `framequat`, plus body `framepos` and `framelinvel` for ground truth. MuJoCo `sensor_noise` is not used.

## 5. Sim runtime

### 5.1 Process model
- Each env process (each `SubprocVecEnv` worker and each eval worker) owns **exactly one MjModel and one MjData**. It builds them at start-up from the cache with `sim.build_model`, which takes milliseconds and needs no CAD. If the cache is missing, the worker raises an error: N workers must never run CAD concurrently.
- Workers set `OMP_NUM_THREADS = MKL_NUM_THREADS = 1`. Start method: `forkserver` on Linux, `spawn` on Windows.

### 5.2 Step loop (`DronezBaseEnv.step`, motors mode)
```
a = clip(action, -1, 1); u = command_map(a)            # u = clip(u_h + u_h·a, 0, 1), linear in ω (ADR-0006 §3)
for k in range(decimation):                            # physics rate
    sim.set_wrench(*disturbance.wrench(t, true))       # world frame, at the body CoM (xfrc_applied, verify)
    sim.substep(u)                                     # delay.push → command_to_omega → motor.step
                                                       # → data.ctrl = T → mj_step(model, data)
true = sim.true_state(); sensed = imu_noise.apply(true, dt_policy, rng_noise)   # policy rate
obs = obs_builder(sensed, target, action_history); r, terms = _reward(); term/trunc
```
The motor lag uses the exact discretization `ω += (ω_cmd − ω)(1 − e^{−dt/τ_m})`, then clips to [ω_min, ω_max], and computes `T = k_f ω²`. The gear makes MuJoCo apply `−s_i k_m ω²` consistently. Over a policy step the motor model is **open-loop**: it depends only on commands and its own state, not on the MuJoCo state. This is what makes batching possible later (§5.4).

### 5.3 Where randomness and sim-to-real effects live
The plant (`sim`) is deterministic and RNG-free. The **env layer** owns every RNG stream, samples every parameter and decides when each effect is stepped. The `sim/sensors.py` and `sim/disturbances.py` models are pure numpy and receive the RNG from the env.

| Effect | Sampled (stream) | Stepped at | Applied by |
|---|---|---|---|
| mass, inertia, CoM offset | reset (`rng_dr`) | – | `model.body_mass[b]`, `body_inertia[b]`, `body_ipos[b]` |
| k_f, k_m, τ_m, per-motor mismatch, ω_max (battery) | reset (`rng_dr`) | physics | `MotorModel` params plus `model.actuator_gear[a, 5] = −s_i k_m,i/k_f,i` |
| latency (whole physics steps) | reset (`rng_dr`) | physics | `ActionDelay(n_steps)` |
| gyro/accel/attitude noise, bias, bias random walk | bias at reset, noise each step (`rng_noise`) | policy | `ImuNoiseModel` → `Sensed` |
| impulses, wind | schedule at reset (`rng_dist`) | physics | `data.xfrc_applied[b]` |
| initial state, target | reset (`rng_init`) | – | `qpos`/`qvel`, initial ω |

`DomainRandomizer.apply` always writes **from the nominal snapshot** in `ModelIndex.nominal`, so randomization never drifts across episodes. Inertia rules:
- Inertia scales are applied to the principal moments.
- The result must satisfy the triangle inequalities (`I_a + I_b ≥ I_c`). Resample or clip otherwise: a flat quad sits near `I_zz ≈ I_xx + I_yy`, so independent ±20 % scaling breaks physics.
- Inertia scales with the mass scale times an anisotropic factor.

After the edits, call `mj_setConst` if 4.x needs it for runtime mass changes (verify), then `mj_resetData`, set the state, and run `mj_forward`. Ranges and the widening schedule belong to the Phase 4/5 ADR. `PhysParams` is a plain dataclass so it can also go into `info["params"]`.

### 5.4 Vectorization plan
1. **Now:** `SubprocVecEnv`. Pick `n_envs` from `scripts/benchmark_sps.py` (4 vCPU now, possibly 10 later; start at physical cores − 2, then measure).
2. **If SPS is insufficient:** first profile the env step and keep `info_level: minimal`. Then add `BatchedDronezVecEnv` (an SB3 `VecEnv`, one process) holding N models and data, with motor, noise and observation code batched over `(n_envs, …)` (the `MotorModel` API already broadcasts). Because the motor model is open-loop per policy step, the `(n_envs, decimation, 4)` thrust sequence can be precomputed and stepped with `mujoco.rollout` over multiple models and threads (verify multi-model support and `xfrc_applied` as a control input). Rotor drag or ground effect would make thrust state-dependent and break that shortcut. Those flags then force the per-substep path.

## 6. Env and RL interfaces

The binding obs/action/reward/termination/curriculum/DR/PPO spec is [ADR-0006](adr/0006-rl-spec.md). This section keeps the interface shape.

- **Step contract (Gymnasium 1.x):**
  - `reset(seed=None, options=None) -> (obs, info)` and `step(a) -> (obs, r, terminated, truncated, info)`. `options` carries eval overrides as plain data (ADR-0006 §9: `qpos0, qvel0, a0, target_schedule, pos_target, phys_params, disturbance, noise`);
  - `obs` is a float32 `Box` laid out by `ObsLayout` (Attitude D = 29, Hover D = 34; ADR-0006 §2). It is raw SI, built from `Sensed` only. VecNormalize normalizes it in training **with frozen fixed statistics** (`rl.normalization.FixedRunningMeanStd`, offsets/scales from `ObsLayout`), so `vecnormalize.pkl` still carries μ, σ², ε and clip for §9;
  - `a` is float32 `Box(-1, 1, (4,))` in motor index order, mapped by `control.CommandMap` to `u = clip(u_h + Δu·a, 0, 1)`, linear in ω (ADR-0006 §3). CTBR: `[T_norm, p, q, r]`;
  - `terminated` covers crash (a non-finite state), rate limit, ground contact and out-of-bounds (thresholds in ADR-0006 §5); `truncated` means the time limit (`episode_len_s`: Attitude 5 s, Hover 8 s, enforced internally, no `TimeLimit` wrapper).
- **Registration:** in `dronez/envs/__init__.py`.
- **`info` keys** (`info_level: minimal` in training sends only those marked ᵐ):
  - `sensed`: `{quat_wxyz, gyro_b [rad/s], accel_b [m/s²]}`, the only measurement input baselines may use;
  - `true`: `{pos_w, quat_wxyz, linvel_w, angvel_b, omega_rotor [rad/s], thrust_n}`;
  - `params`: `PhysParams` as a dict (`mass_kg, com_offset_b_m, inertia_diag_kg_m2, k_f, k_m, tau_m_s, omega_max_rad_s, latency_steps, battery_scale`);
  - `disturbance`: `{force_w_n, torque_w_nm, wind_w_m_s}`; `target`: `{quat_wxyz, pos_w}`;
  - ᵐ`privileged_obs`: a float32 vector with a fixed layout (`ObsLayout.privileged`, D = 46) for an optional asymmetric critic. It is emitted only when `EnvConfig.privileged_obs: true` (ADR-0006 §2);
  - ᵐ`reward_terms`: `{name: float}`, the weighted contributions, summing to `r`; ᵐ`metrics`: `{att_err_rad, tilt_rad, saturation_frac}`;
  - ᵐ`termination`: `null | crash | rate_limit | ground_contact | out_of_bounds | time_limit`;
  - ᵐ`episode_seed`, ᵐ`curriculum_level`, and ᵐ`episode_success` (on the last step of an episode).
- **Seeds:**
  - At each reset, `episode_seed = seed if seed is not None else self.np_random.integers(2**63)`.
  - `SeedSequence(episode_seed).spawn(4)` → `rng_init, rng_dr, rng_noise, rng_dist`. Changing the noise config therefore never changes initial states, and any episode can be replayed from its `episode_seed`.
  - Training: `TrainConfig.run.seed` → env rank i gets `seed + i`, plus `set_random_seed(seed)` for torch and numpy.
  - Eval: `episode_seed = SeedSequence([suite_seed, scenario_idx, episode_idx]).generate_state(1)[0]`, so every controller sees identical episodes (paired comparison).
- **Run directory `$DRONEZ_DATA_DIR/runs/<run_id>/`.** `run_id` matches `[a-z0-9-]+` and defaults to `<train_cfg_stem>-<YYYYmmdd-HHMMSS>` in UTC. Contents:
  - `config.yaml`: the resolved TrainConfig with the sim and drone configs inline, plus `config_hash`, `model_hash`, seed, git SHA and library versions;
  - `status.json`;
  - `train.log`: Python logging only, no per-step spam;
  - `tb/`: TensorBoard events. These include `reward/<term>`, `curriculum/level`, `curriculum/dr_scale`, `episode/success_rate`, `episode/termination/<reason>`, `action/saturation_frac`, `action/mean_abs_delta` and `obs/clip_frac` (ADR-0006 §4);
  - `checkpoints/step_<012d>/{model.zip, vecnormalize.pkl, ckpt.json}`: model and normalization stats are **always** written together, into the same directory, via a tmp-dir rename;
  - `checkpoints/best/`: same files.

  `ckpt.json` holds `{timesteps, wall_time_s, curriculum_level, metric, created_at}`. The background stdout goes to `runs/<run_id>.stdout`, beside the directory. Resume loads the latest `step_*` directory with `reset_num_timesteps=False`.
- **`status.json`** (written atomically by `StatusCallback` at most every 30 s):
  `{schema_version, run_id, state: starting|running|finished|failed|stopped, pid, started_at, updated_at (ISO-8601 UTC), timesteps_done, timesteps_total, sps, eta_s, n_envs, device, seed, git_sha, git_dirty, config_hash, model_hash, curriculum_level, latest_checkpoint, best_checkpoint, best_metric: {name, value}, last_scalars: {ep_rew_mean, ep_len_mean, approx_kl, explained_variance}, error: str|null}`.

## 7. Controller interface (baselines and policies treated identically)

`Controller.act(obs, sensed) -> action` returns an action in the env's own action space, for the env's `action_mode`.
- Policies (`PolicyController`, `OnnxController`) read only `obs`, the raw vector, and apply their own frozen normalization.
- Baselines (`CascadedPID`, `GeometricSO3`) read only `sensed` and their own internal state, use **nominal** `DroneConfig` parameters (they do not know the DR draw), and map thrusts to commands with `Mixer` + `MotorModel.thrust_to_command`.
- Neither may read `info["true"]`, `info["params"]` or `info["privileged_obs"]` (enforced by a unit test on the eval harness).
- `reset(seed)` clears integrators and history.

The eval harness calls `obs, info = env.reset(seed=s)`, then loops `a = ctrl.act(obs, info["sensed"])` and `env.step(a)`, recording `info["true"]`, `sensed`, `a` and `reward_terms`.

## 8. Eval and report pipeline

- **Inputs and execution:** `configs/eval/<suite>.yaml` → `EvalSuite` (scenarios: name, env id, n_episodes, init distribution and tilt bin, rate range, disturbance schedule, DR mode `nominal | extremes | sample`, latency override, duration). `run_suite` fans episodes out to `n_workers` processes, one env each.
- **Metrics:** `eval/metrics.py` holds pure functions over trajectories: success, time to recover below 5°, RMS attitude error, overshoot, saturation %, mean |Δa|, and the fraction of action-spectrum energy above 20 Hz (FFT at the policy rate). Values are stored in SI (rad, s) and converted to degrees only in `report.*`.
- **Output, `$DRONEZ_DATA_DIR/eval/<run_id>/<checkpoint>/`:**
  - `suite.resolved.yaml`;
  - `metrics.json` (`{schema_version, controller, run_id, checkpoint, suite_hash, model_hash, git_sha, scenarios: {name: {...}}, gates: {G4: {pass, criteria: [{name, value, threshold, pass}]}}}`);
  - `report.md`, `report.html`;
  - `plots/*.png`, `videos/*.mp4` (`mujoco.Renderer`, `MUJOCO_GL=egl` on Linux);
  - `trajectories/*.npz` (a subset);
  - `obs_recorded.npy` (raw observations, the deploy parity set).
- **Baselines:** rows are cached in `eval/baseline-<pid|geometric>/<suite_hash>-<model_hash>/` and joined into every report.

## 9. Deploy pipeline

- **ONNX graph** (one exported `nn.Module`, batch dimension dynamic):
  ```
  obs_raw [B, D] f32 → (x − μ)/√(σ² + ε) → clip(±clip_obs) → actor MLP (deterministic path of the SB3 policy)
     → action [B, 4] ∈ [−1, 1]     (output 1: fed back into the last-K-action history)
     → motor_cmd [B, 4] ∈ [0, 1]   (output 2: command_map, motor index order M0..M3 = PX4 motors 1..4)
  ```
- **Normalization constants:** μ, σ², ε and `clip_obs` are read from the checkpoint's `vecnormalize.pkl`, never assumed.
- **Opset and exporter:** the opset and the torch 2.15 exporter path (`dynamo`) are (verify), and the result is recorded in the metadata.
- **Checks:** parity ≤ 1e-5 on 10k random observations plus `obs_recorded.npy`; latency p50/p99, single thread.
- **`releases/<version>/policy_meta.json`:**
  `{schema_version, policy_version, created_at, git_sha, git_dirty, run_id, checkpoint, config_hash, model_hash, env_id, action_mode, frames: {world: "ENU z-up", body: "FLU", quat: "wxyz"}, obs: {dim, history_k, layout: [{name, start, end, units, frame}], normalization: {mean, var, epsilon, clip}}, action: {dim, raw_range: [-1, 1], command_map, omega_min_rad_s, omega_max_rad_s, motors: [{index, name, px4_motor, betaflight_motor, spin, position_b_m}]}, rates: {physics_hz, policy_hz, decimation}, latency_budget_s, onnx: {opset, inputs, outputs, sha256}, parity: {max_abs_diff, n}, latency: {p50_us, p99_us, threads}, versions: {torch, onnx, onnxruntime, mujoco, sb3}}`.
- **Contract:** `docs/deployment_contract.md` is generated from, and checked against, this file in Phase 8.

## 10. `dronez-sim` MCP server (Phase 4, `tools/mcp_server/`)

Python `mcp` SDK, stdio transport, registered in `.mcp.json` (the FastMCP API is (verify)).
- **Output rule:**
  - every tool returns **one JSON object**, ≤ 4 KB, no prose or log text;
  - arrays are capped at 32 elements with `"truncated": true`;
  - units go in key suffixes (`_s`, `_rad`, `_n`);
  - every number read from disk carries `"source": "<path>"`.
- **Side effects:** none except bounded in-process simulation.

| Tool | Returns |
|---|---|
| `list_runs(limit=10)` | `[{run_id, state, timesteps_done, updated_at}]` |
| `training_status(run_id)` | the `status.json` subset plus ETA |
| `tb_scalars(run_id, tags, points=20)` | per tag: last value, min, max, and up to 20 downsampled points |
| `list_checkpoints(run_id)` | step directories, `best`, and each `ckpt.json` |
| `eval_summary(run_id, checkpoint)` | headline per-scenario metrics plus gate verdicts from `metrics.json` |
| `gate_check(run_id, checkpoint, gate)` | `{gate, pass, criteria: [...], source}` |
| `model_info(drone)` | hash, cache path, mass, CoM, principal inertia, hover command, T/W |
| `sim_rollout(controller, scenario, seed, duration_s≤10)` | summary metrics only (no trajectories) |

## 11. Rates

Final per [ADR-0006 §1](adr/0006-rl-spec.md#1-rates-and-latency):
- Physics 1 kHz (`physics_dt_s = 0.001`, rk4), policy 100 Hz, decimation 10, all in `configs/sim/default.yaml`.
- Latency is configured in seconds and applied as a whole number of physics steps (`round(L / physics_dt)`), 0–20 ms under DR; `latency_budget_s = 0.020`.
- Two switches are pre-authorized by ADR-0006, each gated by stated acceptance criteria: 500 Hz physics (dt 2 ms, decimation 5) and a 200 Hz policy. Nothing in this document depends on the specific values.

## 12. Extension points

- **New airframe:** a YAML only. Add `configs/drone/<new>.yaml`, run `dronez-build-model`, and follow the drone-model-pipeline skill checklist. Non-quad layouts (hex, octo) need a new `layout` value with its motor-order validator and a mixer generalized to n motors. `MotorModel` and the gear derivation are already n-motor.
- **New env:** subclass `DronezBaseEnv`, implement the four hooks, add its `ObsLayout` and reward terms, and register an id. Sim, DR, noise, seeding, the `info` keys and eval reuse come for free.
- **CTBR action mode:** `EnvConfig.action_mode = "ctbr"`. The action `[T_norm, p, q, r]` (FLU, rad/s) drives `control.RateController` plus `Mixer` **inside the substep loop** (physics rate), then the motor model. The inner loop uses the noisy gyro at the physics rate (ADR-0006 §3). The ONNX output then becomes rate setpoints, and the contract adds the FLU→FRD conversion of `p, q, r`.

## 13. Open questions for implementers

1. MuJoCo 4.0.1 (verify): MjSpec method and attribute names (bodies, sites, site actuators, `fullinertia`, explicit-inertial flag, mesh file paths, `to_xml`); default actuator gain and bias; `xfrc_applied` frame and point of application; whether runtime edits of `body_mass`, `body_inertia` or `body_ipos` need `mj_setConst`; accelerometer semantics; `mujoco.rollout` multi-model support.
2. Integrator (Euler, RK4 or implicitfast) and the 500 Hz option: sim-engineer decides using the hover and energy-drift tests against the measured SPS.
3. build123d 0.13 mass-property API and unit handling; scaling meshes to metres on export.
4. ~~`command_map` linear in ω or linear in thrust~~ Resolved: linear in ω, symmetric around hover (ADR-0006 §3).
5. `CAD_VERSION` bump discipline: a forgotten bump means a stale cache. Consider a test that fingerprints `src/dronez/cad/*.py`.
6. ONNX opset and exporter path for torch 2.15.
7. ~~CTBR inner loop: noisy gyro at the physics rate, or true rates~~ Resolved: noisy gyro at 1 kHz (ADR-0006 §3).
