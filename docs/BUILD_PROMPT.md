# DRONEZ: build prompt for Claude Code

> Paste this whole file into a fresh Claude Code session opened in
> `C:\Users\gsppp\OneDrive\Desktop\PAO800A_REPO\DRONEZ\Dronez`, or save it there as
> `docs/BUILD_PROMPT.md` and tell Claude Code: "Read docs/BUILD_PROMPT.md and execute it."

---

## 1. Mission

Build **Dronez**: a production-quality Python repository that

1. designs a **detailed, parametric quadcopter in code-CAD (build123d)**, derives its mass properties from the CAD, and exports visual meshes;
2. turns that CAD into a **MuJoCo simulation model** with a realistic motor/propeller model;
3. trains, with **reinforcement learning**, a neural policy that gives the drone **stable flight attitude**: holding a level or commanded attitude, rejecting disturbances, recovering from extreme orientations up to inverted, and holding hover;
4. hardens the policy for **sim-to-real** (domain randomization, latency, sensor noise, motor dynamics), and **exports it to ONNX** with a documented, hardware-agnostic deployment contract.

You are the **orchestrator**. You plan, delegate to the project subagents defined below, integrate their work, and enforce the quality gates. You are also responsible for setting up the Claude Code tooling of the repo itself: `CLAUDE.md`, subagents, skills, hooks, MCP servers and permissions. That tooling ships with the repo, so future sessions keep working the same way.

---

## 2. Decisions already made (do not re-open without a strong, written reason)

| Topic | Decision |
|---|---|
| Physics | **MuJoCo** (latest stable `mujoco` from PyPI, version pinned in `uv.lock`) + **Gymnasium** API |
| CAD | **build123d** (parametric Python CAD). Export STL/OBJ visual meshes and STEP; mass properties come from the CAD solids |
| Drone | Generic **X-configuration quad**, fully parametric. Default parameter set: ~250 mm motor-to-motor diagonal, 5" props, 22xx-class motors, 4S battery, ~600–700 g take-off mass, thrust-to-weight ≈ 4. The parameters live in YAML so other airframes are a config change |
| RL scope | Attitude stabilization → attitude + hover/altitude hold → sim-to-real hardening → ONNX export |
| Real hardware target | **None yet.** Keep deployment generic: ONNX plus a precise interface contract (frames, units, normalization, rates, latency budget) and frame-conversion notes for FRD/NED stacks (PX4, Betaflight) |
| Compute | **No NVIDIA GPU.** The default is CPU training. An Intel GPU (PyTorch `xpu` device) *may* exist. Support it as an opt-in device, and only install XPU wheels if a benchmark shows a real speed-up (for small MLP policies the CPU usually wins, because env stepping dominates). No CUDA-only dependencies (no MJX-Warp, Isaac or CUDA JAX) |
| OS | **Windows native is primary** (Python via `uv`). WSL2 Ubuntu is available as a fallback if a dependency misbehaves on Windows. CI runs on Linux **and** Windows |
| RL library | **Stable-Baselines3 PPO** as the primary algorithm (SAC as an optional comparison). Write a custom CleanRL-style PPO with an **asymmetric actor-critic** (privileged critic) *only if* the robustness gates in §8 fail with SB3. Record that decision in an ADR |
| Tracking | **TensorBoard** (local). Weights & Biases as an optional flag, off by default |
| Repo hosting | **GitHub** repo + GitHub Actions CI + GitHub MCP |
| Claude plan | **Claude Pro**. Opus 5.5 is available but usage limits are tight: budget Opus deliberately (see §4) |

---

## 3. How you operate

- **Start in plan mode.** Explore the empty folder, check the environment (Windows version, Python, `uv`, `git`, `gh`, whether `torch.xpu` is available), then present a written plan that follows §9. Wait for my approval once. After that, work autonomously phase by phase.
- **Ask me only about these things**, all up front in one go: (a) the GitHub repo name and public/private visibility; (b) whether `GITHUB_PAT` is already set as a Windows user environment variable (if not, give me the exact `setx` command and the minimum token scopes); (c) whether I have a Context7 API key (it's optional). Decide everything else yourself, document the decision in `docs/adr/`, and move on.
- **Use the task list** for every phase, and tick items as they finish.
- **Git discipline:** create one feature branch per phase, make small conventional commits, and open a PR through the GitHub MCP when a phase passes its gate. Merge only when CI is green. Never force-push `main`.
- **Never fabricate results.** Every number you report (metrics, SPS, success rates) must come from a file the code produced. Cite the path.
- **Resume protocol (Pro limits will interrupt you):** keep `docs/PROGRESS.md` current. It records the phase, what's done, what's next, open problems, and the latest run IDs with key metrics. A new session must be able to continue from `CLAUDE.md` + `docs/PROGRESS.md` alone. Update it at the end of each phase and before any long-running job.
- **Long jobs:** start training runs in the background (`run_in_background`), log to files, and check them periodically through the training-status skill or MCP tool. Never block the session on a multi-hour run, and never tail huge logs into the context. Delegate summarizing to `log-summarizer`.
- **OneDrive caution:** the repo lives in a OneDrive-synced folder. Syncing a virtualenv, checkpoints, TensorBoard event files and videos causes file locks and heavy sync traffic. Therefore:
  - put the virtualenv outside OneDrive by setting `UV_PROJECT_ENVIRONMENT` (e.g. `%LOCALAPPDATA%\Dronez\.venv`). Document this in the README and in a `scripts/setup_windows.ps1` script;
  - write all heavy artifacts (runs, checkpoints, videos, eval reports, generated meshes) under a configurable `DRONEZ_DATA_DIR` that defaults to `%LOCALAPPDATA%\Dronez\data` on Windows and `~/.local/share/dronez` on Linux;
  - `.gitignore` everything generated. Only final, small release artifacts (for example the exported `.onnx` and its metadata) are committed, under `releases/`.

---

## 4. Model routing and subagents

### 4.1 Routing policy (Pro plan)

- **Main session (you, the orchestrator):** run on **Sonnet 5.5** by default. If `/model` offers `opusplan` (Opus in plan mode, Sonnet when executing), use it. Switch the main session to Opus yourself only for the initial plan, and only if `opusplan` isn't available.
- **Opus 5.5** is reserved for decisions where a subtle mistake silently ruins the project: architecture/ADR design, reward/observation/action design, independent physics review, and diagnosis after **two** failed fix attempts. Budget: **no more than ~3 Opus subagent calls per phase.** Write in `docs/PROGRESS.md` every time you invoke an Opus agent and why.
- **Sonnet 5.5** handles all implementation: CAD, sim, env, training code, tests, CI, export.
- **Haiku 5.5** handles high-volume, low-risk work: summarizing logs and CI output, docs, changelogs, commit/PR text, repo searches. The built-in `Explore` agent already covers read-only codebase search.
- **Do not set `CLAUDE_CODE_SUBAGENT_MODEL`.** It overrides every agent's `model` field.
- Subagents must **return concise summaries** (≤ 250 words, plus file paths). Detailed output goes to files, not into your context.
- **Parallelize** independent work. For example, `cad-engineer` (parametric parts) and `test-engineer` (CI/tooling skeleton) can run at the same time in Phase 1.

### 4.2 Subagents to create in `.claude/agents/`

Before writing these files, check the current subagent frontmatter format in the Claude Code docs (via Context7 or https://code.claude.com/docs/en/sub-agents); it changes. Use at least `name`, `description`, `tools`, `model`, and use optional fields such as `skills` where they're supported. Write the `description` so automatic delegation triggers correctly (say *when* to use the agent). Each body must contain the agent's responsibilities, the files it owns, the conventions it must respect (link to `docs/conventions.md`), its definition of done, and its output format.

Example of the format (adapt it):

```markdown
---
name: physics-auditor
description: Independent read-only reviewer for flight dynamics. Use PROACTIVELY after any change to src/dronez/sim/, src/dronez/cad/mass_props.py, motor model, frames/quaternion utils, or reward/observation code, and before merging any PR that touches them.
tools: Read, Grep, Glob, Bash
model: opus
---
You are a flight-dynamics and rigid-body-physics reviewer...
```

| Agent | Model | Tools | Owns / does |
|---|---|---|---|
| `architect` | **opus** | Read, Grep, Glob, Write (docs only), WebFetch | System design, ADRs, the observation/action/reward spec, curriculum and domain-randomization plan, PPO hyperparameter starting points, sim-to-real strategy. Called at Phase 0, at the start of Phase 5, and when training is stuck after two attempts. Writes only to `docs/`. |
| `physics-auditor` | **opus** | Read, Grep, Glob, Bash (read-only commands, running tests) | Independent review of frames, signs, units, motor numbering and spin directions, mixer, yaw-torque sign, inertia tensor validity, motor lag, integration timestep and decimation. It never edits code: it returns a findings list with severity. It must not be the same agent that wrote the code under review. |
| `cad-engineer` | sonnet | Read, Write, Edit, Bash | `src/dronez/cad/`: parametric parts (frame plates, arms, motor bells and stators, props, FC/ESC stack, battery, standoffs, camera), assembly, per-part materials and densities, mass and inertia from the solids, STL/OBJ/STEP export, preview renders. |
| `sim-engineer` | sonnet | Read, Write, Edit, Bash | `src/dronez/sim/`: MJCF generation through the MuJoCo `MjSpec` Python API (no hand-edited XML), simplified collision geometry, motor/prop model, sensors, disturbances, ground. |
| `rl-engineer` | sonnet | Read, Write, Edit, Bash | `src/dronez/envs/` and `src/dronez/rl/`: Gymnasium envs, wrappers, domain randomization, curriculum, vectorization, training entry points, callbacks, checkpointing. |
| `controls-engineer` | sonnet | Read, Write, Edit, Bash | `src/dronez/control/`: a cascaded PID attitude/rate controller and a geometric SO(3) attitude controller as baselines and sim validators. |
| `training-analyst` | sonnet | Read, Grep, Glob, Bash | Reads TensorBoard scalars and eval reports (through the project MCP or scripts) and diagnoses training: reward hacking, entropy collapse, value-loss blow-up, action saturation, oscillation. Proposes ranked fixes and escalates to `architect` after two failed attempts. |
| `deploy-engineer` | sonnet | Read, Write, Edit, Bash | `src/dronez/deploy/`: ONNX export with the normalization baked in, onnxruntime parity tests, latency benchmark, the deployment contract, an optional C header with weights. |
| `test-engineer` | sonnet | Read, Write, Edit, Bash | `tests/`, `pyproject` tooling (ruff, mypy, pytest markers), pre-commit, `.github/workflows/`, the physics sanity suite. |
| `log-summarizer` | **haiku** | Read, Grep, Glob, Bash | Condenses long training logs, CI logs and pytest output into ≤ 150-word summaries containing the exact failing items. |
| `docs-writer` | **haiku** | Read, Write, Edit, Grep, Glob | README, docstrings, `docs/` prose, CHANGELOG. It never changes code behaviour. |
| `git-ops` | **haiku** | Read, Bash, GitHub MCP tools | Commit messages, PR titles and bodies, labels, issue creation from TODOs, and checking CI status through the GitHub MCP. |

---

## 5. Claude Code project configuration to create

### 5.1 `CLAUDE.md` (keep it under ~150 lines; link to docs instead of inlining them)

It must contain:

- a one-paragraph project summary;
- the exact commands (`uv sync`, `uv run dronez-build-model`, `uv run dronez-train --config ...`, `uv run dronez-eval ...`, `uv run dronez-export ...`, `uv run pytest -m fast`, `uv run ruff check`, `uv run mypy src`);
- the **conventions** below (also stated in full in `docs/conventions.md`);
- the rules "use `uv add`, never `pip install`" and "generated files are never hand-edited";
- the subagent routing table (short form) and the Opus budget rule;
- a pointer to `docs/PROGRESS.md` for current state.

**Conventions (non-negotiable, so the physics-auditor can check them):**

- SI units everywhere (m, kg, s, N, N·m, rad). Degrees only in human-facing reports.
- World frame: MuJoCo default, **z-up (ENU-like)**. Body frame: **FLU** (x forward, y left, z up).
- Quaternions: MuJoCo order **(w, x, y, z)**. All quaternion math lives in one module (`utils/rotations.py`) with tests; nothing else implements it ad hoc.
- Attitude error: the geodesic angle on SO(3). Avoid Euler angles in the reward and observations (gimbal lock near inverted flight).
- Motor numbering and spin directions are defined once, in the drone config, with an ASCII diagram in `docs/conventions.md`. The mixer and the yaw-torque signs are derived from that definition, never typed by hand.
- The deployment contract documents the FLU/ENU → FRD/NED conversion explicitly.

### 5.2 Skills in `.claude/skills/<name>/SKILL.md`

Check the current SKILL.md format first. Write each `description` so it triggers on the right requests. Put helper scripts in each skill's `scripts/` folder and long references in `references/`.

| Skill | Purpose |
|---|---|
| `drone-model-pipeline` | Regenerate params → CAD → mass properties → meshes → MJCF; a validation checklist; how to add a new airframe config. |
| `physics-sanity` | Run and interpret the physics sanity suite (hover thrust = weight, free fall, single-motor roll/pitch sign, yaw-torque sign, inertia validity, motor-lag step response). |
| `train-policy` | Launch smoke and full training runs in the background, choose `n_envs` from the SPS benchmark, find logs and checkpoints, and resume a run. |
| `training-diagnostics` | A checklist for diagnosing PPO runs: which scalars to read, healthy ranges, the symptoms of reward hacking, oscillatory policies, saturation, and value divergence, plus the standard fixes. |
| `evaluate-policy` | Run the eval scenario suite and robustness sweeps, generate the Markdown/HTML report, render MP4s, and compare against the PID/geometric baselines. |
| `export-policy` | Export to ONNX, run the parity and latency checks, and update the deployment contract and the release folder. |
| `frames-and-units` | A reference skill: the conventions above plus worked conversion examples (FLU↔FRD, ENU↔NED, quaternion conventions). |

### 5.3 Hooks in `.claude/settings.json`

Write every hook as a **Python script** in `.claude/hooks/` so it runs the same on Windows and Linux. Invoke it as `uv run python "$CLAUDE_PROJECT_DIR/.claude/hooks/<script>.py"`. Check the current hooks reference for the JSON input/output schema and the exit-code semantics. **Test every hook** by piping sample JSON into it, and record those tests in `tests/hooks/`.

| Event | Matcher | Behaviour |
|---|---|---|
| `SessionStart` | — | Inject a short status block: git branch and dirty state, the `docs/PROGRESS.md` "Next" section, the MuJoCo/torch versions, the available device (cpu/xpu), the latest run ID and its status. |
| `PreToolUse` | `Bash` | Block (exit 2 with a reason) `pip install` (tell Claude to use `uv add`), `rm -rf` on the repo root or the data dir, `git push --force` to `main`, and `git reset --hard` on `main`. Warn when a training command is launched in the foreground with more than 200k timesteps. |
| `PreToolUse` | `Edit\|Write` | Block edits to `.env*`, `uv.lock` (only `uv` may change it), and anything under the generated-assets directory or `releases/` (these are produced by the pipeline). |
| `PostToolUse` | `Edit\|Write` | For `*.py` files, run `ruff format` and `ruff check --fix` on the edited file only. Report remaining issues back to Claude. It must be fast (< 3 s). |
| `Stop` | — | Run `pytest -m fast -q` (target < 60 s). If it fails and `stop_hook_active` is false, block the stop with a concise failure summary so Claude fixes it. If `stop_hook_active` is true, allow the stop (no infinite loops). |
| `SubagentStop` | — | Append `{timestamp, agent, duration, summary_first_line}` to `.claude/logs/agents.jsonl` (gitignored), to audit Opus usage. |
| `PreCompact` | — | Remind Claude to update `docs/PROGRESS.md` before the context is compacted. |
| `Notification` | — | Optional: a Windows toast via PowerShell when Claude needs input. It must degrade silently on Linux and CI. |

Also set `permissions` in `.claude/settings.json`. Allow `uv run *`, `uv sync`, `uv add *`, `git status/diff/log/add/commit/checkout/branch/switch/push` (non-force), `gh *`, and `pytest`. Deny reads of `.env*`. Keep anything personal in `.claude/settings.local.json` (gitignored).

### 5.4 MCP servers in `.mcp.json` (project scope, committed, **no secrets inline**)

Use environment-variable expansion for every credential, so the file is safe to commit.

1. **GitHub (official remote server)**: `type: http`, `url: https://api.githubcopilot.com/mcp/`, header `Authorization: Bearer ${GITHUB_PAT}`. Limit the toolsets to what's needed (repos, issues, pull_requests, actions) with the `X-MCP-Toolsets` header, if the current server docs still support it. Use it for repo creation, PRs and CI status. `gh` is the fallback.
2. **Context7** (`https://mcp.context7.com/mcp`, optional `CONTEXT7_API_KEY` header): up-to-date docs for MuJoCo, Gymnasium, Stable-Baselines3, build123d, onnx/onnxruntime and PyTorch. Add a rule in `CLAUDE.md`: "Before using an API from these libraries you're not certain about, check Context7."
3. **`dronez-sim`, a project-local MCP server you build** (Python `mcp` SDK / FastMCP, stdio, under `tools/mcp_server/`, launched with `uv run`), registered once Phase 5 starts. Tools:
   - `model_info(config)` → mass, inertia, hover throttle, thrust-to-weight, motor positions;
   - `render_snapshot(config, qpos?, camera?)` → path to a PNG (so agents can *look* at the drone);
   - `rollout(policy_or_controller, scenario, seed)` → a metrics JSON plus an MP4 path;
   - `training_status(run_id)` → the latest scalars from the TensorBoard event files, summarized;
   - `compare_runs(run_ids, metrics)` → a compact table.
   These tools must return **small JSON**, never raw arrays.

Keep the MCP set this small, because every server costs context. Don't add the community MuJoCo MCP servers: the project-local server is tailored to this repo and auditable.

---

## 6. Technical specification

### 6.1 CAD (build123d)

- `configs/drone/default_5in.yaml` holds the geometry, materials (densities for carbon fibre, aluminium, copper/steel motor parts, LiPo, PLA/TPU), motor and prop specs, and the battery.
- Build each part as a function of the parameters. The assembly places the parts and keeps a **part → material** map.
- **Mass properties come from the CAD:** volume × density per part, giving total mass, centre of mass and the full inertia tensor about the CoM. Also support a "measured overrides" section in the YAML, for when real weighed values exist later.
- Exports:
  - simplified visual meshes per part (decimated if needed so MuJoCo loads them fast), as STL/OBJ;
  - a STEP file of the assembly;
  - a JSON of mass properties;
  - preview PNG renders (top, side, iso).
- Tests: the inertia is symmetric positive-definite and satisfies the triangle inequalities; the total mass is within ±10 % of the target AUW in the config; the CoM lies within a few mm of the geometric centre in x/y for the symmetric default.

### 6.2 Simulation (MuJoCo, built through `MjSpec`)

- One free-joint body that carries the CAD-derived inertial properties. The visual meshes come from the CAD, and the collision geometry is simplified primitives (boxes and cylinders for the arms and body, and prop-disk cylinders that can be toggled).
- **Motor model, in the env layer and fully vectorizable in numpy:**
  - normalized command → target rotor speed;
  - first-order lag with time constant τ_m (default ~30 ms, randomized later);
  - thrust T = k_f ω² and drag torque Q = k_m ω²;
  - saturation at min/max rpm.
  - Apply thrust through site actuators at each motor, with a `gear` that includes the yaw-torque coefficient (±k_m/k_f by spin direction), so MuJoCo applies force and reaction torque consistently. Optionally add rotor-induced linear drag and ground effect behind config flags.
- Sensors: gyro, accelerometer and frame orientation (ground truth). Noise, bias and bias random-walk are added in the env layer so they can be randomized.
- Timestep: physics at 1 kHz (or 500 Hz if the stability tests pass and SPS matters). Policy at **100–250 Hz** via decimation (a config value).
- Disturbances: random impulse torques and forces, constant plus gusting wind force, and a random **action delay of 0–N policy steps** (latency).
- **Physics sanity suite (`pytest -m physics`):**
  - hover at the analytically computed throttle holds altitude within ±2 cm over 2 s;
  - free fall matches g;
  - raising the two right motors rolls the drone left (FLU sign check), and the same check for pitch;
  - speeding up the CW pair produces the expected yaw direction;
  - the motor step response matches τ_m;
  - simulation is deterministic for a fixed seed.

### 6.3 Classical baselines (validate the sim before any RL)

A cascaded PID (attitude → rate → mixer) and a geometric SO(3) controller. They must stabilize from 30° tilt and hold hover. These baselines are the reference row in every eval report.

### 6.4 RL environments (Gymnasium)

- Envs: `DronezAttitude-v0` (attitude hold and recovery; altitude ignored or softly regulated) and `DronezHover-v0` (attitude plus altitude/position hold at a setpoint).
- **Observations (actor):**
  - the attitude-error representation (the first two columns of the rotation-error matrix, or the gravity vector in the body frame plus a yaw-error encoding; the `architect` decides);
  - body angular rates from the noisy gyro;
  - the last K actions;
  - the target attitude;
  - for hover, the altitude error and vertical velocity.
  - Observations are normalized with running statistics that are frozen at export.
- **Privileged observations** (true parameters, true state, disturbance) go in `info` for an optional asymmetric critic.
- **Actions:** 4 normalized motor commands in [-1, 1] (end-to-end). Also offer a configurable **collective thrust + body-rate** (CTBR) mode for stacks that accept rate setpoints.
- **Reward (the architect owns the spec; start from this):**
  - an attitude-error term (geodesic angle);
  - an angular-rate penalty;
  - action-magnitude and **action-rate (smoothness)** penalties;
  - for hover, altitude/position terms;
  - a small alive bonus;
  - termination on crash or excessive rate.
  - Each term is logged separately to TensorBoard.
- **Curriculum:** initial tilt range from small (≤ 15°) up to full SO(3) including inverted, initial angular rates up to ~5 rad/s. Disturbances and domain-randomization ranges widen as success rates rise.
- **Domain randomization:** mass ±20 %, inertia ±20 %, CoM offset of a few mm, k_f ±15 %, k_m ±20 %, τ_m 15–60 ms, per-motor thrust mismatch ±5 %, latency 0–20 ms, sensor noise and bias, battery-sag thrust scaling, wind.
- **Vectorization:** benchmark single-env **SPS** first. Use `SubprocVecEnv` with `n_envs ≈ physical cores − 2`. If SPS is too low for the time budget, profile it and move the motor model, noise and observation building into batched numpy, or use `mujoco.rollout`, before considering anything heavier.

### 6.5 Training

- SB3 PPO; MLP actor and critic (start at 2×64 for the actor, wider for the critic); tanh outputs; `gSDE` or tuned log-std; YAML configs per stage; seeds logged; checkpoints plus the normalization stats saved together.
- Device selection in `utils/device.py`: `cpu` by default; `--device xpu` only if `torch.xpu.is_available()`; log which device was used. Run a one-off CPU-vs-XPU benchmark and record the result in an ADR.
- Size the full runs from the measured SPS so a stage completes in **≤ 4–6 hours on this laptop**. State the expected wall-clock time before launching.

### 6.6 Evaluation

- Fixed, seeded scenario suite: hover hold; step attitude commands; recovery from random orientations (bins: <30°, 30–90°, 90–180°/inverted); impulse disturbances; wind; a robustness sweep over the domain-randomization extremes; latency stress.
- Metrics:
  - success rate;
  - time to recover below 5°;
  - RMS attitude error;
  - overshoot;
  - motor saturation %;
  - mean |Δa| (smoothness);
  - action spectrum energy above ~20 Hz (oscillation that would burn real motors).
- Output: a Markdown + HTML report per checkpoint with plots and MP4s, and comparison rows against the PID and geometric baselines.

### 6.7 Sim-to-real export

- ONNX export with the normalization and action scaling baked in, and the opset documented. Parity against PyTorch on 10k random and recorded observations must give max |Δ| ≤ 1e-5. Benchmark the latency (target < 50 µs per inference on a desktop CPU).
- `docs/deployment_contract.md`: observation vector layout, units, frames, normalization constants, rates, required latency budget, action semantics and the motor-order mapping, FRD/NED conversion, and a safety section (arming, failsafe, saturation handling).
- Optional stretch: export the weights to a C header plus a tiny reference C inference function, checked against ONNX outputs.

---

## 7. Repository layout (target)

```
Dronez/
├─ CLAUDE.md  README.md  CHANGELOG.md  LICENSE  pyproject.toml  uv.lock  .mcp.json  .gitignore  .pre-commit-config.yaml
├─ .claude/  settings.json  agents/  skills/  hooks/
├─ .github/workflows/ci.yml
├─ configs/  drone/  sim/  train/  eval/
├─ src/dronez/
│  ├─ cad/       parts.py  assembly.py  materials.py  mass_props.py  export.py  render.py
│  ├─ sim/       builder.py (MjSpec)  motor_model.py  sensors.py  disturbances.py
│  ├─ control/   pid.py  geometric.py  mixer.py
│  ├─ envs/      attitude_env.py  hover_env.py  randomization.py  curriculum.py  wrappers.py
│  ├─ rl/        train.py  callbacks.py  policies.py
│  ├─ eval/      scenarios.py  metrics.py  report.py  video.py
│  ├─ deploy/    export_onnx.py  parity.py  c_export.py
│  └─ utils/     rotations.py  config.py  device.py  paths.py (DRONEZ_DATA_DIR)
├─ tools/mcp_server/
├─ tests/  unit/  physics/  integration/  hooks/
├─ scripts/  setup_windows.ps1  setup_wsl.sh  benchmark_sps.py
├─ releases/   (committed: final .onnx + metadata only)
└─ docs/  architecture.md  conventions.md  deployment_contract.md  PROGRESS.md  adr/
```

Expose the console scripts (`dronez-build-model`, `dronez-train`, `dronez-eval`, `dronez-export`) through `pyproject.toml`. Use Python **3.12**.

---

## 8. Quality gates and targets

The architect may revise a target, but only through an ADR that explains why.

| Gate | Pass criteria |
|---|---|
| G1 CAD | Mass-property tests pass; renders look right (check the PNGs yourself with Read); total mass within ±10 % of the target |
| G2 Sim | The full physics sanity suite passes; `physics-auditor` reports no high-severity findings |
| G3 Baselines | PID and geometric controllers stabilize from 30° and hold hover ±5 cm |
| G4 RL attitude | Over 1,000 seeded episodes with nominal parameters and sensor noise: hover-hold RMS tilt < 2°; recovery from < 90° tilt with ≥ 95 % success and < 1.5 s to < 5° |
| G5 Full recovery | Including inverted starts and rates up to 5 rad/s: ≥ 90 % success, < 2.5 s; no ground contact when starting from ≥ 3 m |
| G6 Robustness | ≥ 90 % success across the domain-randomization-extreme sweep and 20 ms latency; smoothness and high-frequency action-energy metrics no worse than the PID baseline |
| G7 Export | ONNX parity ≤ 1e-5; latency target met; deployment contract complete and reviewed by `physics-auditor` (frames section) |
| CI | ruff, mypy (strict on `src/dronez/utils` and `sim`), and pytest fast plus physics suites pass on `ubuntu-latest` and `windows-latest`, plus a ~2-minute smoke training on CPU |

---

## 9. Phased plan

0. **Bootstrap.** Check the environment; set up `uv` with the venv outside OneDrive; create `pyproject.toml`, the repo skeleton, `CLAUDE.md`, `docs/conventions.md`, `.claude/` (agents, skills, hooks, settings), `.mcp.json` (GitHub + Context7), pre-commit, and the CI skeleton. Create the GitHub repo (after my answers) and push. `architect` (Opus) writes `docs/architecture.md` and the first ADRs.
1. **CAD:** `cad-engineer` builds the parametric drone; then gate G1.
2. **Sim:** `sim-engineer` builds the MJCF with `MjSpec`, the motor model and the sensors; `test-engineer` writes the physics suite; `physics-auditor` (Opus) reviews; then gate G2.
3. **Baselines:** `controls-engineer` writes the PID and geometric controllers; gate G3. Run a first eval report with video.
4. **Env and tooling:** `rl-engineer` builds the Gymnasium envs, domain randomization and curriculum; run the SPS benchmark and the CPU/XPU benchmark; `architect` (Opus) finalizes the obs/action/reward spec; build the `dronez-sim` MCP server and register it.
5. **Training, attitude:** smoke run, then a full run in the background. `training-analyst` monitors through the MCP; iterate; gates G4 → G5.
6. **Sim-to-real hardening:** full domain randomization, latency, noise, wind; robustness sweeps; gate G6. Use the asymmetric-critic ADR only if needed.
7. **Hover env:** train or fine-tune the hover policy; evaluate; then export.
8. **Export:** `deploy-engineer` handles ONNX, parity, latency and the contract; gate G7. Tag the release `v0.1.0` and attach the report.
9. **Polish:** `docs-writer` finishes the README (quick start on Windows, the OneDrive notes, results table, GIFs), CHANGELOG and docs; final PR.

At the end of every phase: update `docs/PROGRESS.md`, open the PR, confirm CI is green, and report to me in ≤ 10 lines (what passed, key metrics with file paths, next phase).

---

## 10. Definition of done

- `uv sync && uv run dronez-build-model && uv run pytest` works on a clean Windows machine by following the README.
- One command reproduces the training of the released policy, and one command reproduces its eval report.
- All gates G1–G7 are passed and documented, with links to the reports.
- `releases/v0.1.0/` contains `policy.onnx`, `policy_meta.json` (obs layout, normalization, rates, git SHA, config hash) and the eval report.
- The `.claude/` agents, skills and hooks are committed, documented in `CLAUDE.md`, and the hooks are tested.

**Start now:** enter plan mode, inspect the environment, ask me the three questions from §3 in one message, then present the plan.
---

## 11. Answers to the §3 questions (already given, so don't ask again)

- (a) GitHub repo: **`dronez`**, **private**.
- (b) `GITHUB_PAT` **is already set** as a Windows user environment variable. Check it's visible in the session (`echo %GITHUB_PAT%` must not be empty, and don't print the value). If it isn't, the terminal was opened before `setx`: tell me to restart Claude Code.
- (c) **No Context7 API key.** Register Context7 without the `CONTEXT7_API_KEY` header (it runs on the lower keyless rate limit).
- Housekeeping: the folder may hold an empty `.probe` file left by an earlier write test. Delete it.

So: enter plan mode, inspect the environment, then present the plan, without asking these questions again.
