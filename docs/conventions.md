# Dronez conventions (non-negotiable)

The `physics-auditor` checks code against this file. If code and this file disagree, the code is wrong — or this file is changed first, through an ADR.

## Units

SI everywhere: m, kg, s, N, N·m, rad, rad/s. Degrees appear **only** in human-facing reports, plots and log messages (and are converted at the boundary with `np.rad2deg`). Rotor speed is rad/s internally; RPM only in reports and in YAML fields whose name ends in `_rpm`.

## Frames

| Frame | Definition |
|---|---|
| World `W` | MuJoCo default: **z up**, ENU-like (x east, y north). Gravity is `(0, 0, -9.81)` m/s². |
| Body `B` | **FLU**: x forward, y left, z up, origin at the centre of mass. |
| Motor site `M_i` | Parallel to `B`, located at rotor hub `i`; thrust acts along `+z_B`. |

Angular velocity `ω` is expressed in the body frame (what a gyro measures). Linear velocity is in the world frame unless named `*_body`.

## Quaternions and attitude

- Order: **(w, x, y, z)** (MuJoCo order), Hamilton product, unit norm.
- `q` maps body → world: `v_W = R(q) v_B`.
- All quaternion and rotation math lives in `src/dronez/utils/rotations.py` (tested against MuJoCo's `mju_*` functions). Nothing else may implement it ad hoc — call MuJoCo's functions or that module.
- Attitude error: `q_e = q_target* ⊗ q`, `R_e = R_targetᵀ R`; its magnitude is the **geodesic angle** `2·atan2(‖q_e.xyz‖, |q_e.w|)` ∈ [0, π].
- **No Euler angles** in rewards or observations (gimbal lock near inverted flight). Yaw angle is allowed for setpoints and reports only.

## Motor numbering and spin directions

Defined **once**, in the drone YAML (`configs/drone/*.yaml`, `motors:` list). Index order follows PX4's Quad-X order (internal index `i` = PX4 motor `i+1`). Default 5" Quad-X, viewed **from above**, nose up the page:

```
                 FRONT (+x)
         M2 (FL)            M0 (FR)
          CW  ⟳               ⟲  CCW
             \               /
              \             /
   LEFT (+y)       [ CoM ]           RIGHT (-y)
              /             \
             /               \
          CCW ⟲               ⟳  CW
         M1 (RL)            M3 (RR)
                 REAR (-x)

  index  name  position (x, y) [FLU]     spin (from above)  PX4  Betaflight (props-in)
  M0     FR    (+a, -a)                  CCW                 1    2
  M1     RL    (-a, +a)                  CCW                 2    3
  M2     FL    (+a, +a)                  CW                  3    4
  M3     RR    (-a, -a)                  CW                  4    1
  a = diagonal / (2·√2)  (≈ 0.0884 m for the 250 mm default)
```

Derived quantities (computed from the YAML, **never typed by hand**):

- Rotor angular velocity vector: `+z_B` for CCW, `-z_B` for CW.
- Reaction (drag) torque on the body: opposite to the rotor spin, `τ_z,i = -s_i · k_m ω_i²` with `s_i = +1` for CCW, `-1` for CW. So `τ_z,i = -s_i · (k_m/k_f) · T_i`.
- MuJoCo site-actuator gear: `[0, 0, 1, 0, 0, -s_i · k_m/k_f]`, ctrl = thrust in N.
- Mixer: `[T_total, τ_x, τ_y, τ_z]ᵀ = A · T`, with column `i` of `A` = `[1, y_i, -x_i, -s_i·k_m/k_f]`.

Sign sanity (checked by `pytest -m physics`):

| Action | Expected response (FLU) |
|---|---|
| Increase right motors (M0, M3; y < 0) | `τ_x < 0` → **rolls right-side up, i.e. rolls left** (body rotates about −x) |
| Increase front motors (M0, M2; x > 0) | `τ_y < 0` → **nose pitches up** |
| Increase CW pair (M2, M3) | `τ_z > 0` → **yaws left (CCW from above)** |

## Rates and timing

- Physics `dt` and policy decimation come from `configs/sim/*.yaml`. Default physics is 1 kHz and the policy runs at 100 Hz (finalised by the architect ADR in Phase 4).
- The motor model is stepped at the physics rate inside the decimation loop.
- Latency is modelled as a whole number of physics steps of action delay.

## Deployment frames (FLU/ENU → FRD/NED)

Flight stacks such as PX4 and Betaflight use body **FRD** and world **NED**:

- World vectors: `v_NED = (v_N, v_E, v_D) = (y_ENU, x_ENU, -z_ENU)`.
- Body vectors and rates: `v_FRD = (x, -y, -z)_FLU`, so `p_FRD = p_FLU`, `q_FRD = -q_FLU`, `r_FRD = -r_FLU`.
- Attitude: `R_NED←FRD = R_NED←ENU · R_ENU←FLU · R_FLU←FRD` (`rotations.quat_enu_flu_to_ned_frd`).
- Motor order: see the table above. The full contract is in `docs/deployment_contract.md`.

## Code conventions

- Python 3.12, `uv` only (`uv add`, never `pip install`). `ruff` formatting, line length 100.
- `mypy --strict` on `dronez.utils` and `dronez.sim`.
- Generated files (meshes, MJCF, STEP, renders, ONNX, reports) are **never hand-edited**. Change the code or config, then regenerate.
- Heavy artifacts go to `DRONEZ_DATA_DIR` (`dronez.utils.paths`), never into the repo. Only `releases/` holds committed artifacts.
- Randomness: always pass an explicit `np.random.Generator`. Seeds are logged with every run.
- Configs: YAML in `configs/<kind>/`, validated by pydantic models; a run's config hash is recorded with its artifacts.
