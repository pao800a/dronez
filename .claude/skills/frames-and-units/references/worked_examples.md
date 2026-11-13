# Worked conversion examples

All of these are reproduced by `tests/unit/test_rotations.py`.

## 1. Body rates FLU → FRD
A drone rolling right (right side down) at 1 rad/s and yawing left (CCW from above) at 0.5 rad/s:
- In FLU, a positive rotation about +x turns +y (left) toward +z (up). The left side rises, so rolling right is `p_FLU = +1`. Yawing left is positive about +z_FLU: `r_FLU = +0.5`.
- `(p, q, r)_FRD = (p, −q, −r)_FLU = (+1, 0, −0.5)`. In FRD, roll-right is still +p, and yaw-left is −r (PX4: yaw-right is +r). ✔

## 2. World position ENU → NED
Drone 3 m above the origin, 2 m east, 1 m north: `p_ENU = (2, 1, 3)` → `p_NED = (1, 2, −3)`.

## 3. Attitude quaternion ENU/FLU → NED/FRD
Level, nose pointing **east** in ENU: `q_ENU←FLU = (1, 0, 0, 0)`.
`q_NED←FRD = q_NED←ENU ⊗ q ⊗ q_FLU←FRD`, with `q_NED←ENU = (0, √½, √½, 0)` and `q_FLU←FRD = (0, 1, 0, 0)`:
`(0, √½, √½, 0) ⊗ (0, 1, 0, 0) = (−√½, 0, 0, −√½)`, canonically `(√½, 0, 0, √½)`. That is +90° about NED z (down), i.e. heading north → east. Nose east ✔. Check: FRD forward maps to NED `(0, 1, 0)` (east), and FRD down maps to NED `(0, 0, 1)` (down). `rotations.quat_enu_flu_to_ned_frd` computes this.

## 4. Motor order mapping
| internal | position | spin | PX4 output | Betaflight (props-in default) |
|---|---|---|---|---|
| M0 | FR | CCW | 1 | 2 |
| M1 | RL | CCW | 2 | 3 |
| M2 | FL | CW | 3 | 4 |
| M3 | RR | CW | 4 | 1 |

## 5. Gravity in the body frame
Level: `g_B = (0, 0, −1)`. Inverted (180° about x): `g_B = (0, 0, +1)`. Nose up 90° (rotation −90° about y): body +x points to world +z, and `g_B = R(q)ᵀ(0, 0, −1) = (−1, 0, 0)`.

## 6. Accelerometer at rest
An accelerometer measures specific force `f = a − g`. At rest and level: `f_B = (0, 0, +9.81)`.
