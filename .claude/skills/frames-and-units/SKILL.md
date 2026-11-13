---
name: frames-and-units
description: Reference for Dronez frames, units and quaternion conventions (SI, world z-up ENU-like, body FLU, wxyz Hamilton quaternions, geodesic attitude error, PX4 Quad-X motor order) with worked conversions FLU↔FRD, ENU↔NED and quaternion conversion for PX4/Betaflight. Use whenever writing or reviewing code that touches rotations, signs, motor order, sensor frames or deployment interfaces.
user-invocable: true
---
# Frames and units (quick reference)

The authoritative text is `docs/conventions.md`; worked examples are in [references/worked_examples.md](references/worked_examples.md).

- **Units:** SI only (m, kg, s, N, N·m, rad, rad/s). Degrees only in reports.
- **World:** z up, ENU-like. g = (0, 0, −9.81).
- **Body:** FLU (x forward, y left, z up). Gyro rates are body-frame.
- **Quaternion:** (w, x, y, z), Hamilton, `q` maps body → world (`v_W = R(q) v_B`). Code: `dronez.utils.rotations` only.
- **Attitude error:** `q_e = q_t* ⊗ q`, angle `2·atan2(‖q_e.xyz‖, |q_e.w|)`. No Euler angles in rewards or obs.
- **Motors (PX4 Quad-X order):** M0 FR CCW · M1 RL CCW · M2 FL CW · M3 RR CW. Reaction torque `τ_z,i = −s_i k_m ω_i²` (s = +1 CCW).
- **Signs (FLU):** right motors up → roll left (ω_x < 0). Front motors up → nose up (ω_y < 0). CW pair up → yaw left (ω_z > 0).
- **To FRD/NED:** `v_FRD = (x, −y, −z)`, `v_NED = (y_ENU, x_ENU, −z_ENU)`. Attitude via `rotations.quat_enu_flu_to_ned_frd`.
