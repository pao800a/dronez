---
name: drone-model-pipeline
description: Regenerate the Dronez drone model end to end (YAML params → build123d CAD → mass properties → meshes/STEP/renders → MuJoCo MjSpec model), validate it with a checklist, or add a new airframe config. Use when the drone geometry, materials, motors/props or battery change, when generated model files are missing or stale, or when someone asks to add or modify an airframe.
---
# Drone model pipeline

Single source of truth: `configs/drone/<name>.yaml`. Everything downstream is generated into `$DRONEZ_DATA_DIR/models/<name>-<config_hash>/` and **never hand-edited**.

## Regenerate
```bash
uv run dronez-build-model --drone default_5in          # CAD → mass_props.json, meshes, STEP, renders, model
uv run dronez-build-model --drone default_5in --force  # ignore the cache
uv run pytest -m "cad or physics" -q                   # validate
```
The output directory name contains the config hash, so a changed YAML gets a fresh build automatically.

## Validation checklist
1. `mass_props.json`: total mass within ±10 % of `target_auw_kg`. Inertia is SPD and satisfies `Ixx+Iyy ≥ Izz` (and permutations). CoM xy within a few mm of 0 for symmetric frames.
2. Look at the renders (`render_top.png`, `render_side.png`, `render_iso.png`) with Read: arms on the diagonals, props clear of each other and the frame, battery on top, camera at the front (+x).
3. Motor positions in the model match the YAML and the diagram in `docs/conventions.md` (M0 FR, M1 RL, M2 FL, M3 RR).
4. Hover throttle and thrust-to-weight printed by the builder are plausible (T/W ≈ 4 for the default; hover ≈ 25 % of max thrust).
5. `uv run pytest -m physics -q` is green (see the physics-sanity skill).

## Add a new airframe
1. Copy `configs/drone/default_5in.yaml` to `configs/drone/<new>.yaml` and change geometry, materials, motor/prop constants (`k_f`, `k_m`, ω limits, τ_m), battery and `target_auw_kg`.
2. Keep the `motors:` list in PX4 Quad-X order with correct `spin`. The mixer and yaw signs derive from it.
3. If real weighed values exist, fill `measured_overrides:` (mass, CoM, inertia); they replace the CAD values.
4. Regenerate and run the checklist. Commit only the YAML (generated files are gitignored).
