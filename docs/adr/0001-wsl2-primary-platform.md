# ADR-0001: WSL2 Ubuntu as primary dev platform

- **Status:** Accepted (2026-11-13)
- **Deviates from:** BUILD_PROMPT §2 ("Windows native is primary").

## Context
The build prompt assumed a Windows-native checkout inside a OneDrive-synced folder. In practice the session was started in WSL2 Ubuntu 24.04 with the repo at `~/projects/dronez` on ext4. OneDrive syncing a virtualenv, checkpoints and TensorBoard files causes file locks and heavy sync traffic. MuJoCo offscreen rendering (EGL/OSMesa), multiprocessing (`fork`) and most RL tooling are also smoother on Linux. The owner approved WSL2 as primary.

## Decision
- WSL2 Ubuntu is the primary development and training platform. The repo lives on the Linux filesystem, outside OneDrive. The virtualenv is the default in-repo `.venv` (gitignored).
- Windows native stays a supported platform:
  - `scripts/setup_windows.ps1` sets `UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Dronez\.venv` (outside OneDrive) and `DRONEZ_DATA_DIR`;
  - `dronez.utils.paths` resolves `DRONEZ_DATA_DIR` per OS (`%LOCALAPPDATA%\Dronez\data` / `~/.local/share/dronez`);
  - every hook is a Python script, and CI runs `windows-latest` as well as `ubuntu-latest`.
- Torch comes from the PyTorch CPU wheel index on every OS (no CUDA wheels).

## Consequences
- The OneDrive caveats in the README only apply to people who clone into OneDrive on Windows.
- WSL2 resources are capped by `%UserProfile%\.wslconfig` (4 vCPU / 6 GB at the time of writing). `n_envs` for training is sized from the measured SPS benchmark, not assumed.
- Windows-only problems (e.g. `spawn` multiprocessing in `SubprocVecEnv`) are caught by the Windows CI job.
