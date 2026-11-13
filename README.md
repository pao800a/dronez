# Dronez

A parametric quadcopter designed in code-CAD (**build123d**), simulated in **MuJoCo** with a realistic motor/propeller model, and stabilized by a reinforcement-learning policy (**Stable-Baselines3 PPO**). The policy is hardened for sim-to-real and exported to **ONNX** with a hardware-agnostic deployment contract.

> Status: early development. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Quick start (WSL2 / Linux, primary)

```bash
git clone https://github.com/pao800a/dronez.git ~/projects/dronez   # Linux filesystem, not /mnt/c
cd ~/projects/dronez
./scripts/setup_wsl.sh        # installs uv if needed, then `uv sync`
uv run pytest -m fast
```

## Quick start (Windows native)

```powershell
git clone https://github.com/pao800a/dronez.git
cd dronez
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
uv run pytest -m fast
```

**OneDrive:** if the clone sits in a OneDrive-synced folder, the setup script redirects the virtualenv (`UV_PROJECT_ENVIRONMENT`) and all heavy artifacts (`DRONEZ_DATA_DIR`) to `%LOCALAPPDATA%\Dronez\`, so OneDrive never syncs GBs of venv, checkpoints or videos.

## Data directory

Generated models, training runs, checkpoints, eval reports and videos live in `DRONEZ_DATA_DIR`: by default `~/.local/share/dronez` on Linux and `%LOCALAPPDATA%\Dronez\data` on Windows. Only small release artifacts are committed, under `releases/`.

## Docs

- [Conventions](docs/conventions.md): units, frames, quaternions, motor numbering
- [Architecture](docs/architecture.md) · [ADRs](docs/adr/) · [Progress](docs/PROGRESS.md)
- [Build spec](docs/BUILD_PROMPT.md)

## Development

This repo ships its own Claude Code tooling (`CLAUDE.md`, `.claude/agents`, `.claude/skills`, `.claude/hooks`, `.mcp.json`). Use `uv add` to add dependencies, never `pip install`.

## License

MIT
