# Dronez setup for Windows native (PowerShell). See docs/adr/0001-wsl2-primary-platform.md.
#
# The virtualenv and all heavy artifacts are kept OUTSIDE the repo so that a
# checkout inside a OneDrive-synced folder doesn't sync GBs of venv/checkpoints
# or hit file locks:
#   UV_PROJECT_ENVIRONMENT = %LOCALAPPDATA%\Dronez\.venv
#   DRONEZ_DATA_DIR        = %LOCALAPPDATA%\Dronez\data
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$venv = Join-Path $env:LOCALAPPDATA "Dronez\.venv"
$data = Join-Path $env:LOCALAPPDATA "Dronez\data"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# Persist for future shells (user scope) and set for this one.
[Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $venv, "User")
[Environment]::SetEnvironmentVariable("DRONEZ_DATA_DIR", $data, "User")
$env:UV_PROJECT_ENVIRONMENT = $venv
$env:DRONEZ_DATA_DIR = $data

if ((Get-Location).Path -match "OneDrive") {
    Write-Warning "Repo is inside OneDrive. The venv and data dir are redirected to $env:LOCALAPPDATA\Dronez; consider pausing sync for this folder or cloning elsewhere."
}

uv sync
try { uv run pre-commit install } catch { Write-Warning "pre-commit install failed: $_" }

Write-Host ""
Write-Host "venv: $venv"
Write-Host "data: $data"
Write-Host "Next: uv run dronez-build-model; uv run pytest -m fast"
