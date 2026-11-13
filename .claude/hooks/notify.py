"""Notification: Windows balloon/toast when Claude needs input.

Works on Windows natively and from WSL (via ``powershell.exe``). Degrades
silently everywhere else, in CI, or when ``DRONEZ_NOTIFY_DISABLE=1``.
The PowerShell process is detached so the hook returns immediately.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import read_input

PS_TEMPLATE = (
    "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
    "$n = New-Object System.Windows.Forms.NotifyIcon; "
    "$n.Icon = [System.Drawing.SystemIcons]::Information; $n.Visible = $true; "
    "$n.ShowBalloonTip(6000, '{title}', '{message}', "
    "[System.Windows.Forms.ToolTipIcon]::Info); Start-Sleep -Seconds 7; $n.Dispose()"
)


def _ps_quote(text: str) -> str:
    return text.replace("'", "''").replace("\n", " ")[:200]


def powershell() -> str | None:
    if os.environ.get("CI") or os.environ.get("DRONEZ_NOTIFY_DISABLE") == "1":
        return None
    for name in ("powershell.exe", "powershell", "pwsh.exe"):
        path = shutil.which(name)
        if path and (sys.platform == "win32" or name.endswith(".exe")):
            return path
    return None


def main() -> None:
    data = read_input()
    exe = powershell()
    if exe is None:
        return
    title = _ps_quote(f"Claude Code · Dronez · {data.get('title') or 'needs input'}")
    message = _ps_quote(str(data.get("message") or "Claude needs your input"))
    script = PS_TEMPLATE.format(title=title, message=message)
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with contextlib.suppress(OSError):
        subprocess.Popen([exe, "-NoProfile", "-NonInteractive", "-Command", script], **kwargs)  # type: ignore[call-overload]


if __name__ == "__main__":
    main()
