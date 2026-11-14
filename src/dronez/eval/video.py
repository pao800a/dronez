"""MP4 rendering of a recorded trajectory with ``mujoco.Renderer`` (replay of ``qpos``).

``MUJOCO_GL`` is read when ``mujoco`` is imported (MuJoCo 4), and the parent process has already
imported it, so rendering runs in a **child process** (``python -m dronez.eval.video``) whose
environment alone carries ``MUJOCO_GL``. Backends are tried in order (Linux: ``egl`` then
``osmesa``; Windows/macOS: the platform default). If none works, ``render_video`` warns and returns
``None``: eval never fails because there is no GL. Nothing here ever sets ``MUJOCO_GL`` in the
calling process.

This module must stay importable without ``mujoco``: the child sets the variable first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gl_backends() -> list[str | None]:
    """Backends to try, in order (``None`` = leave ``MUJOCO_GL`` unset, platform default)."""
    return ["egl", "osmesa"] if sys.platform.startswith("linux") else [None]


def render_video(
    qpos: NDArray[np.float64],
    out_path: Path,
    *,
    drone: str = "default_5in",
    sim: str = "default",
    fps: int = 25,
    size: tuple[int, int] = (640, 480),
    lookat: tuple[float, float, float] | None = None,
    timeout_s: float = 180.0,
) -> Path | None:
    """Replay ``qpos`` (``(n, 7)``, free-joint pose per frame) into an MP4 at ``out_path``.

    Returns the path, or ``None`` (with a ``RuntimeWarning``) if no GL backend could render.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "qpos.npz"
        np.savez(npz, qpos=np.asarray(qpos, dtype=np.float64))
        for gl in gl_backends():
            env = dict(os.environ)
            env.pop("MUJOCO_GL", None)
            if gl is not None:
                env["MUJOCO_GL"] = gl
                if gl == "egl":
                    env.setdefault("PYOPENGL_PLATFORM", "egl")
                if gl == "osmesa":
                    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
            cmd = [
                sys.executable,
                "-m",
                "dronez.eval.video",
                "--qpos",
                str(npz),
                "--out",
                str(out_path),
                "--drone",
                drone,
                "--sim",
                sim,
                "--fps",
                str(fps),
                "--size",
                f"{size[0]}x{size[1]}",
            ]
            if lookat is not None:
                cmd += ["--lookat", ",".join(f"{v:.4f}" for v in lookat)]
            try:
                res = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                errors.append(f"{gl or 'default'}: {exc}")
                continue
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return out_path
            tail = (res.stderr.strip().splitlines() or ["no stderr"])[-1]
            errors.append(f"{gl or 'default'}: {tail}")
    warnings.warn(
        "no video written (no usable OpenGL backend): " + "; ".join(errors),
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _replay(args: argparse.Namespace) -> None:
    """Child process: ``MUJOCO_GL`` is already in the environment before ``mujoco`` is imported."""
    import imageio.v2 as imageio  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    from dronez.cad.schema import load_drone_config  # noqa: PLC0415
    from dronez.sim import build_model, load_sim_config  # noqa: PLC0415

    qpos = np.load(args.qpos)["qpos"]
    cfg = load_drone_config(args.drone)
    model, _ = build_model(cfg, load_sim_config(args.sim))
    width, height = (int(v) for v in args.size.split("x"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.8
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 0.9
    lookat = (
        np.array([float(v) for v in args.lookat.split(",")])
        if args.lookat
        else np.array([0.0, 0.0, float(qpos[0, 2])])
    )
    cam.lookat[:] = lookat
    with (
        mujoco.Renderer(model, height=height, width=width) as renderer,
        imageio.get_writer(args.out, fps=args.fps, codec="libx264", macro_block_size=1) as writer,
    ):
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dronez.eval.video")
    p.add_argument("--qpos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drone", default="default_5in")
    p.add_argument("--sim", default="default")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", default="640x480")
    p.add_argument("--lookat", default="")
    _replay(p.parse_args(argv))


if __name__ == "__main__":
    main()
