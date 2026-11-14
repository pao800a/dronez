"""Top / side / iso PNG renders from tessellated meshes (matplotlib, Agg; no GL needed)."""

from __future__ import annotations

import colorsys
import zlib
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

Mesh = tuple[F64, I64]  # vertices (n, 3) in metres, triangles (m, 3)

#: name -> (elevation, azimuth) in matplotlib's convention; FLU: x forward, y left, z up.
VIEWS: dict[str, tuple[float, float]] = {
    "top": (90.0, 180.0),
    "side": (0.0, -90.0),
    "iso": (24.0, -55.0),
}
_LIGHT = np.array([-0.4, -0.5, 0.8])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_PART_COLORS: dict[str, str] = {
    "bottom_plate": "#2b2b2f",
    "top_plate": "#2b2b2f",
    "arms": "#3a3a40",
    "props": "#8fa7c4",
    "battery": "#d9a21b",
    "battery_strap": "#555555",
    "camera": "#1f7a3a",
    "camera_mount": "#c0392b",
    "standoffs": "#b0b4b8",
}


def _color(name: str) -> tuple[float, float, float]:
    if name in _PART_COLORS:
        h = _PART_COLORS[name].lstrip("#")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    hue = (zlib.crc32(name.encode()) % 360) / 360.0
    return colorsys.hsv_to_rgb(hue, 0.35, 0.8)


def _shaded_triangles(
    meshes: dict[str, Mesh],
) -> tuple[F64, list[tuple[float, float, float, float]]]:
    tris: list[F64] = []
    colors: list[tuple[float, float, float, float]] = []
    for name, (v, f) in meshes.items():
        t = v[f]  # (m, 3, 3)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
        shade = 0.45 + 0.55 * np.abs(n @ _LIGHT)
        base = np.array(_color(name))
        tris.append(t)
        colors.extend((*(base * s), 1.0) for s in shade)
    return np.concatenate(tris), colors


def render_views(
    meshes: dict[str, Mesh],
    out_dir: Path,
    com_c_m: tuple[float, float, float],
    hubs_c_m: list[tuple[float, float, float]],
    title: str,
) -> dict[str, Path]:
    """Write ``render_<view>.png`` into ``out_dir``; returns view -> path."""
    tris, colors = _shaded_triangles(meshes)
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    centre, half = (lo + hi) / 2.0, float(np.max(hi - lo)) / 2.0 * 0.62
    paths: dict[str, Path] = {}
    for view, (elev, azim) in VIEWS.items():
        fig = Figure(figsize=(7.0, 7.0), dpi=140)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        ax.set_proj_type("ortho")
        coll = Poly3DCollection(tris, facecolors=colors, edgecolors="none", linewidths=0.0)
        ax.add_collection3d(coll)
        ax.scatter([com_c_m[0]], [com_c_m[1]], [com_c_m[2]], c="red", s=40, marker="o", zorder=10)
        if view == "top":
            for i, (x, y, z) in enumerate(hubs_c_m):
                ax.text(x, y, z + 0.03, f"M{i}", ha="center", va="center", fontsize=11, zorder=11)
            ax.text(0.14, 0.0, 0.0, "FRONT (+x)", ha="center", fontsize=9, zorder=11)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.suptitle(f"{title} - {view} (red dot = CoM)", fontsize=10)
        path = out_dir / f"render_{view}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        paths[view] = path
    return paths
