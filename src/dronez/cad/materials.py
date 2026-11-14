"""Part -> material map and densities, resolved from the validated drone config."""

from __future__ import annotations

from dronez.cad.schema import DroneConfig


def part_material(cfg: DroneConfig, part: str) -> str:
    """Material key of a part (raises ``KeyError`` with a clear message when missing)."""
    try:
        return cfg.parts_material[part]
    except KeyError:
        raise KeyError(f"parts_material has no entry for part {part!r}") from None


def density_kg_m3(cfg: DroneConfig, material: str) -> float:
    return cfg.materials[material].density_kg_m3


def check_coverage(cfg: DroneConfig, part_names: list[str]) -> None:
    """``parts_material`` must cover exactly the parts the CAD builds."""
    missing = sorted(set(part_names) - set(cfg.parts_material))
    extra = sorted(set(cfg.parts_material) - set(part_names))
    if missing or extra:
        raise ValueError(
            f"parts_material mismatch: parts without a material {missing}, "
            f"materials for unknown parts {extra}"
        )
