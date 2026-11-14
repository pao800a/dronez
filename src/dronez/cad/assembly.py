"""Assembly: the placed parts (frame C) and the part -> material map."""

from __future__ import annotations

from dataclasses import dataclass

from build123d import Compound

from dronez.cad import materials, parts
from dronez.cad.parts import AnyShape
from dronez.cad.schema import DroneConfig


@dataclass(frozen=True)
class Part:
    """One logical part: ``count`` placed solids sharing a material (all in the CAD frame C)."""

    name: str
    material: str
    density_kg_m3: float
    solids: tuple[AnyShape, ...]

    @property
    def count(self) -> int:
        return len(self.solids)

    def compound(self) -> Compound:
        return Compound(children=list(self.solids), label=self.name)


@dataclass(frozen=True)
class Assembly:
    parts: dict[str, Part]

    @property
    def materials(self) -> dict[str, str]:
        """Part name -> material key."""
        return {n: p.material for n, p in self.parts.items()}

    def compound(self) -> Compound:
        return Compound(children=[p.compound() for p in self.parts.values()], label="assembly")


def build_assembly(cfg: DroneConfig) -> Assembly:
    builders = parts.part_builders(cfg)
    materials.check_coverage(cfg, list(builders))
    out: dict[str, Part] = {}
    for name, build in builders.items():
        solids = tuple(build(cfg))
        for i, s in enumerate(solids):
            if not s.is_valid or len(s.solids()) != 1 or s.volume <= 0.0:
                raise ValueError(f"part {name!r} instance {i} is not a single valid solid")
        mat = materials.part_material(cfg, name)
        out[name] = Part(name, mat, materials.density_kg_m3(cfg, mat), solids)
    return Assembly(out)
