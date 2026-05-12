from __future__ import annotations

from dataclasses import dataclass
import math

from talos.architecture.abstract_accelerator import AbstractAccelerator, AbstractComponent
from talos.ip.ip_characterization import IPBlock
from talos.ip.ip_pool import IPPool


@dataclass(frozen=True)
class ImplementedComponent:
    abstract_component: AbstractComponent
    ip: IPBlock


@dataclass(frozen=True)
class ImplementedAccelerator:
    components: list[ImplementedComponent]


@dataclass(frozen=True)
class Level2GeneSpec:
    component: AbstractComponent
    candidates: list[IPBlock]

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def bounds(self) -> tuple[int, int]:
        return (0, len(self.candidates) - 1)


@dataclass(frozen=True)
class Level2GenomeSpec:
    genes: list[Level2GeneSpec]

    @classmethod
    def from_accelerator_and_pool(
        cls,
        accelerator: AbstractAccelerator,
        ip_pool: IPPool,
    ) -> "Level2GenomeSpec":
        genes: list[Level2GeneSpec] = []
        for component in accelerator.components:
            try:
                candidates = ip_pool.find_compatible(component)
            except ValueError as exc:
                raise ValueError(
                    "Unable to build Level2GenomeSpec for component "
                    f"name={component.name!r}, type={component.type!r}, "
                    f"required_capacity_bits={component.required_capacity_bits!r}, "
                    f"required_bandwidth_bits={component.required_bandwidth_bits!r}."
                ) from exc
            genes.append(Level2GeneSpec(component=component, candidates=candidates))
        return cls(genes=genes)

    def gene_names(self) -> list[str]:
        return [gene.name for gene in self.genes]

    def gene_bounds(self) -> list[tuple[int, int]]:
        return [gene.bounds for gene in self.genes]

    def default_genome(self) -> list[int]:
        return [0] * len(self.genes)

    def decode(self, genome: list[float]) -> ImplementedAccelerator:
        if len(genome) != len(self.genes):
            raise ValueError(f"Expected {len(self.genes)} Level-2 genes, got {len(genome)}.")

        implemented: list[ImplementedComponent] = []
        for raw_gene, spec in zip(genome, self.genes, strict=True):
            try:
                value = float(raw_gene)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Level-2 gene for component {spec.name!r} must be numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Level-2 gene for component {spec.name!r} must be finite.")
            lower, upper = spec.bounds
            index = int(round(value))
            index = max(lower, min(index, upper))
            implemented.append(
                ImplementedComponent(
                    abstract_component=spec.component,
                    ip=spec.candidates[index],
                )
            )
        return ImplementedAccelerator(components=implemented)
