from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
import math

from talos.architecture.abstract_accelerator import AbstractAccelerator, AbstractComponent
from talos.ip.ip_characterization import IPBlock
from talos.ip.ip_pool import IPPool


@dataclass(frozen=True)
class ImplementedComponent:
    abstract_component: AbstractComponent
    ip: IPBlock
    covered_by_pe_id: str | None = None


@dataclass(frozen=True)
class ImplementedAccelerator:
    components: list[ImplementedComponent]


def physical_components(
    components: Sequence[ImplementedComponent],
) -> list[ImplementedComponent]:
    by_name = {component.abstract_component.name: component for component in components}
    if len(by_name) != len(components):
        raise ValueError("Implemented component names must be unique.")

    pe_by_id = {
        component.ip.id: component
        for component in components
        if component.abstract_component.type == "pe"
    }
    for pe_id, pe in pe_by_id.items():
        for role, rf_id in pe.ip.included_rfs.items():
            rf = by_name.get(role)
            if (
                rf is None
                or rf.abstract_component.type != "register_file"
                or rf.ip.id != rf_id
                or rf.covered_by_pe_id != pe_id
                or rf.abstract_component.count != pe.abstract_component.count
            ):
                raise ValueError(
                    f"PE {pe_id!r} does not validly cover {role!r}:{rf_id!r}."
                )

    for component in components:
        if component.covered_by_pe_id is None:
            continue
        pe = pe_by_id.get(component.covered_by_pe_id)
        if (
            pe is None
            or pe.ip.included_rfs.get(component.abstract_component.name)
            != component.ip.id
        ):
            raise ValueError(
                f"Component {component.abstract_component.name!r} is not covered "
                f"by selected PE {component.covered_by_pe_id!r}."
            )

    return [
        component for component in components if component.covered_by_pe_id is None
    ]


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

        genes_by_name = {gene.name: gene for gene in genes}
        filtered_genes: list[Level2GeneSpec] = []
        for gene in genes:
            if gene.component.type != "pe":
                filtered_genes.append(gene)
                continue
            candidates = [
                candidate
                for candidate in gene.candidates
                if all(
                    role in genes_by_name
                    and genes_by_name[role].component.type == "register_file"
                    and genes_by_name[role].component.count == gene.component.count
                    and any(
                        rf.id == referenced_id
                        for rf in genes_by_name[role].candidates
                    )
                    for role, referenced_id in candidate.included_rfs.items()
                )
            ]
            if not candidates:
                raise ValueError(
                    "Unable to build Level2GenomeSpec for component "
                    f"name={gene.name!r}: no PE candidate has compatible included RFs."
                )
            filtered_genes.append(
                Level2GeneSpec(component=gene.component, candidates=candidates)
            )
        pe_genes = [
            gene for gene in filtered_genes if gene.component.type == "pe"
        ]
        if len(pe_genes) != 1 and any(
            candidate.included_rfs
            for gene in pe_genes
            for candidate in gene.candidates
        ):
            raise ValueError(
                "Composite PE selection requires exactly one PE component."
            )
        return cls(genes=filtered_genes)

    def gene_names(self) -> list[str]:
        return [gene.name for gene in self.genes]

    def gene_bounds(self) -> list[tuple[int, int]]:
        return [gene.bounds for gene in self.genes]

    def default_genome(self) -> list[int]:
        return self.canonicalize([0] * len(self.genes))

    def canonicalize(self, genome: Sequence[float]) -> list[int]:
        indices = self._normalized_indices(genome)
        self._apply_included_rf_coverage(indices)
        return indices

    def iter_genomes(self) -> Iterator[tuple[int, ...]]:
        for domains in self._conditional_domains():
            yield from product(*domains)

    def genome_count(self) -> int:
        return sum(
            math.prod(len(domain) for domain in domains)
            for domains in self._conditional_domains()
        )

    def decode(self, genome: Sequence[float]) -> ImplementedAccelerator:
        indices = self._normalized_indices(genome)
        covered_by_pe = self._apply_included_rf_coverage(indices)
        implemented = [
            ImplementedComponent(
                abstract_component=spec.component,
                ip=spec.candidates[index],
                covered_by_pe_id=covered_by_pe.get(gene_index),
            )
            for gene_index, (index, spec) in enumerate(
                zip(indices, self.genes, strict=True)
            )
        ]
        return ImplementedAccelerator(components=implemented)

    def _normalized_indices(self, genome: Sequence[float]) -> list[int]:
        if len(genome) != len(self.genes):
            raise ValueError(f"Expected {len(self.genes)} Level-2 genes, got {len(genome)}.")

        indices: list[int] = []
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
            indices.append(index)
        return indices

    def _apply_included_rf_coverage(self, indices: list[int]) -> dict[int, str]:
        gene_index_by_name = {
            spec.name: index for index, spec in enumerate(self.genes)
        }
        covered_by_pe: dict[int, str] = {}
        for pe_gene_index, pe_spec in enumerate(self.genes):
            if pe_spec.component.type != "pe":
                continue
            pe = pe_spec.candidates[indices[pe_gene_index]]
            for role, referenced_id in pe.included_rfs.items():
                rf_gene_index = gene_index_by_name[role]
                if rf_gene_index in covered_by_pe:
                    raise ValueError(
                        f"RF role {role!r} is covered by more than one selected PE."
                    )
                rf_spec = self.genes[rf_gene_index]
                indices[rf_gene_index] = next(
                    index
                    for index, candidate in enumerate(rf_spec.candidates)
                    if candidate.id == referenced_id
                )
                covered_by_pe[rf_gene_index] = pe.id
        return covered_by_pe

    def _conditional_domains(self) -> Iterator[tuple[tuple[int, ...], ...]]:
        pe_gene_indices = [
            index
            for index, spec in enumerate(self.genes)
            if spec.component.type == "pe"
        ]
        pe_domains = [
            range(len(self.genes[index].candidates))
            for index in pe_gene_indices
        ]
        gene_index_by_name = {
            spec.name: index for index, spec in enumerate(self.genes)
        }

        for pe_values in product(*pe_domains):
            fixed = dict(zip(pe_gene_indices, pe_values, strict=True))
            valid = True
            for pe_gene_index, pe_value in zip(
                pe_gene_indices, pe_values, strict=True
            ):
                pe = self.genes[pe_gene_index].candidates[pe_value]
                for role, referenced_id in pe.included_rfs.items():
                    rf_gene_index = gene_index_by_name[role]
                    rf_index = next(
                        index
                        for index, candidate in enumerate(
                            self.genes[rf_gene_index].candidates
                        )
                        if candidate.id == referenced_id
                    )
                    if rf_gene_index in fixed:
                        valid = False
                        break
                    fixed[rf_gene_index] = rf_index
                if not valid:
                    break
            if valid:
                yield tuple(
                    (fixed[index],)
                    if index in fixed
                    else tuple(range(len(spec.candidates)))
                    for index, spec in enumerate(self.genes)
                )
