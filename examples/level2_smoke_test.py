from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.level1.genome import decode_genome, default_genome
from talos.level2.architecture.level1_importer import (
    abstract_accelerator_from_level1_config,
)
from talos.level2.architecture.zigzag_yaml_importer import (
    abstract_accelerator_from_zigzag_yaml,
)
from talos.level2.ip import IPPool
from talos.level2 import Level2Evaluator, Level2GenomeSpec, Level2PymooProblem


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


def run_level2_flow(label: str, accelerator) -> None:
    pool = IPPool.from_yaml(IP_POOL_PATH)
    spec = Level2GenomeSpec.from_accelerator_and_pool(accelerator, pool)
    genome = spec.default_genome()
    implemented = spec.decode(genome)
    result = Level2Evaluator().evaluate(implemented)
    problem = Level2PymooProblem(
        accelerator=accelerator,
        ip_pool=pool,
        objective_names=["area", "power", "delay", "inv_throughput"],
    )
    problem_out: dict[str, object] = {}
    problem._evaluate(problem.spec.default_genome(), problem_out)

    print(f"=== {label} ===")
    print(f"accelerator={accelerator.name}")
    print(f"components={[component.name for component in accelerator.components]}")
    print(f"gene_names={spec.gene_names()}")
    print(f"gene_bounds={spec.gene_bounds()}")
    print(f"default_genome={genome}")
    print(f"valid={result.valid}")
    print(f"area={result.area}")
    print(f"power={result.power}")
    print(f"delay={result.delay}")
    print(f"throughput={result.throughput}")
    print(f"error_message={result.error_message}")
    print(f"problem_objectives={problem_out['F']}")
    print()


def main() -> None:
    level1_config = decode_genome(default_genome())
    from_level1 = abstract_accelerator_from_level1_config(level1_config)
    run_level2_flow("from_level1", from_level1)

    from_yaml = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))
    run_level2_flow("from_zigzag_yaml", from_yaml)


if __name__ == "__main__":
    main()
