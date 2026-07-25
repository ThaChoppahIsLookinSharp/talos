from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.architecture import abstract_accelerator_from_zigzag_yaml
from talos.ip import IPPool
from talos.level2 import run_level2_nsga2


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


def main() -> None:
    ip_pool = IPPool.from_yaml(IP_POOL_PATH)
    accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))

    result = run_level2_nsga2(
        accelerator=accelerator,
        ip_pool=ip_pool,
        objective_names=["area", "delay", "inv_throughput"],
        pop_size=6,
        n_gen=2,
        seed=1,
    )

    print("Level 2 NSGA-II run finished.")
    print(f"solutions={len(result.solutions)}")
    print(f"csv_path={result.csv_path}")

    for solution in result.solutions[:3]:
        print(
            "solution "
            f"{solution['solution_index']}: "
            f"valid={solution['valid']} "
            f"area={solution['area']} "
            f"delay={solution['delay']} "
            f"throughput={solution['throughput']} "
            f"selected_ips={solution['selected_ips']}"
        )


if __name__ == "__main__":
    main()
