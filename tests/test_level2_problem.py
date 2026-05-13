from __future__ import annotations

import unittest
from pathlib import Path

from talos.level2.architecture.zigzag_yaml_importer import (
    abstract_accelerator_from_zigzag_yaml,
)
from talos.level2.ip import IPPool
from talos.level2.problem import Level2PymooProblem


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


class Level2ProblemTests(unittest.TestCase):
    def test_level2_problem_evaluates_default_genome(self) -> None:
        ip_pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))
        problem = Level2PymooProblem(
            accelerator=accelerator,
            ip_pool=ip_pool,
            objective_names=["area", "power", "delay", "inv_throughput"],
        )
        out: dict[str, object] = {}

        problem._evaluate(problem.spec.default_genome(), out)

        self.assertEqual(len(out["F"]), 4)
        self.assertTrue(all(float(value) > 0 for value in out["F"]))


if __name__ == "__main__":
    unittest.main()
