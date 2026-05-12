from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from talos.architecture import abstract_accelerator_from_zigzag_yaml
from talos.ip import IPPool
from talos.level2 import Level2NSGA2RunResult, run_level2_nsga2


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


@unittest.skipUnless(
    importlib.util.find_spec("pymoo") is not None,
    "pymoo is not installed",
)
class Level2RunnerTests(unittest.TestCase):
    def test_level2_nsga2_runner_returns_solutions(self) -> None:
        ip_pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))

        result = run_level2_nsga2(
            accelerator=accelerator,
            ip_pool=ip_pool,
            pop_size=4,
            n_gen=1,
            save_csv=False,
        )

        self.assertIsInstance(result, Level2NSGA2RunResult)
        self.assertIsNotNone(result.pymoo_result)
        self.assertIsNotNone(result.problem)
        self.assertIsInstance(result.solutions, list)
        self.assertIsNone(result.csv_path)

        if result.solutions:
            solution = result.solutions[0]
            expected_keys = {
                "solution_index",
                "genome",
                "selected_ips",
                "area",
                "power",
                "delay",
                "throughput",
                "valid",
                "objective_names",
                "objective_values",
                "pop_size",
                "n_gen",
                "seed",
            }
            self.assertTrue(expected_keys.issubset(solution))
            self.assertEqual(solution["pop_size"], 4)
            self.assertEqual(solution["n_gen"], 1)


if __name__ == "__main__":
    unittest.main()
