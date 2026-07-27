from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from talos.architecture import abstract_accelerator_from_zigzag_yaml
from talos.architecture.abstract_accelerator import AbstractAccelerator, AbstractComponent
from talos.architecture.genome import decode_genome
from talos.architecture.level1_importer import abstract_accelerator_from_level1_config
from talos.constraints import UserConstraints
from talos.evaluation.workload_activity import LayerActivity, WorkloadActivityProfile
from talos.ip import IPBlock, IPPool, PowerCharacterization
from talos.level2 import (
    DEFAULT_LEVEL2_OBJECTIVES,
    Level2ExhaustiveRunResult,
    Level2NSGA2RunResult,
    run_level2,
    run_level2_exhaustive,
    run_level2_nsga2,
)
from talos.level2.problem import Level2PymooProblem


REPO_ROOT = Path(__file__).resolve().parents[1]
IP_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_example.yaml"
SYNTHETIC_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_synthetic_28nm.yaml"
ZIGZAG_YAML_PATH = REPO_ROOT / "configs" / "zigzag_accelerator_example.yaml"


def activity_profile() -> WorkloadActivityProfile:
    return WorkloadActivityProfile(
        layers=(
            LayerActivity(
                layer_id="test_layer",
                latency_cycles=1000,
                mac_count=1000,
                spatially_used_pes=1,
                memory_accesses={
                    "rf_i1": 100,
                    "rf_i2": 100,
                    "rf_o": 100,
                    "gb": 100,
                },
            ),
        )
    )


def power_model(
    *,
    frequency: float = 500.0,
    voltage: float = 1.0,
) -> PowerCharacterization:
    return PowerCharacterization(
        source="synthetic",
        activity_method="vectorless",
        reference_frequency_mhz=frequency,
        p_idle_w=0.1,
        p_active_w=0.2,
        voltage_v=voltage,
        temperature_c=25.0,
        corner="tt",
    )


def dram_ip(
    *,
    frequency: float = 500.0,
    voltage: float = 1.0,
) -> IPBlock:
    return IPBlock(
        id="dram",
        type="dram",
        area=0,
        throughput=1,
        delay=20,
        fmax_mhz=frequency,
        bandwidth_bits=512,
        metadata={"accesses_per_cycle": 1},
        power_model=power_model(frequency=frequency, voltage=voltage),
    )


class Level2ExhaustiveRunnerTests(unittest.TestCase):
    def test_default_level2_objectives(self) -> None:
        self.assertEqual(
            DEFAULT_LEVEL2_OBJECTIVES,
            ["area", "energy", "workload_latency_s"],
        )

    def test_exhaustive_runner_finds_known_strict_frequency_solution(self) -> None:
        ip_pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(
            decode_genome([1, 2, 0, 2, 1, 0, 1])
        )

        result = run_level2_exhaustive(
            accelerator=accelerator,
            ip_pool=ip_pool,
            constraints=UserConstraints(
                max_area_mm2=0.4,
                max_power_w=0.12,
                min_frequency_mhz=800.0,
            ),
            activity_profile=activity_profile(),
            save_csv=False,
        )

        self.assertIsInstance(result, Level2ExhaustiveRunResult)
        self.assertIsNone(result.csv_path)
        self.assertGreater(result.explored_combinations, 0)
        self.assertGreater(len(result.solutions), 0)
        solution = result.solutions[0]
        self.assertIsNotNone(solution["power"])
        self.assertIsNotNone(solution["workload_energy_j"])
        self.assertIsNotNone(solution["workload_latency_s"])
        self.assertEqual(solution["reference_frequency_mhz"], 500.0)
        self.assertEqual(
            solution["objective_names"],
            ["area", "energy", "workload_latency_s"],
        )
        self.assertEqual(
            solution["objective_values"][1],
            solution["workload_energy_j"],
        )
        self.assertEqual(solution["dram_accesses"], 0)
        self.assertGreater(solution["dram_energy_j"], 0)
        self.assertTrue(
            all(
                row["area"] <= 0.4
                and row["power"] <= 0.12
                and row["physical_fmax_mhz"] >= 800.0
                for row in result.solutions
            )
        )
        self.assertEqual(solution["workload_cycles_per_inference"], 1000)
        self.assertEqual(solution["workload_latency_s"], 2e-6)
        self.assertEqual(solution["workload_throughput_ips"], 500_000)
        self.assertEqual(
            solution["timing_margin_mhz"],
            solution["physical_fmax_mhz"] - solution["reference_frequency_mhz"],
        )
        self.assertEqual(
            {row["workload_latency_s"] for row in result.solutions},
            {2e-6},
        )
        self.assertEqual(solution["strategy"], "exhaustive")

    def test_exhaustive_runner_respects_constraints(self) -> None:
        ip_pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(
            decode_genome([1, 2, 0, 2, 1, 0, 1])
        )

        result = run_level2_exhaustive(
            accelerator=accelerator,
            ip_pool=ip_pool,
            constraints=UserConstraints(max_power_w=0.00001),
            activity_profile=activity_profile(),
            save_csv=False,
        )

        self.assertEqual(result.solutions, [])

    def test_exhaustive_runner_combination_cap_raises(self) -> None:
        ip_pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(
            decode_genome([1, 2, 0, 2, 1, 0, 1])
        )

        with self.assertRaisesRegex(ValueError, "above limit"):
            run_level2_exhaustive(
                accelerator=accelerator,
                ip_pool=ip_pool,
                activity_profile=activity_profile(),
                max_combinations=1,
                save_csv=False,
            )

    def test_level2_dispatcher_runs_exhaustive(self) -> None:
        ip_pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        accelerator = abstract_accelerator_from_level1_config(
            decode_genome([1, 2, 0, 2, 1, 0, 1])
        )

        result = run_level2(
            accelerator=accelerator,
            ip_pool=ip_pool,
            strategy="exhaustive",
            activity_profile=activity_profile(),
            save_csv=False,
        )

        self.assertIsInstance(result, Level2ExhaustiveRunResult)

    def test_power_aware_preflight_validation(self) -> None:
        accelerator = AbstractAccelerator(
            name="a",
            components=[AbstractComponent(name="pe_array", type="pe")],
        )
        uncharacterized_pool = IPPool(
            [
                IPBlock(id="pe", type="pe", area=1, throughput=1, delay=1),
                dram_ip(),
            ]
        )
        with self.assertRaisesRegex(ValueError, "activity profile is missing"):
            Level2PymooProblem(
                accelerator=accelerator,
                ip_pool=uncharacterized_pool,
                objective_names=["energy"],
            )
        with self.assertRaisesRegex(ValueError, "activity profile is missing"):
            Level2PymooProblem(
                accelerator=accelerator,
                ip_pool=uncharacterized_pool,
                objective_names=["area"],
                constraints=UserConstraints(max_power_w=1),
            )
        problem = Level2PymooProblem(
            accelerator=accelerator,
            ip_pool=uncharacterized_pool,
            objective_names=["power"],
            activity_profile=activity_profile(),
        )
        out: dict[str, object] = {}
        problem._evaluate([0], out)
        self.assertEqual(out["F"], [float("inf")])

        area_problem = Level2PymooProblem(
            accelerator=accelerator,
            ip_pool=uncharacterized_pool,
            objective_names=["area", "delay"],
        )
        self.assertIsNone(area_problem.activity_profile)

    def test_power_preflight_rejects_frequency_and_pvt_mismatch(self) -> None:
        accelerator = AbstractAccelerator(
            name="a",
            components=[
                AbstractComponent(name="pe_array", type="pe"),
                AbstractComponent(name="rf_i1", type="register_file"),
                AbstractComponent(name="rf_i2", type="register_file"),
                AbstractComponent(name="rf_o", type="register_file"),
                AbstractComponent(name="gb", type="global_buffer"),
            ],
        )

        def ip(
            ip_id: str,
            model: PowerCharacterization,
            *,
            fmax_mhz: float = 600,
        ) -> IPBlock:
            return IPBlock(
                id=ip_id,
                type="pe",
                area=1,
                throughput=1,
                delay=1,
                fmax_mhz=fmax_mhz,
                metadata={"macs_per_cycle": 1, "precision_bits": 8},
                power_model=model,
            )

        memory_ips = [
            IPBlock(
                id=f"{name}_ip",
                type=ip_type,
                area=1,
                throughput=1,
                delay=1,
                fmax_mhz=600,
                metadata={"accesses_per_cycle": 1},
                power_model=power_model(),
            )
            for name, ip_type in (
                ("rf_i1", "register_file"),
                ("rf_i2", "register_file"),
                ("rf_o", "register_file"),
                ("gb", "global_buffer"),
            )
        ]
        frequency_problem = Level2PymooProblem(
            accelerator=accelerator,
            ip_pool=IPPool(
                [
                    ip("compatible", power_model()),
                    ip("wrong_reference", power_model(frequency=400)),
                    *memory_ips,
                    dram_ip(),
                ]
            ),
            objective_names=["power"],
            activity_profile=activity_profile(),
        )
        valid_out: dict[str, object] = {}
        invalid_out: dict[str, object] = {}
        frequency_problem._evaluate([0, 0, 0, 0, 0], valid_out)
        frequency_problem._evaluate([1, 0, 0, 0, 0], invalid_out)
        self.assertNotEqual(valid_out["F"], [float("inf")])
        self.assertEqual(invalid_out["F"], [float("inf")])

        voltage_problem = Level2PymooProblem(
            accelerator=accelerator,
            ip_pool=IPPool(
                [
                    ip("compatible", power_model()),
                    ip("wrong_voltage", power_model(voltage=0.9)),
                    *memory_ips,
                    dram_ip(),
                ]
            ),
            objective_names=["power"],
            activity_profile=activity_profile(),
        )
        voltage_out: dict[str, object] = {}
        voltage_problem._evaluate([1, 0, 0, 0, 0], voltage_out)
        self.assertEqual(voltage_out["F"], [float("inf")])


@unittest.skipUnless(
    importlib.util.find_spec("pymoo") is not None,
    "pymoo is not installed",
)
class Level2RunnerTests(unittest.TestCase):
    def test_level2_nsga2_returns_no_invalid_solutions(self) -> None:
        ip_pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))

        result = run_level2_nsga2(
            accelerator=accelerator,
            ip_pool=ip_pool,
            pop_size=4,
            n_gen=1,
            constraints=UserConstraints(max_power_w=1e-12),
            activity_profile=activity_profile(),
            save_csv=False,
        )

        self.assertEqual(result.problem.n_ieq_constr, 1)
        self.assertEqual(result.solutions, [])

    def test_level2_nsga2_runner_returns_solutions(self) -> None:
        ip_pool = IPPool.from_yaml(IP_POOL_PATH)
        accelerator = abstract_accelerator_from_zigzag_yaml(str(ZIGZAG_YAML_PATH))

        result = run_level2_nsga2(
            accelerator=accelerator,
            ip_pool=ip_pool,
            pop_size=4,
            n_gen=1,
            activity_profile=activity_profile(),
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
                "workload_cycles_per_inference",
                "workload_latency_s",
                "workload_throughput_ips",
                "reference_frequency_mhz",
                "physical_critical_delay",
                "physical_fmax_mhz",
                "timing_margin_mhz",
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
