from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pymoo.optimize import minimize

from examples.constraint_sweep import build_command, build_parser, sweep_cases
from examples.full_flow_example import (
    Level1Candidate,
    SUMMARY_FIELDNAMES,
    build_summary_rows,
    iter_level1_candidates,
    main as full_flow_main,
    parse_args,
    select_level1_candidates,
)
from examples.objective_sweep import (
    build_command as build_objective_sweep_command,
    build_parser as build_objective_sweep_parser,
    objective_cases,
)
from talos.architecture.abstract_accelerator import AbstractAccelerator, AbstractComponent
from talos.architecture.genome import GENOME_LENGTH, GENOME_SPEC, decode_genome
from talos.architecture.level1_importer import abstract_accelerator_from_level1_config
from talos.constraints import (
    UserConstraints,
    estimated_fps,
    estimated_inferences_per_second,
)
from talos.evaluation.workload_activity import LayerActivity, WorkloadActivityProfile
from talos.evaluation.zigzag_evaluator import EvaluationResult
from talos.ga.pymoo_runner import TalosPymooProblem, _build_nsga2
from talos.ip import IPBlock, IPPool, PowerCharacterization
from talos.level2 import Level2Evaluator
from talos.level2.genome import ImplementedAccelerator, ImplementedComponent
from talos.level2.problem import Level2PymooProblem
from talos.level2.runner import _build_solution_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_POOL_PATH = REPO_ROOT / "configs" / "ip_pool_synthetic_28nm.yaml"


class FakeAdapter:
    def __init__(self, result: EvaluationResult) -> None:
        self.result = result

    def build_objectives(self, names: list[str]):
        return [lambda _genome, name=name: self._value(name) for name in names]

    def evaluate(self, _genome: list[float]) -> EvaluationResult:
        return self.result

    def _value(self, name: str) -> float:
        if name == "latency":
            return self.result.latency
        if name == "energy":
            return self.result.energy
        if name == "area":
            return self.result.area
        raise ValueError(name)


def implemented_ip(
    *,
    area: float = 1.0,
    fmax_mhz: float | None = 500.0,
    count: int = 1,
    metadata: dict[str, float] | None = None,
) -> ImplementedAccelerator:
    component = AbstractComponent(name="pe", type="pe", count=count)
    ip = IPBlock(
        id="pe0",
        type="pe",
        area=area,
        throughput=1.0,
        delay=1.0,
        fmax_mhz=fmax_mhz,
        metadata=metadata,
    )
    return ImplementedAccelerator(
        components=[ImplementedComponent(abstract_component=component, ip=ip)]
    )


class UserConstraintsTests(unittest.TestCase):
    def test_full_flow_can_fill_pareto_set_from_feasible_final_population(self) -> None:
        population = SimpleNamespace(
            get=lambda name: {
                "X": np.array([[0.0] * GENOME_LENGTH, [1.0] * GENOME_LENGTH]),
                "F": np.array([[2.0], [3.0]]),
                "feasible": np.array([[True], [False]]),
            }[name]
        )
        genomes, objectives = iter_level1_candidates(
            SimpleNamespace(
                X=np.array([[2.0] * GENOME_LENGTH]),
                F=np.array([[1.0]]),
                pop=population,
            )
        )

        self.assertEqual(genomes, [[2.0] * GENOME_LENGTH, [0.0] * GENOME_LENGTH])
        self.assertEqual(objectives, [[1.0], [2.0]])

    def test_user_constraints_validate_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_area_mm2"):
            UserConstraints(max_area_mm2=0)

        with self.assertRaisesRegex(ValueError, "min_frequency_mhz"):
            UserConstraints(min_frequency_mhz=-1)

    def test_estimated_fps_supports_runtime_and_legacy_fmax_inputs(self) -> None:
        self.assertEqual(
            estimated_inferences_per_second(workload_latency_s=2e-3),
            500.0,
        )
        self.assertEqual(estimated_fps(workload_latency_s=2e-3), 500.0)
        self.assertEqual(
            estimated_fps(
                latency_cycles=1000,
                implementation_fmax_mhz=500,
            ),
            500_000.0,
        )

    def test_level1_latency_constraint_is_exported_to_pymoo(self) -> None:
        problem = TalosPymooProblem(
            workload_path="unused.onnx",
            objective_names=["latency"],
            adapter=FakeAdapter(
                EvaluationResult(latency=10.0, energy=1.0, area=1.0, valid=True)
            ),
            constraints=UserConstraints(max_latency_cycles=5.0),
        )
        out: dict[str, list[float]] = {}

        problem._evaluate(np.zeros(GENOME_LENGTH), out)

        self.assertEqual(out["G"], [5.0])

    def test_level1_nsga2_keeps_integer_unique_genomes(self) -> None:
        problem = TalosPymooProblem(
            workload_path="unused.onnx",
            objective_names=["energy"],
            adapter=FakeAdapter(
                EvaluationResult(latency=1.0, energy=1.0, area=1.0, valid=True)
            ),
        )

        result = minimize(problem, _build_nsga2(16), ("n_gen", 3), seed=1)
        genomes = result.pop.get("X")

        np.testing.assert_array_equal(genomes, np.rint(genomes))
        self.assertEqual(len(genomes), len(np.unique(genomes, axis=0)))

    def test_level2_rejects_area_power_and_frequency_violations(self) -> None:
        self.assertFalse(
            Level2Evaluator(UserConstraints(max_area_mm2=0.5))
            .evaluate(implemented_ip(area=1.0))
            .valid
        )
        self.assertFalse(
            Level2Evaluator(UserConstraints(max_power_w=0.5))
            .evaluate(implemented_ip())
            .valid
        )
        self.assertFalse(
            Level2Evaluator(UserConstraints(min_frequency_mhz=600.0))
            .evaluate(implemented_ip(fmax_mhz=500.0))
            .valid
        )

    def test_level2_rejects_missing_fmax_when_frequency_is_requested(self) -> None:
        result = Level2Evaluator(UserConstraints(min_frequency_mhz=500.0)).evaluate(
            implemented_ip(fmax_mhz=None)
        )

        self.assertFalse(result.valid)
        self.assertIn("unavailable", result.error_message or "")

    def test_level2_constraints_are_exported_to_pymoo(self) -> None:
        component = AbstractComponent(name="pe", type="pe")
        problem = Level2PymooProblem(
            accelerator=AbstractAccelerator(name="a", components=[component]),
            ip_pool=IPPool(
                [
                    IPBlock(
                        id="pe0",
                        type="pe",
                        area=1.0,
                        throughput=1.0,
                        delay=1.0,
                        fmax_mhz=500.0,
                    )
                ]
            ),
            objective_names=["area"],
            constraints=UserConstraints(
                max_area_mm2=0.5,
                min_frequency_mhz=600.0,
            ),
        )
        out: dict[str, list[float]] = {}

        problem._evaluate(np.zeros(1), out)

        self.assertEqual(problem.n_ieq_constr, 2)
        self.assertEqual(out["F"], [1.0])
        self.assertEqual(out["G"], [0.5, 100.0])

    def test_full_flow_summary_reports_constraints_and_inference_rate(self) -> None:
        rows = build_summary_rows(
            architecture_index=0,
            level1_raw_genome=[0.0] * GENOME_LENGTH,
            level1_discrete_genome=[0] * GENOME_LENGTH,
            level1_architecture_config={},
            level1_objective_values=[10.0, 1.0, 2.0],
            level1_objective_names=["latency", "energy", "area"],
            level2_objective_names=["area", "energy", "workload_latency_s"],
            level1_csv_path="level1.csv",
            level2_csv_path=None,
            level2_solutions=[
                {
                    "solution_index": 0,
                    "valid": True,
                    "physical_fmax_mhz": 200.0,
                    "power": 0.1,
                    "workload_energy_j": 2e-6,
                    "workload_latency_s": 20e-6,
                    "reference_frequency_mhz": 100.0,
                    "workload_throughput_ips": 50_000.0,
                    "dram_accesses": 1000,
                    "dram_energy_j": 1e-6,
                    "covered_by_pe": {"rf_i1": "pe_tile"},
                    "constraint_violations": [],
                }
            ],
            constraints=UserConstraints(
                max_latency_cycles=20.0,
                min_frequency_mhz=100.0,
            ),
        )

        self.assertIn("constraints_satisfied", SUMMARY_FIELDNAMES)
        self.assertIn("workload_energy_j", SUMMARY_FIELDNAMES)
        self.assertIn("reference_frequency_mhz", SUMMARY_FIELDNAMES)
        self.assertIn("dram_accesses", SUMMARY_FIELDNAMES)
        self.assertIn("dram_energy_j", SUMMARY_FIELDNAMES)
        self.assertIn("covered_by_pe", SUMMARY_FIELDNAMES)
        self.assertNotIn("level1_area", SUMMARY_FIELDNAMES)
        self.assertTrue(rows[0]["constraints_satisfied"])
        self.assertEqual(rows[0]["constraint_violations"], [])
        self.assertAlmostEqual(rows[0]["inferences_per_second"], 50_000.0)
        self.assertEqual(rows[0]["reference_frequency_mhz"], 100.0)
        self.assertEqual(rows[0]["level2_power"], 0.1)
        self.assertEqual(rows[0]["workload_energy_j"], 2e-6)
        self.assertEqual(rows[0]["dram_accesses"], 1000)
        self.assertEqual(rows[0]["dram_energy_j"], 1e-6)
        self.assertEqual(rows[0]["covered_by_pe"], {"rf_i1": "pe_tile"})

    def test_full_flow_summary_reports_base_level1_metrics(self) -> None:
        rows = build_summary_rows(
            architecture_index=0,
            level1_raw_genome=[0.0] * GENOME_LENGTH,
            level1_discrete_genome=[0] * GENOME_LENGTH,
            level1_architecture_config={},
            level1_objective_values=[30.0],
            level1_objective_names=["area"],
            level2_objective_names=["area"],
            level1_csv_path="level1.csv",
            level2_csv_path=None,
            level2_solutions=[
                {
                    "solution_index": 0,
                    "valid": True,
                    "physical_fmax_mhz": 500.0,
                    "constraint_violations": [],
                }
            ],
            constraints=UserConstraints(),
            level1_evaluation=EvaluationResult(
                latency=10.0,
                energy=20.0,
                area=30.0,
                valid=True,
            ),
        )

        self.assertEqual(rows[0]["level1_latency"], 10.0)
        self.assertEqual(rows[0]["level1_energy"], 20.0)
        self.assertEqual(rows[0]["level1_area_proxy"], 30.0)
        self.assertEqual(rows[0]["inferences_per_second"], "")

    def test_level1_selection_skips_physically_infeasible_candidates(self) -> None:
        pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        profile = WorkloadActivityProfile(
            layers=(
                LayerActivity(
                    layer_id="test",
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
        candidates = select_level1_candidates(
            level1_genomes=[
                [0, 0, 0, 0, 1, 3, 2],
                [1, 2, 0, 2, 1, 0, 1],
            ],
            level1_objectives=[
                [255_671_633.0, 1.0, 1.0],
                [14_194_047.0, 1.0, 1.0],
            ],
            level1_objective_names=["latency", "energy", "area"],
            max_architectures=1,
            pool=pool,
            decode_genome=decode_genome,
            gene_bounds=lambda: [(0, len(spec.options) - 1) for spec in GENOME_SPEC],
            abstract_accelerator_from_level1_config=abstract_accelerator_from_level1_config,
            constraints=UserConstraints(
                max_area_mm2=0.4,
                max_power_w=0.12,
                min_frequency_mhz=800.0,
            ),
            evaluate_activity=lambda _genome: EvaluationResult(
                latency=1,
                energy=1,
                area=1,
                valid=True,
                activity_profile=profile,
            ),
        )

        self.assertEqual(candidates[0].source_index, 1)
        self.assertEqual(candidates[0].discrete_genome, [1, 2, 0, 2, 1, 0, 1])

    def test_level1_physical_prefilter_does_not_reject_large_spaces(self) -> None:
        failures: list[str] = []
        candidates = select_level1_candidates(
            level1_genomes=[[0] * GENOME_LENGTH],
            level1_objectives=[[1.0]],
            level1_objective_names=["area"],
            max_architectures=1,
            pool=IPPool.from_yaml(SYNTHETIC_POOL_PATH),
            decode_genome=decode_genome,
            gene_bounds=lambda: [(0, len(spec.options) - 1) for spec in GENOME_SPEC],
            abstract_accelerator_from_level1_config=abstract_accelerator_from_level1_config,
            constraints=UserConstraints(max_area_mm2=1.0),
            exhaustive_max_combinations=1,
            failures=failures,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(failures, [])

    def test_level1_selection_deduplicates_discrete_genomes(self) -> None:
        pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        candidates = select_level1_candidates(
            level1_genomes=[
                [0.1, 0, 0, 0, 0, 0, 0],
                [0.2, 0, 0, 0, 0, 0, 0],
                [1.0, 0, 0, 0, 0, 0, 0],
            ],
            level1_objectives=[[1.0], [1.0], [2.0]],
            level1_objective_names=["area"],
            max_architectures=3,
            pool=pool,
            decode_genome=decode_genome,
            gene_bounds=lambda: [(0, len(spec.options) - 1) for spec in GENOME_SPEC],
            abstract_accelerator_from_level1_config=abstract_accelerator_from_level1_config,
        )

        self.assertEqual(
            [candidate.discrete_genome for candidate in candidates],
            [[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
        )

    def test_level2_solution_rows_are_deduplicated_by_selected_ips(self) -> None:
        component = AbstractComponent(name="pe", type="pe")
        problem = Level2PymooProblem(
            accelerator=AbstractAccelerator(name="a", components=[component]),
            ip_pool=IPPool(
                [
                    IPBlock(id="pe0", type="pe", area=1, throughput=1, delay=1),
                    IPBlock(id="pe1", type="pe", area=2, throughput=1, delay=1),
                ]
            ),
            objective_names=["area"],
        )

        rows = _build_solution_rows(
            problem=problem,
            solution_vectors=[[0.1], [0.2], [1.0]],
            pop_size=3,
            n_gen=1,
            seed=1,
        )

        self.assertEqual([row["selected_ips"]["pe"] for row in rows], ["pe0", "pe1"])

    def test_level2_solution_rows_exclude_infeasible_candidates(self) -> None:
        component = AbstractComponent(name="pe", type="pe")
        problem = Level2PymooProblem(
            accelerator=AbstractAccelerator(name="a", components=[component]),
            ip_pool=IPPool(
                [
                    IPBlock(
                        id="pe0",
                        type="pe",
                        area=1,
                        throughput=1,
                        delay=1,
                    )
                ]
            ),
            objective_names=["area"],
            constraints=UserConstraints(max_area_mm2=0.5),
        )

        rows = _build_solution_rows(
            problem=problem,
            solution_vectors=[[0]],
            pop_size=1,
            n_gen=1,
            seed=1,
        )

        self.assertEqual(rows, [])

    def test_level2_energy_objective_requires_activity_profile(self) -> None:
        component = AbstractComponent(name="pe", type="pe")
        with self.assertRaisesRegex(ValueError, "activity profile"):
            Level2PymooProblem(
                accelerator=AbstractAccelerator(name="a", components=[component]),
                ip_pool=IPPool(
                    [
                        IPBlock(
                            id="pe0",
                            type="pe",
                            area=1,
                            throughput=1,
                            delay=1,
                        ),
                    ]
                ),
                objective_names=["energy"],
            )

    def test_synthetic_ip_pool_loads_and_covers_level1_genome_space(self) -> None:
        pool = IPPool.from_yaml(SYNTHETIC_POOL_PATH)
        total = 0

        for genome in product(*[range(len(spec.options)) for spec in GENOME_SPEC]):
            accelerator = abstract_accelerator_from_level1_config(
                decode_genome(list(genome))
            )
            for component in accelerator.components:
                self.assertTrue(pool.find_compatible(component))
            total += 1

        self.assertEqual(total, 57_600)

    def test_full_flow_accepts_workers_flag(self) -> None:
        args = parse_args(["--workers", "8", "--level2-strategy", "exhaustive"])

        self.assertEqual(args.workers, 8)
        self.assertEqual(args.level2_strategy, "exhaustive")
        self.assertEqual(parse_args([]).level2_strategy, "nsga2")
        self.assertEqual(parse_args(["--level1-objectives", "latency"]).level1_objectives, ["latency"])
        self.assertEqual(parse_args(["--level2-objectives", "delay"]).level2_objectives, ["delay"])
        self.assertEqual(parse_args(["--level2-objectives", "energy"]).level2_objectives, ["energy"])

    def test_full_flow_distinguishes_failure_from_no_feasible_designs(self) -> None:
        config = decode_genome([0] * GENOME_LENGTH)
        candidate = Level1Candidate(
            source_index=0,
            raw_genome=[0.0] * GENOME_LENGTH,
            objective_values=[1.0],
            discrete_genome=[0] * GENOME_LENGTH,
            architecture_config=config,
            accelerator=abstract_accelerator_from_level1_config(config),
        )
        level1_result = SimpleNamespace(
            X=np.zeros((1, GENOME_LENGTH)),
            F=np.ones((1, 1)),
            talos=SimpleNamespace(csv_path=None),
        )
        dram = IPBlock(
            id="dram",
            type="dram",
            area=0,
            throughput=1,
            delay=1,
            fmax_mhz=500,
            bandwidth_bits=512,
            metadata={"accesses_per_cycle": 1},
            power_model=PowerCharacterization(
                source="test",
                activity_method="access_rate",
                reference_frequency_mhz=500,
                p_idle_w=0.02,
                p_active_w=4.5,
            ),
        )
        pool = SimpleNamespace(
            by_type=lambda ip_type: [dram] if ip_type == "dram" else []
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload = root / "workload.onnx"
            ip_pool = root / "pool.yaml"
            workload.touch()
            ip_pool.touch()
            args = parse_args(
                [
                    "--workload",
                    str(workload),
                    "--ip-pool",
                    str(ip_pool),
                    "--results-dir",
                    str(root / "results"),
                    "--level1-objectives",
                    "area",
                    "--level2-objectives",
                    "area",
                ]
            )

            for outcome, expected_code in (
                (RuntimeError("characterization failed"), 1),
                (SimpleNamespace(solutions=[], csv_path=None), 0),
            ):
                run_patch = (
                    patch(
                        "talos.level2.runner.run_level2",
                        side_effect=outcome,
                    )
                    if isinstance(outcome, Exception)
                    else patch(
                        "talos.level2.runner.run_level2",
                        return_value=outcome,
                    )
                )
                with (
                    patch(
                        "examples.full_flow_example.parse_args",
                        return_value=args,
                    ),
                    patch("talos.ip.IPPool.from_yaml", return_value=pool),
                    patch(
                        "talos.ga.pymoo_runner.run_nsga2_pymoo",
                        return_value=level1_result,
                    ),
                    patch("talos.evaluation.zigzag_evaluator.ZigZagEvaluator"),
                    patch(
                        "examples.full_flow_example.select_level1_candidates",
                        return_value=[candidate],
                    ),
                    run_patch,
                ):
                    self.assertEqual(full_flow_main(), expected_code)

    def test_constraint_sweep_builds_seven_worker_aware_commands(self) -> None:
        cases = sweep_cases()
        args = build_parser().parse_args(["--workers", "8"])
        command = build_command(
            python="python",
            args=args,
            case=cases[0],
            case_dir=Path("results/constraint_sweep/test/baseline"),
        )

        self.assertEqual(len(cases), 7)
        self.assertEqual(cases[0].min_frequency_mhz, 700.0)
        self.assertEqual(args.max_architectures, 40)
        self.assertEqual(args.level2_strategy, "exhaustive")
        self.assertIn("--workers", command)
        self.assertEqual(command[command.index("--workers") + 1], "8")
        self.assertIn("--level2-strategy", command)
        self.assertEqual(
            command[command.index("--level2-strategy") + 1],
            "exhaustive",
        )
        self.assertIn("ip_pool_synthetic_28nm.yaml", command[command.index("--ip-pool") + 1])

    def test_objective_sweep_builds_paired_objective_commands(self) -> None:
        cases = objective_cases()
        args = build_objective_sweep_parser().parse_args(["--workers", "8"])
        command = build_objective_sweep_command(
            python="python",
            args=args,
            case=cases[0],
            case_dir=Path("results/objective_sweep/test/case"),
        )

        self.assertEqual(
            [case.name for case in cases],
            [
                "energy",
                "area",
                "performance",
                "energy_area",
                "area_performance",
                "energy_performance",
                "energy_area_performance",
            ],
        )
        self.assertEqual(
            cases[-1].level1_objectives,
            ["energy", "area", "latency"],
        )
        self.assertEqual(
            cases[-1].level2_objectives,
            ["energy", "area", "workload_latency_s"],
        )
        self.assertIn("--level1-objectives", command)
        self.assertIn("--level2-objectives", command)
        self.assertIn("--workers", command)
        self.assertEqual(command[command.index("--workers") + 1], "8")
        self.assertEqual(args.level2_strategy, "exhaustive")
        self.assertEqual(args.max_architectures, 3)
        self.assertEqual(command[command.index("--min-frequency-mhz") + 1], "600.0")

        args = build_objective_sweep_parser().parse_args(["--no-constraints"])
        command = build_objective_sweep_command(
            python="python",
            args=args,
            case=cases[0],
            case_dir=Path("results/objective_sweep/test/case"),
        )
        self.assertNotIn("--max-area-mm2", command)
        self.assertNotIn("--max-power-w", command)
        self.assertNotIn("--min-frequency-mhz", command)


if __name__ == "__main__":
    unittest.main()
