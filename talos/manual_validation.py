from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Any

from talos.architecture.genome import GENOME_LENGTH, decode_genome, default_genome
from talos.architecture.memory_specs import GB_SIZE_OPTIONS, RF_SIZE_OPTIONS, bits_to_bytes
from talos.evaluation.zigzag_evaluator import EvaluationResult, ZigZagEvaluator


@dataclass(frozen=True)
class ManualValidationSummary:
    genome: list[int]
    decoded_architecture: str
    valid: bool
    latency: float
    energy: float
    area: float
    area_source: str
    area_is_proxy: bool
    raw_zigzag_area: float | None
    zigzag_area_path: str | None
    error_message: str | None
    area_policy: str
    memory_cost_mode: str
    accelerator_yaml_path: str | None
    timed_out: bool = False


def manual_smoke_genome_candidates() -> list[list[int]]:
    rf_code_by_bytes = {
        bits_to_bytes(size_bits): idx for idx, size_bits in enumerate(RF_SIZE_OPTIONS)
    }
    gb_code_by_bytes = {
        bits_to_bytes(size_bits): idx for idx, size_bits in enumerate(GB_SIZE_OPTIONS)
    }

    candidates: list[list[int]] = []
    candidate_specs = [
        (1, 1, 4096, 16384, 3),
        (1, 1, 4096, 16384, 0),
        (1, 1, 2048, 16384, 3),
        (1, 1, 1024, 16384, 3),
        (1, 0, 4096, 16384, 3),
        (1, 0, 2048, 16384, 3),
        (1, 0, 1024, 8192, 3),
        (0, 0, 1024, 4096, 3),
        (0, 0, 2048, 4096, 3),
        (0, 0, 1024, 8192, 3),
        (0, 0, 2048, 8192, 3),
        (0, 0, 2048, 16384, 1),
        (0, 0, 2048, 16384, 2),
    ]

    for pe_x_code, pe_y_code, rf_bytes, gb_bytes, served_dims_code in candidate_specs:
        gb_code = gb_code_by_bytes.get(gb_bytes)
        rf_code = rf_code_by_bytes.get(rf_bytes)
        if gb_code is None or rf_code is None:
            continue
        candidates.append([pe_x_code, pe_y_code, rf_code, gb_code, served_dims_code])
    return candidates


def known_valid_manual_genome() -> list[int] | None:
    return None


def parse_genome_arg(value: str) -> list[int]:
    genome = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(genome) != GENOME_LENGTH:
        raise ValueError(
            f"Expected {GENOME_LENGTH} comma-separated genes, got {len(genome)}."
        )
    return genome


def fallback_manual_diagnostic_genome() -> list[int]:
    candidates = manual_smoke_genome_candidates()
    return candidates[-1] if candidates else default_genome()


def select_manual_reference_genome() -> list[int] | None:
    known = known_valid_manual_genome()
    if known is not None:
        return known
    return None


def format_area_diagnostics(summary: ManualValidationSummary) -> str:
    lines = [
        f"area={summary.area}",
        f"area_source={summary.area_source}",
        f"area_is_proxy={summary.area_is_proxy}",
        f"raw_zigzag_area={summary.raw_zigzag_area}",
        f"zigzag_area_path={summary.zigzag_area_path}",
        f"area_policy={summary.area_policy}",
    ]
    return "\n".join(lines)


def format_validation_summary(summary: ManualValidationSummary) -> str:
    lines = [
        f"genome={summary.genome}",
        f"decoded_architecture={summary.decoded_architecture}",
        f"memory_cost_mode={summary.memory_cost_mode}",
        f"valid={summary.valid}",
        f"timed_out={summary.timed_out}",
        f"latency={summary.latency}",
        f"energy={summary.energy}",
        format_area_diagnostics(summary),
        f"accelerator_yaml_path={summary.accelerator_yaml_path}",
        f"error_message={summary.error_message}",
    ]
    return "\n".join(lines)


def _summary_from_result(
    *,
    genome: list[int],
    area_policy: str,
    memory_cost_mode: str,
    accelerator_yaml_path: str | None,
    result: EvaluationResult,
) -> ManualValidationSummary:
    cfg = decode_genome(genome)
    return ManualValidationSummary(
        genome=list(genome),
        decoded_architecture=repr(cfg),
        valid=result.valid,
        latency=result.latency,
        energy=result.energy,
        area=result.area,
        area_source=result.area_source,
        area_is_proxy=result.area_is_proxy,
        raw_zigzag_area=result.raw_zigzag_area,
        zigzag_area_path=result.zigzag_area_path,
        error_message=result.error_message,
        area_policy=area_policy,
        memory_cost_mode=memory_cost_mode,
        accelerator_yaml_path=accelerator_yaml_path,
    )


def _evaluate_genome_worker(
    queue: mp.Queue,
    *,
    genome: list[int],
    workload_path: str,
    workdir: str,
    memory_cost_mode: str,
    area_policy: str,
    lpf_limit: int,
    nb_spatial_mappings_generated: int,
) -> None:
    try:
        evaluator = ZigZagEvaluator(
            workload=workload_path,
            workdir=workdir,
            memory_cost_mode=memory_cost_mode,
            area_policy=area_policy,
            lpf_limit=lpf_limit,
            nb_spatial_mappings_generated=nb_spatial_mappings_generated,
        )
        accelerator = evaluator.build_accelerator_from_genome(genome)
        accelerator_yaml_path = evaluator._write_accelerator_yaml(accelerator)
        result = evaluator.evaluate(genome)
        summary = _summary_from_result(
            genome=genome,
            area_policy=area_policy,
            memory_cost_mode=memory_cost_mode,
            accelerator_yaml_path=accelerator_yaml_path,
            result=result,
        )
        queue.put(asdict(summary))
    except Exception as exc:
        cfg = decode_genome(genome)
        queue.put(
            asdict(
                ManualValidationSummary(
                    genome=list(genome),
                    decoded_architecture=repr(cfg),
                    valid=False,
                    latency=float("inf"),
                    energy=float("inf"),
                    area=float("inf"),
                    area_source="missing",
                    area_is_proxy=False,
                    raw_zigzag_area=None,
                    zigzag_area_path=None,
                    error_message=str(exc),
                    area_policy=area_policy,
                    memory_cost_mode=memory_cost_mode,
                    accelerator_yaml_path=None,
                )
            )
        )


def evaluate_genome_with_timeout(
    genome: list[int],
    *,
    workload_path: str,
    workdir: str,
    memory_cost_mode: str = "manual",
    area_policy: str = "prefer_zigzag_then_proxy",
    lpf_limit: int = 1,
    nb_spatial_mappings_generated: int = 1,
    timeout_seconds: float = 10.0,
) -> ManualValidationSummary:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_evaluate_genome_worker,
        kwargs={
            "queue": queue,
            "genome": list(genome),
            "workload_path": workload_path,
            "workdir": workdir,
            "memory_cost_mode": memory_cost_mode,
            "area_policy": area_policy,
            "lpf_limit": lpf_limit,
            "nb_spatial_mappings_generated": nb_spatial_mappings_generated,
        },
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        cfg = decode_genome(genome)
        return ManualValidationSummary(
            genome=list(genome),
            decoded_architecture=repr(cfg),
            valid=False,
            latency=float("inf"),
            energy=float("inf"),
            area=float("inf"),
            area_source="missing",
            area_is_proxy=False,
            raw_zigzag_area=None,
            zigzag_area_path=None,
            error_message=f"Evaluation timed out after {timeout_seconds:.0f}s.",
            area_policy=area_policy,
            memory_cost_mode=memory_cost_mode,
            accelerator_yaml_path=None,
            timed_out=True,
        )

    try:
        payload: dict[str, Any] = queue.get_nowait()
    except Empty:
        cfg = decode_genome(genome)
        return ManualValidationSummary(
            genome=list(genome),
            decoded_architecture=repr(cfg),
            valid=False,
            latency=float("inf"),
            energy=float("inf"),
            area=float("inf"),
            area_source="missing",
            area_is_proxy=False,
            raw_zigzag_area=None,
            zigzag_area_path=None,
            error_message="Evaluation process exited without returning a result.",
            area_policy=area_policy,
            memory_cost_mode=memory_cost_mode,
            accelerator_yaml_path=None,
        )

    return ManualValidationSummary(**payload)


def find_first_valid_manual_genome(
    *,
    workload_path: str,
    workdir: str,
    timeout_seconds: float = 10.0,
) -> tuple[ManualValidationSummary | None, list[ManualValidationSummary]]:
    attempts: list[ManualValidationSummary] = []
    for idx, genome in enumerate(manual_smoke_genome_candidates()):
        summary = evaluate_genome_with_timeout(
            genome,
            workload_path=workload_path,
            workdir=str(Path(workdir) / f"candidate_{idx:03d}"),
            memory_cost_mode="manual",
            area_policy="prefer_zigzag_then_proxy",
            lpf_limit=1,
            nb_spatial_mappings_generated=1,
            timeout_seconds=timeout_seconds,
        )
        attempts.append(summary)
        if summary.valid:
            return summary, attempts
    return None, attempts
