from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.manual_validation import (
    fallback_manual_diagnostic_genome,
    find_first_valid_manual_genome,
    format_validation_summary,
    parse_genome_arg,
    select_manual_reference_genome,
    evaluate_genome_with_timeout,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect area source in manual mode")
    parser.add_argument(
        "--genome",
        type=str,
        help="Comma-separated genome codes, for example: 1,1,3,4,3",
    )
    return parser


def _resolve_genome(genome_arg: str | None) -> tuple[list[int], str]:
    if genome_arg is not None:
        return parse_genome_arg(genome_arg), "cli"

    known = select_manual_reference_genome()
    if known is not None:
        return known, "known_valid_manual"

    workload = repo_root() / "workloads" / "alexnet.onnx"
    found, _attempts = find_first_valid_manual_genome(
        workload_path=str(workload),
        workdir=str(repo_root() / ".talos_zigzag" / "check_area_source" / "manual_search"),
        timeout_seconds=10.0,
    )
    if found is not None:
        return found.genome, "search_valid_manual"

    return fallback_manual_diagnostic_genome(), "fallback_diagnostic_candidate"


def main() -> None:
    args = build_parser().parse_args()
    genome, genome_source = _resolve_genome(args.genome)
    workload = repo_root() / "workloads" / "alexnet.onnx"
    base_workdir = repo_root() / ".talos_zigzag" / "check_area_source"

    print(f"reference_genome_source={genome_source}")
    print(f"reference_genome={genome}")
    print()

    for area_policy in ("prefer_zigzag_then_proxy", "zigzag_only"):
        print(f"=== area_policy={area_policy} ===")
        summary = evaluate_genome_with_timeout(
            genome,
            workload_path=str(workload),
            workdir=str(base_workdir / area_policy),
            memory_cost_mode="manual",
            area_policy=area_policy,
            lpf_limit=1,
            nb_spatial_mappings_generated=1,
            timeout_seconds=15.0,
        )
        print(format_validation_summary(summary))
        if summary.area_source == "zigzag" and not summary.area_is_proxy:
            print("interpretation=ZigZag exposed usable area and TALOS used it.")
        elif summary.area_source == "proxy" and summary.area_is_proxy:
            print("interpretation=TALOS fell back to its internal proxy area.")
        elif summary.area_source == "missing":
            print("interpretation=No usable area made it through this path.")
        print()


if __name__ == "__main__":
    main()
