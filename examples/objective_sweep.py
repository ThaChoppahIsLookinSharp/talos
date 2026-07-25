from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IP_POOL = REPO_ROOT / "configs" / "ip_pool_synthetic_28nm.yaml"
DEFAULT_WORKLOAD = REPO_ROOT / "workloads" / "alexnet.onnx"
FULL_FLOW_SCRIPT = REPO_ROOT / "examples" / "full_flow_example.py"

OBJECTIVE_MAP = {
    "energy": (["energy"], ["energy"]),
    "area": (["area"], ["area"]),
    "performance": (["latency"], ["delay"]),
}
OBJECTIVE_CASES = [
    ("energy", ["energy"]),
    ("area", ["area"]),
    ("performance", ["performance"]),
    ("energy_area", ["energy", "area"]),
    ("area_performance", ["area", "performance"]),
    ("energy_performance", ["energy", "performance"]),
    ("energy_area_performance", ["energy", "area", "performance"]),
]


@dataclass(frozen=True)
class ObjectiveCase:
    name: str
    level1_objectives: list[str]
    level2_objectives: list[str]


def objective_cases() -> list[ObjectiveCase]:
    cases = []
    for name, objective_names in OBJECTIVE_CASES:
        level1_objectives: list[str] = []
        level2_objectives: list[str] = []
        for objective_name in objective_names:
            level1_names, level2_names = OBJECTIVE_MAP[objective_name]
            level1_objectives.extend(level1_names)
            level2_objectives.extend(level2_names)
        cases.append(
            ObjectiveCase(
                name=name,
                level1_objectives=level1_objectives,
                level2_objectives=level2_objectives,
            )
        )
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TALOS paired objective sweep.")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--ip-pool", type=Path, default=DEFAULT_IP_POOL)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "objective_sweep",
    )
    parser.add_argument("--level1-pop-size", type=int, default=40)
    parser.add_argument("--level1-generations", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--level2-pop-size", type=int, default=32)
    parser.add_argument("--level2-generations", type=int, default=4)
    parser.add_argument(
        "--level2-strategy",
        choices=["nsga2", "exhaustive"],
        default="nsga2",
    )
    parser.add_argument("--level2-exhaustive-max-combinations", type=int, default=100_000)
    parser.add_argument("--max-architectures", type=int, default=12)
    parser.add_argument("--max-area-mm2", type=float, default=0.40)
    parser.add_argument("--max-power-w", type=float, default=0.12)
    parser.add_argument("--min-frequency-mhz", type=float, default=550.0)
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(
    *,
    python: str,
    args: argparse.Namespace,
    case: ObjectiveCase,
    case_dir: Path,
) -> list[str]:
    command = [
        python,
        str(FULL_FLOW_SCRIPT),
        "--workload",
        str(args.workload),
        "--ip-pool",
        str(args.ip_pool),
        "--results-dir",
        str(case_dir),
        "--level1-pop-size",
        str(args.level1_pop_size),
        "--level1-generations",
        str(args.level1_generations),
        "--level1-objectives",
        *case.level1_objectives,
        "--workers",
        str(args.workers),
        "--level2-pop-size",
        str(args.level2_pop_size),
        "--level2-generations",
        str(args.level2_generations),
        "--level2-objectives",
        *case.level2_objectives,
        "--level2-strategy",
        args.level2_strategy,
        "--level2-exhaustive-max-combinations",
        str(args.level2_exhaustive_max_combinations),
        "--max-architectures",
        str(args.max_architectures),
        "--seed",
        str(args.seed),
    ]
    if not args.no_constraints:
        command.extend(
            [
                "--max-area-mm2",
                str(args.max_area_mm2),
                "--max-power-w",
                str(args.max_power_w),
                "--min-frequency-mhz",
                str(args.min_frequency_mhz),
            ]
        )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for case in objective_cases():
        case_dir = run_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            python=sys.executable,
            args=args,
            case=case,
            case_dir=case_dir,
        )
        return_code = 0
        if args.dry_run:
            print(" ".join(command))
        else:
            return_code = subprocess.run(command, cwd=REPO_ROOT).returncode
        rows.append(
            {
                "case": case.name,
                "level1_objectives": json.dumps(case.level1_objectives),
                "level2_objectives": json.dumps(case.level2_objectives),
                "results_dir": str(case_dir),
                "summary_csv": str(case_dir / "full_flow_summary.csv"),
                "command": json.dumps(command),
                "return_code": return_code,
            }
        )

    manifest_path = run_root / "manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Manifest: {manifest_path}")
    return 1 if any(row["return_code"] != 0 for row in rows) else 0


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case",
        "level1_objectives",
        "level2_objectives",
        "results_dir",
        "summary_csv",
        "command",
        "return_code",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
