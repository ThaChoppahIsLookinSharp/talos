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


@dataclass(frozen=True)
class SweepCase:
    name: str
    max_area_mm2: float
    max_power_w: float
    min_frequency_mhz: float


def sweep_cases() -> list[SweepCase]:
    return [
        SweepCase("baseline", 0.40, 0.12, 700.0),
        SweepCase("strict_area", 0.04, 0.12, 700.0),
        SweepCase("strict_power", 0.40, 0.011, 700.0),
        SweepCase("strict_frequency", 0.40, 0.12, 800.0),
        SweepCase("relaxed_area", 1.50, 0.12, 700.0),
        SweepCase("relaxed_power", 0.40, 0.50, 700.0),
        SweepCase("relaxed_frequency", 0.40, 0.12, 550.0),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TALOS 7-case user-constraint sweep.",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=DEFAULT_WORKLOAD,
    )
    parser.add_argument(
        "--ip-pool",
        type=Path,
        default=DEFAULT_IP_POOL,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "constraint_sweep",
    )
    parser.add_argument("--level1-pop-size", type=int, default=40)
    parser.add_argument("--level1-generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--level2-pop-size", type=int, default=24)
    parser.add_argument("--level2-generations", type=int, default=4)
    parser.add_argument(
        "--level2-strategy",
        choices=["nsga2", "exhaustive"],
        default="exhaustive",
    )
    parser.add_argument("--level2-exhaustive-max-combinations", type=int, default=100_000)
    parser.add_argument("--max-architectures", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(
    *,
    python: str,
    args: argparse.Namespace,
    case: SweepCase,
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
        "--workers",
        str(args.workers),
        "--level2-pop-size",
        str(args.level2_pop_size),
        "--level2-generations",
        str(args.level2_generations),
        "--level2-strategy",
        args.level2_strategy,
        "--level2-exhaustive-max-combinations",
        str(args.level2_exhaustive_max_combinations),
        "--max-architectures",
        str(args.max_architectures),
        "--seed",
        str(args.seed),
        "--max-area-mm2",
        str(case.max_area_mm2),
        "--max-power-w",
        str(case.max_power_w),
        "--min-frequency-mhz",
        str(case.min_frequency_mhz),
    ]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.csv"

    rows: list[dict[str, object]] = []
    for case in sweep_cases():
        case_dir = run_root / case.name
        command = build_command(
            python=sys.executable,
            args=args,
            case=case,
            case_dir=case_dir,
        )
        case_dir.mkdir(parents=True, exist_ok=True)

        return_code = 0
        if args.dry_run:
            print(" ".join(command))
        else:
            completed = subprocess.run(command, cwd=REPO_ROOT)
            return_code = completed.returncode

        rows.append(
            {
                "case": case.name,
                "max_area_mm2": case.max_area_mm2,
                "max_power_w": case.max_power_w,
                "min_frequency_mhz": case.min_frequency_mhz,
                "results_dir": str(case_dir),
                "summary_csv": str(case_dir / "full_flow_summary.csv"),
                "command": json.dumps(command),
                "return_code": return_code,
            }
        )

    write_manifest(manifest_path, rows)
    print(f"Manifest: {manifest_path}")
    return 1 if any(row["return_code"] != 0 for row in rows) else 0


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "max_area_mm2",
        "max_power_w",
        "min_frequency_mhz",
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
