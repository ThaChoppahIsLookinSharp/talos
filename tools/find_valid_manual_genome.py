from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.manual_validation import (
    find_first_valid_manual_genome,
    format_validation_summary,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    workload = repo_root() / "workloads" / "alexnet.onnx"
    workdir = repo_root() / ".talos_zigzag" / "find_valid_manual_genome"

    found, attempts = find_first_valid_manual_genome(
        workload_path=str(workload),
        workdir=str(workdir),
        timeout_seconds=10.0,
    )

    for idx, summary in enumerate(attempts):
        print(f"=== candidate_{idx:03d} ===")
        print(format_validation_summary(summary))
        print()

    if found is not None:
        print("FOUND_VALID_MANUAL_GENOME")
        print(format_validation_summary(found))
        return

    print("NO_VALID_MANUAL_GENOME_FOUND")
    completed = [summary for summary in attempts if not summary.timed_out]
    timed_out = [summary for summary in attempts if summary.timed_out]
    print(f"completed_attempts={len(completed)}")
    print(f"timed_out_attempts={len(timed_out)}")
    if completed:
        print("best_completed_failure:")
        print(format_validation_summary(completed[-1]))


if __name__ == "__main__":
    main()
