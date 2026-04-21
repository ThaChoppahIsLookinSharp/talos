from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos.evaluation.zigzag_evaluator import ZigZagEvaluator


GENOME = [2, 2, 3, 2, 3, 2, 3, 3]


class InspectEvaluator(ZigZagEvaluator):
    def run_and_inspect(self, genome: list[float]) -> None:
        cfg = self.build_accelerator_from_genome(genome)
        accelerator_yaml_path = self._write_accelerator_yaml(cfg)

        with self._quiet_zigzag():
            with self._zigzag_runtime_env():
                energy, latency, cme = self._run_zigzag(accelerator_yaml_path)

        area, path = self._extract_zigzag_area(cme)

        print(f"energy={energy}")
        print(f"latency={latency}")
        print(f"cme_summary={self._summarize_cme(cme)}")
        print(f"zigzag_area_found={area is not None}")
        print(f"zigzag_area={area}")
        print(f"zigzag_area_path={path}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    workload = repo_root() / "workloads" / "alexnet.onnx"
    evaluator = InspectEvaluator(
        workload=str(workload),
        workdir=str(repo_root() / ".talos_zigzag" / "inspect_cme_area"),
        lpf_limit=1,
        nb_spatial_mappings_generated=1,
        debug_cme=True,
    )
    evaluator.run_and_inspect(GENOME)


if __name__ == "__main__":
    main()
