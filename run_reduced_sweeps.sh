#!/usr/bin/env bash
set -Eeuo pipefail

root=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd
)
cd "$root"

python="$root/.venv/bin/python"
stamp=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
base="$root/results/reduced_sweeps/$stamp"

mkdir -p "$base"
exec > >(tee "$base/run.log") 2>&1

# Eyeriss: INT16, 80 x 4, 10 workers, 24 architectures.
"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/energy" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives energy \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy \
  --max-architectures 24 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/energy_area" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives energy area \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area \
  --max-architectures 24 \
  --seed 1

"$python" - \
  "$base/int16_eyeriss/energy/full_flow_summary.csv" \
  "$base/int16_eyeriss/energy_area/full_flow_summary.csv" <<'PY'
import csv
import math
import sys


def minimum(path):
    with open(path, newline="", encoding="utf-8") as handle:
        values = [
            float(row["workload_energy_j"])
            for row in csv.DictReader(handle)
            if row["constraints_satisfied"].lower() == "true"
        ]
    if not values or not all(map(math.isfinite, values)):
        raise SystemExit(f"No finite valid energy in {path}")
    return min(values)


energy = minimum(sys.argv[1])
joint = minimum(sys.argv[2])
print(f"energy minimum:      {energy:.12g} J")
print(f"energy+area minimum: {joint:.12g} J")
if energy > joint + max(1e-12, abs(joint) * 1e-9):
    raise SystemExit(
        "Energy-only did not find the lowest energy; review it."
    )
PY

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/area" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives area \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area \
  --max-architectures 24 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/performance" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives latency \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives workload_latency_s \
  --max-architectures 24 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/area_performance" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives area latency \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area workload_latency_s \
  --max-architectures 24 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/int16_eyeriss/energy_performance" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives energy latency \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy workload_latency_s \
  --max-architectures 24 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/eyeriss_alexnet_conv_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir \
    "$base/int16_eyeriss/energy_area_performance" \
  --level1-pop-size 80 \
  --level1-generations 4 \
  --level1-objectives energy area latency \
  --workers 10 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area workload_latency_s \
  --max-architectures 24 \
  --seed 1

# FP16: 40 x 4, 8 workers, 12 architectures.
"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/energy" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/energy_area" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy area \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/area" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives area \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/area_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives area latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/energy_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_fp16.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir "$base/fp16/energy_area_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy area latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area workload_latency_s \
  --max-architectures 12 \
  --seed 1

# INT8: 40 x 4, 8 workers, 12 architectures.
"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/energy" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/energy_area" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy area \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/area" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives area \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/area_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives area latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives area workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/energy_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy workload_latency_s \
  --max-architectures 12 \
  --seed 1

"$python" -u examples/full_flow_example.py \
  --workload workloads/squeezenet1_0_int8.onnx \
  --ip-pool configs/ip_pool_characterized_65nm.yaml \
  --results-dir "$base/int8/energy_area_performance" \
  --level1-pop-size 40 \
  --level1-generations 4 \
  --level1-objectives energy area latency \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --level2-objectives energy area workload_latency_s \
  --max-architectures 12 \
  --seed 1

echo "All sweeps completed: $base"
