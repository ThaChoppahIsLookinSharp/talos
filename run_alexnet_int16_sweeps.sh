#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"

python="$root/.venv/bin/python"
stamp=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
base="$root/results/alexnet_int16_sweeps/$stamp"
handoff="$base/level1_handoff.json"
# Edit these three values to set the Level-1 exploration budget.
talos_l1_pop_size=${TALOS_L1_POP_SIZE:-240}
talos_l1_generations=${TALOS_L1_GENERATIONS:-8}
talos_max_architectures=${TALOS_MAX_ARCHITECTURES:-72}

mkdir -p "$base"
exec > >(tee "$base/run.log") 2>&1

# Run the pool-independent Pareto screen and ZigZag profiling once.
"$python" -u examples/full_flow_example.py \
  --workload workloads/alexnet_int16.onnx \
  --ip-pool configs/ip_pool_int16_65nm.yaml \
  --results-dir "$base/level1" \
  --level1-pop-size "$talos_l1_pop_size" \
  --level1-generations "$talos_l1_generations" \
  --max-architectures "$talos_max_architectures" \
  --workers 10 \
  --min-freq 125 \
  --seed 42 \
  --level1-only \
  --level1-handoff-output "$handoff"

run_level2() {
  local talos_case=$1
  shift
  "$python" -u examples/full_flow_example.py \
    --workload workloads/alexnet_int16.onnx \
    --ip-pool configs/ip_pool_int16_65nm.yaml \
    --results-dir "$base/$talos_case" \
    --level1-handoff "$handoff" \
    --workers 10 \
    --level2-strategy exhaustive \
    --level2-exhaustive-max-combinations 100000 \
    --min-freq 125 \
    --seed 42 \
    --level2-objectives "$@"
}

run_level2 energy energy
run_level2 energy_area energy area
run_level2 area area
run_level2 performance workload_latency_s
run_level2 area_performance area workload_latency_s
run_level2 energy_performance energy workload_latency_s
run_level2 energy_area_performance energy area workload_latency_s

echo "Results: $base"
