# Sweep Index

Common report folder: `results/sweep_reports`

Copied reports:

- `constraint_sweep_20260702_092220.md` -> `results/constraint_sweep/20260702_092220`
- `objective_sweep_20260705_134857.md` -> `results/objective_sweep/20260705_134857`
- `objective_sweep_paired_20260705.md` -> paired objective runs:
  - `results/objective_sweep_paired/exhaustive_fixed/20260705_180352`
  - `results/objective_sweep_paired/nsga2_fixed/20260705_183604`
- `objective_sweep_workload_power_20260718.md` -> `results/objective_sweep_paired/workload_power_long_fixed/20260718_214747`
- `objective_sweep_workload_power_dram_20260719.md` -> `results/objective_sweep_paired/workload_power_dram_long/20260719_133646`
- `objective_sweep_workload_energy_dram_20260725.md` -> `results/objective_sweep_paired/workload_energy_dram_long/20260725_130743`
- `energy_diverse_integer_20260725.md` -> `results/objective_sweep_paired/workload_energy_diverse_integer/20260725_170724`
- `objective_sweep_memory_utilization_1p2_20260728.md` -> `results/objective_sweep_memory_utilization_1p2_a238d4f/20260728_220236`
- `level2_no_constraints_replay_20260802.md` -> Level 2 replay from
  `results/objective_sweep_65nm/20260730_073716`, stored in
  `results/level2_replay_no_constraints_65nm/20260802_193540`
- `objective_sweep_fp16_resnet18_first_layer_20260812.md` ->
  `results/objective_sweep_fp16_resnet18_first_layer/20260812_082135`
- `objective_sweep_fp32_resnet18_20260813.md` ->
  `results/objective_sweep_fp32_resnet18_first_layer/20260813_222734`

## Runs

| family | run dir | cases | completed | notes |
|---|---|---:|---:|---|
| constraint_sweep | `results/constraint_sweep/20260628_230338` | 7 | 7 | first synthetic sweep |
| constraint_sweep | `results/constraint_sweep/20260629_090133` | 7 | 5 | NSGA-II Level 2, some failed cases |
| constraint_sweep | `results/constraint_sweep/20260629_205508` | 7 | 5 | exhaustive Level 2, some failed cases |
| constraint_sweep | `results/constraint_sweep/20260702_092220` | 7 | 5 | report copied here |
| objective_sweep | `results/objective_sweep/20260705_134857` | 9 | 9 | old 3x3 cross-product objective sweep; report copied here |
| objective_sweep | `results/objective_sweep/20260705_174727` | 7 | 7 | dry-run manifest only |
| objective_sweep_paired/exhaustive | `results/objective_sweep_paired/exhaustive/20260705_174739` | 0 | 0 | interrupted |
| objective_sweep_paired/exhaustive_small | `results/objective_sweep_paired/exhaustive_small/20260705_175618` | 0 | 0 | interrupted |
| objective_sweep_paired/exhaustive_fixed | `results/objective_sweep_paired/exhaustive_fixed/20260705_180352` | 7 | 7 | paired objectives, exhaustive Level 2; report copied here |
| objective_sweep_paired/nsga2_fixed | `results/objective_sweep_paired/nsga2_fixed/20260705_183604` | 7 | 7 | paired objectives, NSGA-II Level 2; report copied here |
| objective_sweep_paired/relaxed_all_exhaustive | `results/objective_sweep_paired/relaxed_all_exhaustive/20260705_225108` | 7 | 3 | cancelled during `power_area`; no manifest |
| objective_sweep_paired/no_constraints_exhaustive_big_l1 | `results/objective_sweep_paired/no_constraints_exhaustive_big_l1/20260705_232223` | 7 | 7 | no constraints, exhaustive Level 2, larger Level 1; report in run dir |
| objective_sweep_paired/workload_power_long_fixed | `results/objective_sweep_paired/workload_power_long_fixed/20260718_214747` | 7 | 7 | workload-aware power, permissive constraints, long Level 1, exhaustive Level 2 |
| objective_sweep_paired/workload_power_dram_long | `results/objective_sweep_paired/workload_power_dram_long/20260719_133646` | 7 | 7 | workload power including ZigZag DRAM access energy, permissive constraints, exhaustive Level 2 |
| objective_sweep_paired/workload_energy_dram_long | `results/objective_sweep_paired/workload_energy_dram_long/20260725_130743` | 7 | 7 | workload energy per inference in Level 2, average power retained, exhaustive Level 2 |
| objective_sweep_paired/workload_energy_diverse_integer | `results/objective_sweep_paired/workload_energy_diverse_integer/20260725_170724` | 1 | 1 | 32 integer Pareto candidates, exhaustive Level 2 |
| objective_sweep_memory_utilization_1p2 | `results/objective_sweep_memory_utilization_1p2_a238d4f/20260728_220236` | 7 | 6 | workload-aware performance, 1.2 memory-utilization margin; `performance` has no feasible design |
| objective_sweep_65nm | `results/objective_sweep_65nm/20260730_073716` | 7 | 7 | constrained source run at commit `3b3edfa`; Level 1/ZigZag profiles reused below |
| level2_replay_no_constraints_65nm | `results/level2_replay_no_constraints_65nm/20260802_193540` | 7 | 7 | exhaustive Level 2 replay, no user constraints, 128 architecture/mapping entries and 42164 valid rows |
| objective_sweep_fp16_resnet18_first_layer | `results/objective_sweep_fp16_resnet18_first_layer/20260812_082135` | 7 | 7 | FP16-only PE pool, FP32 first ResNet18 layer, no user constraints, 45600 valid rows |
| objective_sweep_fp32_resnet18_first_layer | `results/objective_sweep_fp32_resnet18_first_layer/20260813_222734` | 7 | 7 | FP32 PEs, FP32 first ResNet18 layer, no user constraints, 45600 valid rows |

## Commands And Options

Exact per-case `full_flow_example.py` commands are stored in each run `manifest.csv` when the sweep completed far enough to write one.

### Constraint Sweeps

`results/constraint_sweep/20260628_230338`

```bash
.venv/bin/python examples/constraint_sweep.py \
  --workers 8 \
  --level1-pop-size 40 \
  --level1-generations 3 \
  --level2-pop-size 12 \
  --level2-generations 3 \
  --max-architectures 4 \
  --seed 1
```

Options: `workloads/alexnet.onnx`, `configs/ip_pool_synthetic_28nm.yaml`, Level 2 `nsga2`, baseline frequency `550 MHz`.

`results/constraint_sweep/20260629_090133`

```bash
.venv/bin/python examples/constraint_sweep.py \
  --workers 8 \
  --level1-pop-size 40 \
  --level1-generations 3 \
  --level2-pop-size 24 \
  --level2-generations 4 \
  --max-architectures 40 \
  --seed 1
```

Options: `workloads/alexnet.onnx`, `configs/ip_pool_synthetic_28nm.yaml`, Level 2 `nsga2`, baseline frequency `700 MHz`.

`results/constraint_sweep/20260629_205508`

```bash
.venv/bin/python examples/constraint_sweep.py \
  --workers 8 \
  --level1-pop-size 40 \
  --level1-generations 3 \
  --level2-pop-size 24 \
  --level2-generations 4 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 40 \
  --seed 1
```

Options: `workloads/alexnet.onnx`, `configs/ip_pool_synthetic_28nm.yaml`, baseline frequency `700 MHz`.

`results/constraint_sweep/20260702_092220`

```bash
.venv/bin/python examples/constraint_sweep.py \
  --workers 8 \
  --level1-pop-size 80 \
  --level1-generations 6 \
  --level2-pop-size 80 \
  --level2-generations 8 \
  --level2-strategy nsga2 \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 80 \
  --seed 42
```

Options: `workloads/alexnet.onnx`, `configs/ip_pool_synthetic_28nm.yaml`, baseline frequency `700 MHz`.

All constraint sweeps used these case options:

| case | max_area_mm2 | max_power_w | min_frequency_mhz |
|---|---:|---:|---:|
| baseline | 0.40 | 0.12 | run baseline |
| strict_area | lower area | 0.12 | run baseline |
| strict_power | 0.40 | lower power | run baseline |
| strict_frequency | 0.40 | 0.12 | higher frequency |
| relaxed_area | 1.50 | 0.12 | run baseline |
| relaxed_power | 0.40 | 0.50 | run baseline |
| relaxed_frequency | 0.40 | 0.12 | 550 |

### Objective Sweeps

All objective sweeps used `workloads/alexnet.onnx` and `configs/ip_pool_synthetic_28nm.yaml` unless noted otherwise.

`results/objective_sweep/20260705_134857`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 8 \
  --level1-pop-size 16 \
  --level1-generations 2 \
  --level2-pop-size 24 \
  --level2-generations 3 \
  --max-architectures 6 \
  --seed 1
```

Options: 9-case cross product, Level 2 `nsga2`, `max_area_mm2=0.40`, `max_power_w=0.12`, `min_frequency_mhz=700`.

`results/objective_sweep/20260705_174727`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 8 \
  --level1-pop-size 16 \
  --level1-generations 3 \
  --level2-pop-size 24 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --min-frequency-mhz 550 \
  --dry-run
```

Options: paired objective cases, dry-run manifest only.

`results/objective_sweep_paired/exhaustive/20260705_174739`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 8 \
  --level1-pop-size 16 \
  --level1-generations 3 \
  --level2-pop-size 24 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --max-area-mm2 0.4 \
  --max-power-w 0.12 \
  --min-frequency-mhz 550 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/exhaustive
```

Options: paired objective cases, interrupted during first case before manifest.

`results/objective_sweep_paired/exhaustive_small/20260705_175618`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 4 \
  --level1-pop-size 4 \
  --level1-generations 3 \
  --level2-pop-size 16 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 8 \
  --max-area-mm2 0.4 \
  --max-power-w 0.12 \
  --min-frequency-mhz 550 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/exhaustive_small
```

Options: paired objective cases, interrupted during first case before manifest.

`results/objective_sweep_paired/exhaustive_fixed/20260705_180352`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 4 \
  --level1-pop-size 4 \
  --level1-generations 3 \
  --level2-pop-size 16 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 8 \
  --max-area-mm2 0.4 \
  --max-power-w 0.12 \
  --min-frequency-mhz 550 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/exhaustive_fixed
```

Options: paired objective cases.

`results/objective_sweep_paired/nsga2_fixed/20260705_183604`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 4 \
  --level1-pop-size 4 \
  --level1-generations 3 \
  --level2-pop-size 16 \
  --level2-generations 3 \
  --level2-strategy nsga2 \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 8 \
  --max-area-mm2 0.4 \
  --max-power-w 0.12 \
  --min-frequency-mhz 550 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/nsga2_fixed
```

Options: paired objective cases.

`results/objective_sweep_paired/relaxed_all_exhaustive/20260705_225108`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workers 4 \
  --level1-pop-size 4 \
  --level1-generations 3 \
  --level2-pop-size 16 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 8 \
  --max-area-mm2 10 \
  --max-power-w 10 \
  --min-frequency-mhz 400 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/relaxed_all_exhaustive
```

Options: paired objective cases, cancelled after `power`, `area`, and `performance`.

`results/objective_sweep_paired/no_constraints_exhaustive_big_l1/20260705_232223`

```bash
.venv/bin/python examples/objective_sweep.py \
  --no-constraints \
  --workers 4 \
  --level1-pop-size 8 \
  --level1-generations 6 \
  --level2-pop-size 16 \
  --level2-generations 3 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 8 \
  --seed 1 \
  --results-dir results/objective_sweep_paired/no_constraints_exhaustive_big_l1
```

Options: paired objective cases, no physical user constraints, Level 1 budget 4x larger than the previous `4 x 3` paired run.

`results/objective_sweep_paired/workload_power_dram_long/20260719_133646`

```bash
.venv/bin/python examples/objective_sweep.py \
  --workload workloads/alexnet.onnx \
  --ip-pool configs/ip_pool_synthetic_28nm.yaml \
  --workers 10 \
  --level1-pop-size 16 \
  --level1-generations 4 \
  --level2-pop-size 32 \
  --level2-generations 4 \
  --max-architectures 8 \
  --max-area-mm2 100 \
  --max-power-w 100 \
  --min-frequency-mhz 500 \
  --seed 42 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --results-dir results/objective_sweep_paired/workload_power_dram_long
```

Options: paired objective cases, synthetic ZigZag DRAM access energy included, permissive constraints, long Level 1, exhaustive Level 2.

`results/objective_sweep_fp16_resnet18_first_layer/20260812_082135`

```bash
.venv/bin/python -u examples/objective_sweep.py \
  --workload workloads/resnet18_first_layer.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir results/objective_sweep_fp16_resnet18_first_layer \
  --level1-pop-size 80 \
  --level1-generations 8 \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --no-constraints \
  --seed 1
```

Options: paired objective cases, first ResNet18 layer, two FP16 PEs,
no user constraints, large Level 1 and exhaustive Level 2. The workload
is FP32 and the format mismatch is retained by policy.

`results/objective_sweep_fp32_resnet18_first_layer/20260813_222734`

```bash
run_base=results/objective_sweep_fp32_resnet18_first_layer
.venv/bin/python -u examples/objective_sweep.py \
  --workload workloads/resnet18_first_layer.onnx \
  --ip-pool configs/ip_pool_fp32_65nm.yaml \
  --results-dir "$run_base" \
  --level1-pop-size 80 \
  --level1-generations 8 \
  --workers 16 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --no-constraints \
  --seed 1
```

Options: paired objective cases, first ResNet18 layer, three FP32 PEs,
all 15 memories with valid PPA, no user constraints, large Level 1 and
exhaustive Level 2. Workload and PEs both use `float32/32`.

## Latest Useful Reports

- `results/sweep_reports/constraint_sweep_20260702_092220.md`
- `results/sweep_reports/objective_sweep_20260705_134857.md`
- `results/sweep_reports/objective_sweep_paired_20260705.md`
- `results/sweep_reports/objective_sweep_workload_power_20260718.md`
- `results/sweep_reports/objective_sweep_workload_power_dram_20260719.md`
- `results/sweep_reports/level2_no_constraints_replay_20260802.md`
- `results/sweep_reports/objective_sweep_fp16_resnet18_first_layer_20260812.md`
- `results/sweep_reports/objective_sweep_fp32_resnet18_20260813.md`
- `results/objective_sweep_paired/no_constraints_exhaustive_big_l1/20260705_232223/CONCLUSIONS.md`
- `results/objective_sweep_paired/no_constraints_exhaustive_big_l1/20260705_232223/POWER_OBJECTIVE_DIAGNOSTIC.md`
