# TALOS

TALOS is a two-level design-space exploration tool for DNN accelerators. It combines workload evaluation with ZigZag, multi-objective search with NSGA-II, and physical IP selection from a characterized component pool.

The project is a research prototype. Characterized IP values are
post-synthesis research data, not sign-off silicon characterization;
the example pools and DRAM proxy remain synthetic.

## What TALOS does

1. **Level 1 — abstract architecture exploration**
   - Searches a discrete seven-gene architecture space.
   - Evaluates each candidate against an ONNX workload with ZigZag.
   - Optimizes latency, energy, area, or compound objectives with pymoo NSGA-II.
2. **Level 2 — physical implementation exploration**
   - Converts each Level 1 candidate into an abstract component graph.
   - Selects compatible PE, register-file, and global-buffer IP blocks.
   - Uses either NSGA-II or exhaustive enumeration.
   - Evaluates physical timing separately from workload latency and throughput.
   - Validates that selected IPs can sustain the mapping, then reports average power and energy per inference.
3. **Reporting and sweeps**
   - Exports Level 1, Level 2, and combined CSV reports.
   - Supports area, power, latency, and frequency constraints.
   - Includes ready-to-run constraint and objective sweeps.

## Installation

Python 3.13 is the tested version; the repository `shell.nix` pins `python313`
for the same reason. `git` must be on `PATH` because `requirements.txt` pins
`nsga-2` to a GitHub commit, and the CACTI binary bundled with ZigZag needs a
working `libstdc++` at runtime.

```bash
git clone https://github.com/ThaChoppahIsLookinSharp/talos.git
cd talos

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TALOS is not installed as a package: `python -m talos` and the `examples/`
scripts resolve their imports from the repository root, so run every command in
this README from there.

On NixOS, `nix-shell` reads `shell.nix`, which provides Python 3.13, exposes
`libstdc++` through `LD_LIBRARY_PATH`, and activates `.venv`. A venv created
from `/run/current-system/sw/bin/python3` follows system upgrades; if
`python -m talos` starts failing with `ModuleNotFoundError: No module named
'yaml'` after one, the interpreter has changed under the venv. Delete `.venv`
and recreate it with the commands above.

### Check the installation

```bash
python -m unittest discover -s tests   # 114 tests, about 10 s
python -m talos --help
python examples/full_flow_example.py --help
```

## Quick start

All commands below are run from the repository root. They use the bundled
AlexNet ONNX model and the synthetic 65 nm pool. Other ONNX models live under
`workloads/` and are described in `workloads/workloads.yaml`; the other pools
are listed in [IP pool format](#ip-pool-format).

### Smoke-test one Level 1 genome

```bash
python -m talos --ip-pool configs/ip_pool_synthetic_65nm.yaml
```

This characterizes the Level 1 CACTI and area tables once, evaluates a single
fixed genome with ZigZag and prints every Level 1 objective for it. Expect
a few minutes on a laptop; almost all of it is the one-off CACTI
characterization, which every later command repeats. Use
`--debug` to show ZigZag output and print the generated mapping and
accelerator YAML.

### Run Level 1 NSGA-II

```bash
python -m talos --ga \
  --workload workloads/alexnet.onnx \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --objectives latency energy area \
  --pop-size 12 \
  --generations 4 \
  --workers 4 \
  --seed 1 \
  --zigzag-lpf-limit 1 \
  --zigzag-spatial-mappings 1 \
  --results-dir results/level1_demo
```

`--workers` parallelizes Level 1 candidate evaluation. Larger ZigZag LPF and spatial-mapping limits explore more mappings but increase runtime.

### Run the complete Level 1 → Level 2 flow

```bash
python examples/full_flow_example.py \
  --workload workloads/alexnet.onnx \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --level1-pop-size 12 \
  --level1-generations 3 \
  --level2-objectives area energy workload_latency_s \
  --level2-strategy nsga2 \
  --level2-pop-size 24 \
  --level2-generations 4 \
  --max-architectures 4 \
  --workers 4 \
  --seed 1 \
  --results-dir results/full_flow_demo
```

This is the main entry point when Level 2 metrics or physical constraints are needed.
Level 1 always screens the fixed energy–latency–area Pareto set in this flow;
`--level1-objectives` is still accepted but deprecated. Use
`examples/full_flow_legacy_objectives.py` with the same arguments to force a
different Level 1 objective set.

### Run with physical constraints

```bash
python examples/full_flow_example.py \
  --workload workloads/alexnet.onnx \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --level2-objectives area energy workload_latency_s \
  --max-latency-cycles 100000000 \
  --max-area-mm2 6.0 \
  --max-power-w 1.2 \
  --min-frequency-mhz 600 \
  --level1-pop-size 24 \
  --level1-generations 3 \
  --level2-pop-size 24 \
  --level2-generations 4 \
  --max-architectures 8 \
  --workers 4 \
  --results-dir results/constrained_demo
```

### Use exhaustive Level 2 selection

```bash
python examples/full_flow_example.py \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 4 \
  --workers 4 \
  --results-dir results/exhaustive_demo
```

The exhaustive runner raises before enumeration if a search would exceed the configured combination limit; the full-flow example reports that failure and continues with the next architecture.

### Reuse Level 1 across several Level 2 runs

Level 1 plus its ZigZag profiles is the expensive half of the flow. Run it
once, then feed the handoff to as many Level 2 configurations as needed:

```bash
python examples/full_flow_example.py \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --level1-pop-size 12 \
  --level1-generations 3 \
  --max-architectures 4 \
  --workers 4 \
  --level1-only \
  --results-dir results/level1_handoff_demo

python examples/full_flow_example.py \
  --ip-pool configs/ip_pool_synthetic_65nm.yaml \
  --level1-handoff results/level1_handoff_demo/level1_handoff.json \
  --level2-objectives area energy \
  --level2-strategy exhaustive \
  --results-dir results/level2_from_handoff
```

`--level1-only` stops after writing `level1_handoff.json` (override the path
with `--level1-handoff-output`). `--level1-handoff` skips the Level 1 search and
the ZigZag profiling and reuses the stored candidates and profiles; only the
CACTI calibration is repeated.

### Inspect Level 2 independently

```bash
python examples/level2_smoke_test.py
python examples/level2_runner_example.py
```

The smoke test shows the dynamic Level 2 genes and evaluates their default implementation, both for a decoded Level 1 genome and for `configs/zigzag_accelerator_example.yaml`. The runner executes a small Level 2 NSGA-II search.

## Architecture

```mermaid
flowchart LR
    A[ONNX workload] --> B[Level 1 genome]
    B --> C[ZigZag evaluation]
    C --> D[pymoo NSGA-II]
    D --> E[Pareto plus feasible final-population architectures]
    E --> F[Abstract component graph]
    G[Characterized IP pool YAML] --> H[Level 2 genome]
    F --> H
    C --> I[Per-layer mapping profile]
    I --> J[Workload performance, capacity, power and energy]
    H --> K{Level 2 strategy}
    K -->|NSGA-II| L[Physical implementations]
    K -->|Exhaustive| L
    J --> L
    L --> M[CSV reports and sweep manifests]
```

The main modules are:

- `talos/architecture`: Level 1 genome, abstract components, and importers.
- `talos/evaluation`: ZigZag integration, objective adapter, and workload activity extraction.
- `talos/ga`: Level 1 NSGA-II runners and CSV export.
- `talos/ip`: IP characterization models and YAML-backed IP pools.
- `talos/level2`: physical genome, evaluator, NSGA-II/exhaustive runners, and power model.
- `examples`: complete flows, smoke tests, and long sweeps.
- `select_talos_winners.py`: post-processes an objective-sweep directory into per-target winners.

### Level 1 genome

Each gene is an integer index into a fixed catalog:

| Gene | Available values |
| --- | --- |
| `pe_x_code` | 4, 8, 16, 32 |
| `pe_y_code` | 4, 8, 16, 32 |
| `rf_size_code` | 64, 128, 256, 512, 1024, 2048 bits |
| `rf_bw_code` | 8, 16, 32, 64, 128, 256 bits/cycle |
| `gb_size_code` | 8192, 16384, 32768, 65536, 131072 bits |
| `gb_bw_code` | 64, 128, 256, 512, 1024 bits/cycle |
| `gb_served_dims_code` | none, D1, D2, or D1+D2 |

DRAM is a fixed platform IP rather than a search gene. If the pool has
one `type: dram` entry, TALOS gives it priority. Otherwise it uses
one shared 512-bit platform proxy for every precision. The proxy
allows one access per cycle and uses `p_idle_w = 0.02` for refresh,
PHY and standby.

Before starting any Level 1 workers, TALOS runs the CACTI binary bundled with
ZigZag at the `technology_nm` declared by the selected IP pool. It characterizes
all 25 `gb_size_code × gb_bw_code`
combinations and one fixed 128 KiB, 16-bit SRAM reference. CACTI is copied to a
temporary directory once per flow and is never run by a candidate, layer, or
worker. CACTI requires a physical depth of at least 32 words, so the logical
1 KiB/512-bit, 1 KiB/1024-bit, and 2 KiB/1024-bit candidates are modeled with
2 KiB, 4 KiB, and 4 KiB physical macros respectively. ZigZag still sees their
original logical capacities, and the overprovisioning is recorded in
`energy_calibration.json`.

Level 1 also precomputes 36 PE-plus-RF area costs and 25 global-buffer area
costs from the selected characterized IP pool. Candidate evaluation then uses
two table lookups, counting RFs embedded in composite PEs only once and keeping
the global-buffer replication implied by `gb_served_dims`.

### Level 1 to Level 2

A decoded Level 1 architecture becomes:

- one `pe_array`, with `pe_x * pe_y` processing elements;
- `rf_i1`, `rf_i2`, and `rf_o`, replicated per PE;
- one or more `gb` instances, with replication determined by the array dimensions not served by the global buffer.

The Level 2 genome has one gene per abstract component. Each gene selects an IP from the pool after filtering by:

- component type;
- minimum capacity;
- minimum bandwidth.

PE entries may declare that their characterized area and timing already include
one or more RF roles. Selecting one of these composite PEs fixes the covered RF
genes to the referenced RF IPs: the RFs remain visible for capacity, bandwidth,
ZigZag accesses, and power, but their area, delay, throughput, and `fmax` are not
counted a second time.

The exhaustive strategy evaluates every compatible combination up to `--level2-exhaustive-max-combinations`. After invalid and constrained candidates are removed, a single objective is ranked directly by its minimum. For multiple objectives, it uses the augmented Tchebycheff score `r_i(x) = ln(f_i(x) / f_i_min)` and `S(x) = 100 * [max_i r_i(x) + 0.05 * sum_i r_i(x)]`. This measures relative multiplicative degradation from the ideal, prioritizes the worst degradation, and uses the small sum term to distinguish similar candidates. The exhaustive CSV's ideal is local to one Level-1 architecture; the full-flow summary recomputes its ideal across every feasible physical implementation from every architecture in that invocation because local scores are not globally comparable. Ranking only orders the full feasible set: it adds no pruning or Pareto filtering. NSGA-II is preferable when the Cartesian product is large.

The full flow sends Pareto solutions first and can fill `--max-architectures` with distinct feasible individuals from the final Level 1 population. The objective sweep starts with three architectures per objective case.
When physical constraints are present, it also screens each Level 1 candidate
for at least one feasible Level 2 combination before consuming an architecture
slot. Spaces above `--level2-exhaustive-max-combinations` are left to the
selected Level 2 strategy instead of being rejected by the prefilter.

## Objectives and constraints

### Level 1 objectives

| Objective | Meaning |
| --- | --- |
| `latency` | ZigZag workload latency |
| `energy` | ZigZag workload energy |
| `area` | Minimum compatible characterized IP area in mm² |
| `edp` | energy × latency |
| `eap` | energy × area |
| `alp` | area × latency |

The Level 1 area is a pool-specific lower bound. It selects the cheapest
compatible PE/RF bundle and global-buffer macro, including composite PEs, while
Level 2 remains authoritative for complete physical combinations and
constraints.

The Level 1 objectives also select ZigZag's mapping criterion. Energy objectives
use `energy`, latency objectives use `latency`, and a mix of both uses `EDP`;
area alone falls back to `EDP`. The selected criterion is reused when the final
candidate profile is generated for Level 2. `python -m talos --ga` honors the
requested `--objectives`; `examples/full_flow_example.py` always screens the
fixed energy–latency–area set, so its mapping criterion is `EDP` and its
`--level1-objectives` option is deprecated.

Level 1 energy uses CACTI together with the 65 nm, 16-bit ratios from Table IV
of [Eyeriss](https://www.cs.cmu.edu/~15740-f20/papers/isca16-chen-eyeriss.pdf):

```text
MAC = 1×, RF = 1×, GB = 6×, DRAM = 200×

E_GB_reference = (E_CACTI_read + E_CACTI_write) / 2
E_MAC = E_GB_reference / 6
E_RF_access = E_MAC × (rf_bandwidth_bits / 16)
E_DRAM_access = 200 × E_MAC × (dram_bandwidth_bits / 16)
```

Each candidate GB instead uses its own CACTI read and write energies directly;
changing a GB can therefore change its mapping and memory energy without
changing `E_MAC`, RF energy, or DRAM energy. The derived `E_MAC` is ZigZag's
dynamic energy per MAC operation, not PE leakage or clock power.

### Level 2 objectives

| Objective | Meaning |
| --- | --- |
| `area` | Sum of selected IP area × instance count |
| `energy` | Total energy in joules for one workload inference |
| `power` | Average workload power in watts |
| `workload_latency_s` | Sequential batch-1 inference latency from ZigZag mapping cycles |
| `delay` | Legacy physical objective: maximum selected-IP delay |
| `inv_throughput` | Legacy local-IP objective; not inference throughput |

Level 2 does not reinterpret an IP delay or local throughput as workload
performance. It preserves the cycles selected by ZigZag and computes:

```text
workload_cycles_per_inference = Σ layer_cycles_mapping
workload_latency_s =
    workload_cycles_per_inference / (reference_frequency_mhz × 1e6)
workload_throughput_ips = 1 / workload_latency_s
```

This assumes batch 1, sequential layers, and no pipeline between inferences.
At the same reference frequency, every valid implementation of the same
mapping has the same workload latency. A faster physical `fmax_mhz` only
increases timing margin; TALOS does not automatically run that candidate faster.

For workload-aware exploration, TALOS reuses the mapping selected by ZigZag. For each layer it extracts the latency, spatially used PEs, and physical accesses at every memory level. PE power directly distinguishes mapped active PEs from the remaining idle PEs. On-chip memory and DRAM power are interpolated from access utilization:

```text
workload_cycles_per_inference = Σ layer_cycles_mapping
workload_latency_s = workload_cycles_per_inference / (reference_frequency_mhz × 1e6)
workload_throughput_ips = 1 / workload_latency_s

P_PE = active_PEs × p_active_w + idle_PEs × p_idle_w
P_memory = instances × (p_idle_w + utilization × (p_active_w - p_idle_w))
u_DRAM = DRAM_accesses / (layer_cycles × accesses_per_cycle)
P_DRAM = p_idle_w + u_DRAM × (p_active_w - p_idle_w)
layer_time = layer_cycles / reference_frequency
E_inference = Σ((P_PE + P_memories + P_DRAM) × layer_time)
P_average = E_inference / inference_time
```

For example, if ZigZag maps a layer onto 16 of `N` PEs, the PE term is exactly `16 × p_active_w + (N - 16) × p_idle_w`. Register-file, global-buffer, and DRAM utilization comes from the accesses in the ZigZag mapping. DRAM remains external to Level 2 on-chip area and on-chip critical timing, but its power and energy are included and its own `fmax_mhz` must still reach the reference frequency. Its `p_active_w` means continuous transfers at the declared bus width and `accesses_per_cycle`; `p_idle_w` means no transfers.

For a synthetic DRAM, TALOS reconstructs `p_active_w` from the Level 1
Eyeriss access energy:

```text
p_active_w = p_idle_w
    + E_DRAM_access
    × reference_frequency
    × accesses_per_cycle
```

This keeps the dynamic energy per DRAM access identical in Level 1 and
Level 2 while preserving idle power. When the pool contains a DRAM,
its characterization takes priority. Otherwise the fixed platform
proxy inherits the pool operating point and uses the global 0.02 W
idle value. This value is a synthetic system-level assumption, not a
measured DRAM characterization. PE power in Level 2 always remains the
IP pool or Genus `p_idle_w`/`p_active_w`; it is never replaced by the
Level 1 MAC proxy.

Memory accesses are normalized from the abstract Level 1 port width to the selected IP width. Every valid combination uses one common characterized `reference_frequency_mhz` and `reference_voltage_v`. Power values are used exactly as characterized, without frequency scaling or interpolation. `physical_fmax_mhz` is the minimum selected-IP `fmax_mhz`; it must meet the reference frequency but never becomes the operating frequency. `physical_critical_delay` and `timing_margin_mhz = physical_fmax_mhz - reference_frequency_mhz` are reported separately.

TALOS infers activation, weight and output representations from the
ONNX tensor types before calling ZigZag. The visible output type is
also used for the accumulator, so a FLOAT model uses 32-bit operands
and outputs while an INT8 model uses 8-bit operands and outputs. TALOS
does not quantize a model or fall back to ZigZag's implicit 8-bit
defaults. The Level 1 MAC energy remains the configured calibration
and is not scaled by precision.

Level 2 rejects a combination when a selected IP is characterized at
an incompatible operating point, misses the reference frequency,
cannot provide the mapped PE MACs/cycle, or cannot sustain the mapped
memory accesses/cycle. It preserves the ZigZag cycles rather than
inserting stalls or remapping.

For each layer, Level 2 also checks that mapped MACs/cycle fit the active PEs'
`macs_per_cycle` and that normalized memory accesses/cycle fit the
selected instances and `accesses_per_cycle`. A PE format or precision
mismatch produces one aggregated warning but does not remove the
candidate from the exploration.
Current ZigZag profiles expose aggregate physical accesses per memory level, so
read/write and operand contention are intentionally checked as one shared rate.

For a composite PE, `p_idle_w` includes the idle baseline of its covered RFs and
`p_active_w - p_idle_w` is the compute-only active increment. TALOS therefore
adds only each covered RF's access-dependent increment:

```text
P_composite =
    PE_count × PE_idle
    + active_PEs × (PE_active - PE_idle)
    + Σ covered_RF_count × utilization × (RF_active - RF_idle)
```

This contract avoids counting the covered RF idle power twice. Standalone RFs
continue to use the full memory formula above.

The `energy` and `power` objectives use the same model: `energy` minimizes joules per inference, while `power` minimizes time-weighted average watts. Physical timing is reported separately as `physical_critical_delay`, `physical_fmax_mhz`, and `timing_margin_mhz`.

### User constraints

| Option | Stage | Condition |
| --- | --- | --- |
| `--max-latency-cycles` | Level 1 | workload latency must not exceed the limit |
| `--max-area-mm2` | Level 2 | physical IP area must not exceed the limit |
| `--max-power-w` | Level 2 | average workload power must not exceed the limit |
| `--min-frequency-mhz` | Level 2 | implementation `fmax` capability must meet the minimum |

Area, power, and frequency constraints require the full Level 1 → Level 2 flow. `python -m talos` only accepts the Level 1 latency constraint.

## Long sweeps

The sweep scripts create a timestamped directory and a `manifest.csv` containing every command, case, result directory, summary path, and return code.

Preview commands before starting a long run:

```bash
python examples/constraint_sweep.py --dry-run
python examples/objective_sweep.py --dry-run
```

Run the seven constraint cases:

```bash
python examples/constraint_sweep.py \
  --workers 8 \
  --level2-strategy exhaustive \
  --results-dir results/constraint_sweep
```

Run all seven objective combinations:

```bash
python examples/objective_sweep.py \
  --workers 8 \
  --level2-strategy exhaustive \
  --max-area-mm2 6.0 \
  --max-power-w 1.2 \
  --min-frequency-mhz 600 \
  --results-dir results/objective_sweep
```

These objective-sweep defaults are calibrated to leave a useful feasible region in the included synthetic pool; they are not silicon design targets. The constraint sweep intentionally uses much tighter values as regression cases. Use `--no-constraints` to compare objectives without area, power, or frequency limits.

### Select sweep winners

```bash
python select_talos_winners.py results/objective_sweep/<timestamp> --output winners.csv
```

The script reads every `full_flow_summary.csv` under the sweep directory,
keeps feasible rows, and picks one winner per objective case. Single
objectives take the minimum; multi-objective cases use
`S(x) = 100 · sqrt(Σ_i ln(f_i(x) / f_i_min)²)` with the ideal computed over
all feasible rows of the sweep. This is a different score from the augmented
Tchebycheff ranking inside the exhaustive runner, and the per-CSV
`augmented_tchebycheff_score` columns are ignored. `--energy-mode onchip`
subtracts DRAM energy before ranking.

### Latest characterized sweeps

The FP16 and FP32 sweeps used the first ResNet18 layer. Both completed
all seven objective cases with 45600 valid rows and 42336 unique
physical implementations.

| best observed | FP16 | FP32 |
|---|---:|---:|
| area | 0.451629 mm2 | 0.573476 mm2 |
| average power | 0.293842 W | 0.329406 W |
| inference energy | 0.0208520 J | 0.0243084 J |
| throughput | 757.93 inf./s | 757.93 inf./s |

The FP32 run matches the workload representation. The FP16 comparison
is descriptive because TALOS retained the format mismatch by policy.
Both full reports were written under `results/sweep_reports`, which Git
ignores; the numbers above are kept here for reference only.

## IP pool format

IP pools are YAML files with one technology for the whole pool and an `ips`
list:

```yaml
technology_nm: 65
ips:
  # PE, RF and GB entries; DRAM is optional
```

`characterizations/ip_catalog_65nm.yaml` is the source of truth for
the 27 characterized 65 nm IPs. It preserves the characterization
output, including areas in um2 and the original numeric-format labels.
The runtime pools convert areas to mm2 and normalize those labels
to `int8`, `uint8`, `float16`, or `float32`. The integer characterized
pool includes the shared 512-bit platform DRAM proxy; the FP16 and FP32
pools let the full flow inject the same proxy.

The six included pools are:

- `configs/ip_pool_example.yaml`: illustrative values for small examples.
- `configs/ip_pool_synthetic_65nm.yaml`: synthetic values used by tests and sweeps; they are not foundry characterization.
- `configs/ip_pool_characterized_65nm.yaml`: post-synthesis TSMC65 PE
  characterization for all integer PEs and all 15 memories, plus the shared
  512-bit platform DRAM proxy.
- `configs/ip_pool_int16_65nm.yaml`: three INT16 PEs characterized with
  Genus at 200 MHz and 1.2 V, the 15 memories and the shared platform DRAM
  proxy. These PEs are not part of `ip_catalog_65nm.yaml`.
- `configs/ip_pool_fp16_65nm.yaml`: the two FP16 PEs and the RF and GB
  characterization from the supplied 65 nm pool.
- `configs/ip_pool_fp32_65nm.yaml`: the three FP32 PEs and all 15
  memories with valid numeric PPA from the supplied characterization.

The synthetic pool contains 2 PE, 4 register-file, 7 global-buffer
choices and one fixed DRAM used by direct Level 2 tests. It covers
every Level 1 genome and produces at most 896 compatible Level 2
combinations for one architecture, so exhaustive selection is
normally preferable. Add variants only from a coherent
characterization flow rather than inventing extra points for
population size.
Its 65 nm name matches the Level 1 CACTI technology; all pool PPA values remain
explicit approximations rather than foundry characterization.

A minimal PE entry looks like this:

```yaml
ips:
  - id: pe_mac_8b_fast
    type: pe
    area: 0.0013
    throughput: 2.0
    delay: 0.6
    fmax_mhz: 900.0
    metadata:
      precision_bits: 8
      numeric_format: int8
      macs_per_cycle: 1
    power_model:
      source: synthetic
      activity_method: vectorless
      reference_frequency_mhz: 500.0
      p_idle_w: 0.00015
      p_active_w: 0.00075
      voltage_v: 1.0
      temperature_c: 25.0
      corner: tt
```

`metadata.macs_per_cycle` validates the frozen mapping.
`precision_bits` and `numeric_format` describe the PE and are compared
with the inferred ONNX representation. A mismatch is reported but
remains eligible. The generic `throughput` field is retained for
legacy local-IP objectives and is not used as inference throughput.

If that PE characterization already includes RF RTL, declare the covered roles
and reference ordinary RF entries from the same pool:

```yaml
ips:
  - id: pe_tile_with_rfs
    type: pe
    area: 0.0032
    throughput: 2.0
    delay: 0.7
    fmax_mhz: 800.0
    metadata:
      precision_bits: 8
      numeric_format: int8
      macs_per_cycle: 1
    included_rfs:
      rf_i1: rf_512b_64b
      rf_i2: rf_512b_64b
      rf_o: rf_512b_64b
    included_rf_power_mode: parent_idle_baseline
    power_model:
      source: synthetic
      activity_method: vectorless
      reference_frequency_mhz: 500.0
      p_idle_w: 0.00045
      p_active_w: 0.00105
      voltage_v: 1.0
      temperature_c: 25.0
      corner: tt

  - id: rf_512b_64b
    type: register_file
    area: 0.0002
    throughput: 1.0
    delay: 0.2
    fmax_mhz: 900.0
    capacity_bits: 512
    bandwidth_bits: 64
    metadata:
      accesses_per_cycle: 1
    power_model:
      source: synthetic
      activity_method: vectorless
      reference_frequency_mhz: 500.0
      p_idle_w: 0.00002
      p_active_w: 0.00008
      voltage_v: 1.0
      temperature_c: 25.0
      corner: tt
```

`included_rfs` currently accepts `rf_i1`, `rf_i2`, and `rf_o`. Partial coverage
is allowed; omitted roles remain standalone IP selections. Referenced RFs must
exist in the pool and satisfy the corresponding abstract capacity and bandwidth.
`included_rf_power_mode: parent_idle_baseline` is mandatory whenever
`included_rfs` is non-empty. Composite selection currently requires exactly one
abstract PE component (`pe_array`).

Memory IPs additionally use `capacity_bits`, `bandwidth_bits`, and `metadata.accesses_per_cycle`.

The pool may contain one DRAM characterization to override the fixed
platform proxy:

```yaml
  - id: dram_ddr_512b
    type: dram
    area: 0.0
    throughput: 1.0
    delay: 20.0
    fmax_mhz: 500.0
    bandwidth_bits: 512
    metadata:
      accesses_per_cycle: 1
    power_model:
      source: synthetic
      activity_method: access_rate
      reference_frequency_mhz: 500.0
      p_idle_w: 0.02
      p_active_w: 4.5
      voltage_v: 1.0
      temperature_c: 25.0
      corner: tt
```

Without this entry, TALOS creates a 512-bit proxy with one access per
cycle. Its dynamic power comes from the Level 1 Eyeriss ratio and its
0.02 W idle power represents refresh, PHY and standby. Both are
system-level assumptions. Replace them with a measured point for the
target memory and access pattern; vendor tools such as the
[Micron DRAM Power Calculator](https://www.micron.com/sales-support/design-tools/dram-power-calculator)
are suitable calibration sources.

`p_idle_w` and `p_active_w` are per-instance power values used directly by the estimator. `reference_frequency_mhz` is both their characterization point and the operating frequency used to convert workload cycles into seconds. Every IP selected in one combination must use the same reference frequency and must meet it with its `fmax_mhz`; incompatible candidates are discarded rather than aborting the whole pool. No frequency scaling is applied. `voltage_v` records the characterization voltage and is checked for compatibility; it is not added or multiplied into the energy calculation. The included values are synthetic, but real values can be obtained from two Genus power scenarios: clocked idle and representative active operation.

The current YAML schema represents one operating point per IP, so its top-level
area, delay, and `fmax_mhz` are assumed to belong to that same characterized
voltage and corner. TALOS does not interpolate operating points.

Workload-aware exploration requires each selected IP to provide:

- `fmax_mhz`;
- `p_idle_w` and `p_active_w`;
- `voltage_v`;
- `macs_per_cycle` for PEs;
- `precision_bits` and `numeric_format` for PE compatibility warnings;
- `accesses_per_cycle` for memories;
- compatible voltage, temperature, and process corner.

## Results

The complete flow writes:

```text
results/full_flow_demo/
├── energy_calibration.json
├── level1/
│   ├── energy_calibration.json
│   └── pymoo_nsga2_results_<timestamp>.csv
├── level1_profiles/
├── level1_handoff.json          # only with --level1-only
├── level2_arch_<index>/
│   └── level2_<nsga2|exhaustive>_results.csv
├── winner/
│   ├── architecture.json        # first row of the summary
│   └── zigzag/                  # its ZigZag mapping dump
└── full_flow_summary.csv
```

The combined summary contains columns for:

- raw and discretized Level 1 genomes;
- decoded architecture parameters;
- selected Level 2 IPs and RF roles covered by a composite PE;
- objective values;
- `layer_cycles_mapping`, `workload_cycles_per_inference`, `workload_latency_s`, and `workload_throughput_ips`;
- `reference_frequency_mhz` and `reference_voltage_v`;
- physical area, power, energy, critical delay, `fmax`, and timing margin;
- DRAM accesses and total DRAM energy;
- constraint status and violations;
- estimated inferences per second;
- paths to the detailed Level 1 and Level 2 CSVs.

The complete Level 1 → Level 2 flow always generates one mapping profile per
selected Level 1 architecture, so workload performance, power, and energy remain
comparable even when they are not optimization objectives.

Generated `results/`, `outputs/`, and `.talos_zigzag/` directories are ignored by Git.

## Python API

Level 2 can also be driven directly:

```python
from talos.architecture import (
    abstract_accelerator_from_level1_config,
    decode_genome,
    default_genome,
)
from talos.ip import IPPool
from talos.level2 import run_level2

config = decode_genome(default_genome())
accelerator = abstract_accelerator_from_level1_config(config)
pool = IPPool.from_yaml("configs/ip_pool_synthetic_65nm.yaml")

result = run_level2(
    accelerator=accelerator,
    ip_pool=pool,
    objective_names=["area", "delay", "inv_throughput"],
    strategy="exhaustive",
    save_csv=False,
)

for solution in result.solutions[:3]:
    print(
        solution["selected_ips"],
        solution["area"],
        solution["physical_critical_delay"],
    )
```

For `energy`, `power`, or `workload_latency_s`, also pass the
`WorkloadActivityProfile` produced by the selected Level 1 ZigZag mapping.

## Development

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Useful help commands:

```bash
python -m talos --help
python examples/full_flow_example.py --help
python examples/constraint_sweep.py --help
python examples/objective_sweep.py --help
python select_talos_winners.py --help
```

## Current limitations

- The repository is a research prototype, not a calibrated PPA sign-off flow.
- The synthetic 65 nm pool exists for repeatable tests and exploration only.
- The MAC, RF, and DRAM costs are 65 nm, 16-bit Eyeriss proxies; MAC energy is not scaled for 8-bit operation.
- CACTI models the GB SRAM but not the FIFOs included in the Eyeriss GB ratio.
- Level 1 does not yet model NoC energy, MAC/RF leakage, or a persistent CACTI cache.
- Level 1 area excludes DRAM, NoC, wiring and arbitration because they are not
  selectable components in the current abstract accelerator graph.
- DRAM is a fixed platform characterization. A pool entry overrides
  the shared proxy; otherwise all precisions use the same 512-bit,
  0.02 W idle model. Its workload power and energy are included.
- All compatible candidate IPs in a workload-aware search must share one characterization frequency and PVT point; characterization tables and interpolation are not implemented.
- Memory capacity validation is aggregate per layer and memory level because the current profile does not retain independent read/write ports or operand contention.
- Composite IP modeling currently covers only a PE with embedded RF roles; generic nested IP bundles are intentionally unsupported.
- `delay` and `inv_throughput` are legacy local-IP objectives, not workload performance; use `workload_latency_s` for inference performance.
- Exhaustive Level 2 runtime grows as the product of compatible candidates per component.

## Main dependencies

- [ZigZag](https://github.com/KULeuven-MICAS/zigzag) for workload and mapping evaluation.
- [pymoo](https://github.com/anyoptimization/pymoo) for NSGA-II.
- ONNX, NumPy, pandas, PyYAML, matplotlib, and related scientific Python packages.

See `requirements.txt` for pinned versions.

## License

MIT. See `LICENSE`.
