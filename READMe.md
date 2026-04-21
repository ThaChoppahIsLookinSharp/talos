# TALOS
Welcome to my caffeine fueled thesis project. The idea is that this thinks of a proper
architecture for your DNN accelerator while you do other stuff.

---
## Overview

Long story short, you specify some parameters, this will try its best to
generate some nice architecture.

It is a "2 level" generator.

Level 1 currently uses a pymoo-based NSGA-II loop. Fitness is computed from ZigZag.
Results are a set of abstract architectures.

Level 2, using these abstract architectures.

---
## Installation

I use Python 3.13, but 3.11+ should be fine.

```bash
git clone https://github.com/ThaChoppahIsLookinSharp/talos
cd talos
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---
## Commands

### Main CLI

Show the main help:

```bash
python -m talos --help
```

Run the default smoke test:

```bash
python -m talos
```

Run the smoke test on a specific workload:

```bash
python -m talos --workload workloads/alexnet.onnx
```

Run the smoke test with evaluator debug enabled:

```bash
python -m talos --debug
```

Run the smoke test with experimental ZigZag automatic memory costs:

```bash
python -m talos --memory-cost-mode zigzag_auto
```

### GA Runs From The Main CLI

Run the pymoo-based GA with defaults:

```bash
python -m talos --ga
```

Run the pymoo-based GA with explicit settings:

```bash
python -m talos --ga \
  --objectives latency energy area \
  --memory-cost-mode manual \
  --pop-size 12 \
  --generations 4 \
  --workers 4 \
  --seed 1 \
  --zigzag-lpf-limit 1 \
  --zigzag-spatial-mappings 1 \
  --results-dir ./results
```

Run the GA without writing the CSV:

```bash
python -m talos --ga --no-save-csv
```

Run the GA with multiple workers:

```bash
python -m talos --ga --workers 4
```

Run the GA with a different seed:

```bash
python -m talos --ga --seed 7
```

Use experimental automatic memory cost extraction:

```bash
python -m talos --ga --memory-cost-mode zigzag_auto
```

Available flags through `python -m talos`:

- `--workload <path>`
- `--debug`
- `--ga`
- `--objectives <obj1> <obj2> ...`
- `--generations <int>`
- `--pop-size <int>`
- `--workers <int>`
- `--memory-cost-mode manual|zigzag_auto`
- `--zigzag-lpf-limit <int>`
- `--zigzag-spatial-mappings <int>`
- `--seed <int>`
- `--no-save-csv`
- `--results-dir <path>`

### Direct Module Entry Points

Run the pymoo runner module directly:

```bash
python -m talos.ga.pymoo_runner
```

Run the legacy NSGA-II runner module directly:

```bash
python -m talos.ga.nsga2_runner
```

Run the package entry point directly:

```bash
python -m talos
```

There is also a minimal placeholder CLI module in `talos/cli.py`, but the
real entry point currently used by the project is `python -m talos`.

### Tests

Run all current unit tests:

```bash
python -m unittest -q tests.test_area_semantics tests.test_memory_cost_modes
```

Run only the area semantics tests:

```bash
python -m unittest -q tests.test_area_semantics
```

Run only the memory cost mode tests:

```bash
python -m unittest -q tests.test_memory_cost_modes
```

### Inspection And Debug Scripts

Compare manual vs automatic memory cost YAML and evaluation behavior:

```bash
python tools/compare_memory_cost_modes.py
```

Inspect the real ZigZag `cme` object and report where area was found:

```bash
python tools/inspect_cme_area.py
```

### Python API

You can also call TALOS from Python directly:

```python
from pathlib import Path
from talos.ga.pymoo_runner import run_nsga2_pymoo

result = run_nsga2_pymoo(
    workload_path=str(Path("workloads/alexnet.onnx").resolve()),
    objective_names=["latency", "energy", "area"],
    pop_size=6,
    n_gen=2,
    seed=1,
    n_workers=1,
    memory_cost_mode="manual",
)
```

---
## Software Used

It is a work in progress. For now:

- ZigZag https://github.com/KULeuven-MICAS/zigzag
- pymoo https://github.com/anyoptimization/pymoo
- Legacy NSGA-II implementation kept in the repo: https://github.com/baopng/NSGA-II

## Status

Work in progress.
