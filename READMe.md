# TALOS
Welcome to my caffeine fueled thesis project. The idea is that this thinks of a proper
architecture for your DNN accelerator while you do other stuff.

---
## Overview

Long story short, you specify some parameters, this will try its best to
generate some nice architecture.

TALOS is organized into independent exploration stages:

- Level 1: abstract architecture exploration with ZigZag.
- Level 2: physical IP selection over an abstract accelerator.
- Pipeline: future orchestration of Level 1 and Level 2.

The code follows that structure:

- `talos.level1`: architecture genome, ZigZag evaluator, objective adapter, and Level 1 runner.
- `talos.level2`: abstract accelerator importers, physical IP pool, dynamic Level 2 genome, evaluator, problem, and runner.
- `talos.pipeline`: placeholder for hierarchical Level 1 -> Level 2 execution.
- `talos.cli`: command-line dispatcher used by `python -m talos`.

---

## Installation

I use python 13, but I guess +3.11 should be allright.

git clone [https://github.com/ThaChoppahIsLookinSharp/talos](https://github.com/ThaChoppahIsLookinSharp/talos)
cd talos

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

---

## Usage

Run a quick smoke test:

```
python -m talos smoke
```

Run Level 1 abstract architecture search:

```
python -m talos level1 \
   --workload workloads/alexnet.onnx \
   --objectives latency energy area \
   --pop-size 6 \
   --generations 2
```

Run Level 2 physical IP selection:

```
python -m talos level2 \
   --accelerator configs/zigzag_accelerator_example.yaml \
   --ip-pool configs/ip_pool_example.yaml \
   --objectives area power delay inv_throughput \
   --pop-size 6 \
   --generations 2
```

The hierarchical pipeline command exists as a placeholder:

```
python -m talos pipeline
```

Legacy Level 1 GA Python imports still exist for compatibility, but new command-line use should go through subcommands:

```
python -m talos level1 \
   --objectives latency energy area \
   --pop-size 12 \
   --generations 4 \
   --workers 4 \
   --seed 1 \
   --zigzag-lpf-limit 1 \
   --zigzag-spatial-mappings 1 \
   --results-dir ./results
```
---

## Software used
It is a work in progress. For now:
- ZigZag https://github.com/KULeuven-MICAS/zigzag
- NSGA-II implementation https://github.com/baopng/NSGA-II
- pymoo https://pymoo.org/
## Status

Work in progress.
