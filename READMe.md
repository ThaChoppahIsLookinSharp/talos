# TALOS
Welcome to my caffeine fueled thesis project. The idea is that this thinks of a proper
architecture for your DNN accelerator while you do other stuff.

---
## Overview

Long story short, you specify some parameters, this will try its best to 
generate some nice architecture.

It is a "2 level" generator.

Level 1, usual NGSA-2. Fitness is outputs of ZigZag.
Results are a set of abstract architectures.

Level 2, using these abstract architectures

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

python -m talos

Define objectives in code:
```
objectives = [
    adapter.latency,
    adapter.energy,
    adapter.area,
]
```
---

## Status

Work in progress.
