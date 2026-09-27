# Geneva building-permit classification

Does the order in which a compact classifier sees LLM labels and original administrative
labels change what it learns? Four training schedules, eight operation codes, predicted from
the French `DESCRIPTION` field alone.

Research plan: `documentation/geneva-building-permit-classification.md` (untracked).

## Status

Structure only. No data downloaded, no runs, no results.

## Layout

```
config.yaml       experiment settings
prompt.md         annotation prompt (not written)
src/data.py       load, split, freeze, leakage checks
src/annotate.py   teacher annotation
src/baselines.py  majority class, TF-IDF + LinearSVC
src/train.py      backbone and the four label-order schedules
src/evaluate.py   metrics, uncertainty, subgroups
tests/            integrity gates
notebooks/        exploration and results only
data/splits/      manifests (tracked); raw and prepared are not
results/<run>/    config, metrics, predictions, figures
```

## Setup

```bash
uv sync
cp .env.example .env
uv run --env-file .env pytest -q
```

`--env-file .env` is required: uv does not read `.env` automatically.

## Hardware

RTX 2070, 7.59 GiB, `sm_75` · torch 2.14.0+cu130 · fp16, measured 27.2 TFLOP/s against
6.4 (fp32) and 3.9 (bf16, emulated on Turing).

## Attribution

Permit data © [SITG](https://sitg.ge.ch), used under its published terms.
