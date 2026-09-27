# LLM labels and training order for Geneva building-permit classification

Does the **order** in which a compact classifier sees LLM-generated labels and original
administrative labels change what it learns? The study compares four training schedules on
French building-permit descriptions from the Canton of Geneva, predicting the recorded
operation category from free text alone.

The application is retrospective completion of missing administrative categories for search
and human review. It does not predict permit approval or establish what was built.

## Status

Scaffolding and environment only. No data has been downloaded, no model trained, no result
produced. The counts and the 0.731 TF-IDF validation macro F1 quoted in the research plan come
from a prior feasibility audit and **have not been reproduced here**.

Open blockers:

| Blocker | Consequence |
|---|---|
| SITG export not yet downloaded | every partition count is unverified |
| Official definitions of the eight categories not located | `prompt.md` cannot be written |
| Teacher model alias and price unverified (needs an API key) | annotation cost unknown |

## Research questions

1. Does adding LLM-labelled data beat original-only training at the same update budget?
2. Does **P → O** outperform mixed training when examples and per-source exposure are matched?
3. How does **O → P** change performance on rare categories and unfamiliar descriptions?
4. Are any gains large enough to justify the annotation cost against a strong conventional
   classifier?

## Data

Geneva's open [SIT_AUTOR_DOSSIER](https://sitg.ge.ch/donnees/sit-autor-dossier) export.
Input is the `DESCRIPTION` field only. Target: eight operation codes — AFF, AGR, AGV, EQU,
NOU, PI, REN, VIL. The AM planning code is excluded from the taxonomy.

Partitions follow the **filing calendar** rather than a 70/10/20 draw, because label
availability changes strongly over time:

| Partition | Filing period | Role |
|---|---|---|
| Original-labelled training, **O** | through 2010 | supervised training |
| Validation | 2011–2012 | model and protocol development |
| Historical test | 2013 | secondary temporal evaluation |
| Recent test | 2014–2026 | primary test source |
| Unlabelled pool → **P** | 2014–2026 | teacher annotation |

`TYPE_OPERATION`, `OPERATION`, dossier identifiers and procedural names never reach a model.
Normalisation is used for grouping only; model input keeps the original text.

## Training conditions

| Condition | Sequence |
|---|---|
| **O-only** | original labels throughout, repeated to match the update budget |
| **Mixed** | O and P interleaved throughout |
| **P → O** | P stage complete, then the O stage |
| **O → P** | O stage complete, then the P stage |

Three seeds (13, 42, 73) per condition — twelve neural runs. Across the three enriched
conditions, examples, per-source exposure, loss, effective batch size and total updates are
held constant; only the order changes.

## Models

Revisions are pinned in `config.yaml` and were verified against the Hugging Face API on
2026-09-27.

| Role | Checkpoint | Revision |
|---|---|---|
| Primary | [`almanach/camembertav2-base`](https://huggingface.co/almanach/camembertav2-base) (110.6M) | `54cc91d6` |
| Challenger | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) | `c5955035` |
| Baselines | majority class; word+char TF-IDF + LinearSVC | — |

## Evaluation

Primary metric is **macro F1 on the text-disjoint subset of the recent test**, with
**P → O versus Mixed** as the primary contrast. Text-disjoint means sharing no folded
description with O, accepted P **or** validation — a P description leaking into the test
subset would manufacture an apparent distillation gain.

Full recent and historical tests, per-class F1 with support, accuracy and confusion matrices
are reported separately, with every seed shown and paired bootstrap intervals clustered by
dossier.

The design is retrospective and label source is confounded with filing period and population,
so the study does not isolate label quality. A null or negative result is informative.

## Procedure

1. **Freeze data** — preserve source bytes, hashes, taxonomy, cleaning rules and split
   manifests; implement the leakage checks in `tests/test_integrity.py`.
2. **Establish baselines** — TF-IDF variants, the subset-size check, the backbone pilot.
3. **Audit teachers** — develop and freeze `prompt.md`, acceptance rules and teacher choice.
4. **Annotate P** — resumable, cached requests; publish accept/reject counts by reason.
5. **Freeze settings** — commit `config.yaml` and `prompt.md` before any final run.
6. **Train** — all twelve runs plus baselines, saving checkpoints, logs and predictions.
7. **Evaluate** — open the test partitions only after development decisions are fixed.

## Reproduction

```bash
uv sync                                        # Python 3.14.6 pinned in .python-version
cp .env.example .env                           # then fill in the teacher API key
uv run --env-file .env pytest -q               # integrity gates
uv run --env-file .env ruff check .
```

`--env-file .env` is required: uv does not load `.env` automatically.

## Hardware

Measured on the development machine, not assumed:

| | |
|---|---|
| GPU | RTX 2070, 7.59 GiB usable, compute capability `sm_75` |
| Driver / CUDA | 580.159.03 / 13.0; torch 2.14.0+cu130 |
| Precision | **fp16** — measured 27.2 TFLOP/s vs 6.4 (fp32) and 3.9 (bf16) |

`torch.cuda.is_bf16_supported()` returns `True` on this card, but bf16 is emulated and runs
slower than fp32, so fp16 is the only reasonable choice. 8 GiB caps the batch size, so
gradient accumulation carries the effective batch.

If measured runtime or memory proves unsuitable, the same code and configuration move to a
single rented GPU.

## Layout

```
config.yaml       all experiment settings; every value states its reason
prompt.md         the frozen annotation prompt (not yet written)
src/data.py       load, clean, split, freeze, leakage-check
src/annotate.py   teacher annotation with caching and acceptance rules
src/baselines.py  majority class and TF-IDF + LinearSVC
src/train.py      backbone setup and the four label-order schedules
src/evaluate.py   metrics, uncertainty, subgroup analysis
tests/            integrity gates
notebooks/        exploration and results only; logic stays in src/
data/splits/      split manifests (tracked); raw and prepared data are not
results/<run>/    config, metrics, predictions, figures per run
```

## Attribution

Permit data © [Système d'information du territoire à Genève (SITG)](https://sitg.ge.ch),
used under its published terms. API keys live in environment variables only.
