"""Neural training: model setup and the four label-order schedules.

Responsibilities:
  - build the eight-class classifier from the pinned backbone revision
  - pair initial checkpoint and head initialisation across conditions within a seed
  - the four schedules: o_only, mixed, p_then_o, o_then_p
  - hold examples, per-source exposure, loss, effective batch size and total updates
    constant across the three enriched conditions; only the order changes
  - one AdamW configuration and one learning-rate schedule, no optimizer reset at a
    stage boundary
  - log exposure counts, truncation rate, runtime and memory to MLflow

The final checkpoint after the complete schedule is the primary artefact, so that
checkpoint selection cannot silently discard the second stage.

NOTE: transformers 5.x changed the Trainer API from 4.x. Read the v5 documentation
before implementing; do not port v4 code from memory.
"""
