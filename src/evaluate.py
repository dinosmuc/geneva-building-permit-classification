"""Evaluation, uncertainty and the supporting subgroup analysis.

Responsibilities:
  - macro F1, accuracy, per-class F1 with support, and confusion matrices
  - the primary metric on the text-disjoint subset of the recent test, plus the full
    recent and historical tests reported separately
  - per-seed results with mean and standard deviation, and paired differences
  - paired bootstrap intervals clustered by dossier, keeping test-sample uncertainty
    distinct from three-seed training variation
  - the frozen permit-description form groups (abbreviated / full sentence /
    multi-operation), applied without consulting predictions or errors

Test partitions stay sealed until development decisions are frozen.
"""
