"""Conventional baselines: majority class and word+character TF-IDF with LinearSVC.

Responsibilities:
  - the three preprocessing variants (basic / stop words / lemmatisation), applied to the
    word-feature branch only so the character branch stays constant across variants
  - variant selection on validation macro F1, reporting all three including regressions
  - the O versus O+P comparison using the selected configuration
  - the nested small/medium/full subset check, fitting vocabulary per subset

TF-IDF is fit on training data only.
"""
