"""Learning curves: C and n-grams retuned on nested training subsets, the weighting fixed."""

import numpy as np
import pandas as pd

from geneva_permits.classical import select, tune
from geneva_permits.config import load_config
from geneva_permits.preprocess import load_texts, train_validation
from geneva_permits.tracking import LEARNING_CURVES, set_experiment


def nested_subsets(n, fractions, seed):
    """Row positions per fraction from one shuffle, so that 10% ⊂ 30% ⊂ 100%."""
    order = np.random.default_rng(seed).permutation(n)
    return {fraction: np.sort(order[: round(fraction * n)]) for fraction in fractions}


def draws(fractions, seeds):
    """(fraction, seed) pairs; the full set has a single possible draw."""
    return [(f, s) for f in fractions for s in (seeds if f < 1 else seeds[:1])]


def run_curves(variant, weighting, texts=None, settings=None, fractions=None, seeds=None):
    """Retune the selected variant and weighting on each draw; the best point per draw."""
    config = load_config()
    settings = settings or config["classical"]
    fractions = fractions or config["selection"]["learning_curves"]["fractions"]
    seeds = seeds or config["training"]["seeds"]
    texts = load_texts() if texts is None else texts
    train, validation = train_validation(texts)
    set_experiment(LEARNING_CURVES)
    best = []
    for fraction, seed in draws(fractions, seeds):
        subset = train.iloc[nested_subsets(len(train), fractions, seed)[fraction]]
        extra = {"fraction": fraction, "seed": seed}
        best.append(select(tune(variant, subset, validation, settings, [weighting], extra)))
    return pd.DataFrame(best).reset_index(drop=True)
