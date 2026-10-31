"""Per-row class weights, computed from the rows actually trained on (config.yaml `weighting`)."""

import numpy as np
import pandas as pd

OPTIONS = ["none", "medium", "full"]


def class_weights(labels, option):
    """Weight per class: 1, the square root of balanced rescaled to mean 1 per row, or balanced."""
    counts = pd.Series(labels).value_counts()
    balanced = len(labels) / (len(counts) * counts)  # n / (K n_c), mean 1 per row
    if option == "none":
        return pd.Series(1.0, index=counts.index)
    if option == "medium":
        root = np.sqrt(balanced)
        return root / root[labels].mean()
    if option == "full":
        return balanced
    raise ValueError(f"unknown weighting option {option!r}; expected one of {OPTIONS}")


def row_weights(labels, option):
    """One weight per row: the class weight of its label."""
    return class_weights(labels, option)[labels].to_numpy()
