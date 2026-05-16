"""Load Dataset A (UCI tabular) and Dataset B (multiclass, >=1000 samples)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits

from .config import RANDOM_STATE


@dataclass
class TabularDataset:
    """Container for X (features), y (labels), feature_names, and human-readable name."""

    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    name: str
    target_name: str


def load_dataset_a_adult() -> TabularDataset:
    """
    Dataset A: Adult Census Income from OpenML (UCI-style tabular).
    Meets requirements: well over 500 samples and >=5 features after encoding raw columns.
    """
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X = adult.data.copy()
    # OpenML target may be categorical/object; avoid np.char.strip on object arrays (NumPy 2 + Py3.13).
    y_series = pd.Series(adult.target, dtype="string").str.strip()
    y = (y_series == ">50K").astype(np.int64).to_numpy()
    return TabularDataset(
        X=X,
        y=y,
        feature_names=list(X.columns),
        name="Adult Census Income (OpenML)",
        target_name="high_income",
    )


def load_dataset_b_digits() -> TabularDataset:
    """
    Dataset B: sklearn digits — 10 classes, 1797 samples (>=1000, >=3 classes).
    Each sample is 8x8 image flattened to 64 features (tabular representation).
    """
    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])])
    y = digits.target.astype(np.int64)
    return TabularDataset(
        X=X,
        y=y,
        feature_names=list(X.columns),
        name="Optical Recognition of Handwritten Digits (sklearn)",
        target_name="digit_class",
    )
