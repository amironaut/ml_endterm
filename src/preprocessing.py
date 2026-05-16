"""Preprocessing: missing values, encoding, scaling, stratified 70/15/15 splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .data import TabularDataset


@dataclass
class ProcessedData:
    """Train/val/test arrays and metadata for reproducibility."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric vs categorical using pandas dtypes."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Numeric: median imputation (robust to skew) + standardization (zero mean, unit variance).
    Categorical: most_frequent imputation + one-hot encoding (ignore unknown at transform).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers)


def preprocess_tabular(
    ds: TabularDataset,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> ProcessedData:
    """
    Stratified 70/15/15 split: first hold out test, then split remaining into train/val
    so that val_fraction of original = 0.15 => val_size of (1-test) ≈ 0.15/(1-0.15).
    """
    numeric_features, categorical_features = infer_feature_types(ds.X)
    pre = build_preprocessor(numeric_features, categorical_features)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=test_size,
        stratify=ds.y,
        random_state=RANDOM_STATE,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    return ProcessedData(
        X_train=np.asarray(X_train_t, dtype=np.float32),
        X_val=np.asarray(X_val_t, dtype=np.float32),
        X_test=np.asarray(X_test_t, dtype=np.float32),
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        preprocessor=pre,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
