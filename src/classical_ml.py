"""Train Task 1 classifiers with hyperparameter search (GridSearchCV / RandomizedSearchCV)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE


@dataclass
class TrainedModelResult:
    name: str
    estimator: Any
    best_params: Dict[str, Any]
    search_type: str


def _knn_pipe() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])


def _dt() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


def _logreg() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )


def _nb() -> GaussianNB:
    return GaussianNB()


def _linear_svc() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=RANDOM_STATE, dual="auto")),
        ]
    )


def _rbf_svc() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))])


def reconstruct_estimator(name: str, best_params: Dict[str, Any]) -> Any:
    """Rebuild an unfitted estimator using Task~2 hyperparameters (same architecture, refit on new X)."""
    builders = {
        "knn": _knn_pipe,
        "decision_tree": _dt,
        "random_forest": _rf,
        "logistic_regression": _logreg,
        "naive_bayes": _nb,
        "svm_linear": _linear_svc,
        "svm_rbf": _rbf_svc,
    }
    if name not in builders:
        raise ValueError(f"Unknown model name: {name}")
    est = builders[name]()
    if best_params:
        est.set_params(**best_params)
    return est


def build_search_spaces(large_tabular: bool) -> List[Tuple[str, Any, Dict[str, Any], str]]:
    """
    Return list of (name, estimator, param_grid, search_kind).
    search_kind is 'grid' or 'random' (RandomizedSearchCV).
    """
    models: List[Tuple[str, Any, Dict[str, Any], str]] = []

    knn_params = {
        "clf__n_neighbors": [3, 5, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    models.append(("knn", _knn_pipe(), knn_params, "grid"))

    dt_params = {
        "max_depth": [None, 8, 12, 18],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 4],
    }
    models.append(("decision_tree", _dt(), dt_params, "grid"))

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 14, 22],
        "min_samples_leaf": [1, 2],
    }
    models.append(("random_forest", _rf(), rf_params, "grid"))

    lr_params = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    models.append(("logistic_regression", _logreg(), lr_params, "grid"))

    models.append(("naive_bayes", _nb(), {}, "grid"))  # no hyperparameters

    if large_tabular:
        lsvm_params = {"clf__C": [0.25, 0.5, 1.0, 2.0, 4.0]}
        models.append(("svm_linear", _linear_svc(), lsvm_params, "grid"))
    else:
        svc_params = {
            "clf__C": [0.5, 1.0, 2.0, 4.0],
            "clf__gamma": ["scale", 0.01, 0.001],
            "clf__kernel": ["rbf"],
        }
        models.append(("svm_rbf", _rbf_svc(), svc_params, "random"))

    return models


def fit_with_search(
    name: str,
    base_est: Any,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_kind: str,
    cv: int = 5,
    n_iter: int = 24,
) -> TrainedModelResult:
    """Run GridSearchCV or RandomizedSearchCV and return best estimator."""
    if not param_grid:
        base_est.fit(X_train, y_train)
        return TrainedModelResult(name=name, estimator=base_est, best_params={}, search_type="none")

    if search_kind == "grid":
        n_jobs = int(os.environ.get("SKLEARN_N_JOBS", "1"))
        search = GridSearchCV(
            base_est,
            param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
    else:
        n_jobs = int(os.environ.get("SKLEARN_N_JOBS", "1"))
        search = RandomizedSearchCV(
            base_est,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )
    search.fit(X_train, y_train)
    return TrainedModelResult(
        name=name,
        estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        search_type=search_kind,
    )


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    large_tabular: bool,
) -> List[TrainedModelResult]:
    """Train every configured classifier with hyperparameter tuning."""
    results: List[TrainedModelResult] = []
    for name, est, grid, sk in build_search_spaces(large_tabular=large_tabular):
        results.append(fit_with_search(name, est, grid, X_train, y_train, sk))
    return results
