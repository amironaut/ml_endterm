"""Execute Task 2 pipeline for one tabular dataset and persist metrics and figures."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score

from .classical_ml import TrainedModelResult, train_all_models
from .config import RANDOM_STATE
from .data import TabularDataset, load_dataset_a_adult, load_dataset_b_digits
from .evaluation import (
    binary_roc_plot,
    classification_metrics_dict,
    full_classification_report_text,
    plot_confusion_matrix,
    plot_multiclass_roc,
    run_stratified_kfold_multi,
    save_json,
)
from .preprocessing import preprocess_tabular


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def decision_scores(estimator: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """Return shape (n_samples, n_classes) probability-like scores for ROC."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))
    raw = estimator.decision_function(X)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        p1 = _sigmoid(raw)
        return np.column_stack([1.0 - p1, p1])
    return raw


def evaluate_trained(
    results: List[TrainedModelResult],
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_tag: str,
    multiclass: bool,
) -> Dict[str, Any]:
    """Metrics, confusion matrix, ROC, and 5-fold CV on full training data for each model."""
    summary: Dict[str, Any] = {}
    for r in results:
        y_pred = r.estimator.predict(X_test)
        metrics = classification_metrics_dict(y_test, y_pred)
        summary[r.name] = {
            "best_params": r.best_params,
            "search_type": r.search_type,
            "test_metrics": metrics,
            "classification_report": full_classification_report_text(y_test, y_pred),
        }
        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"{dataset_tag} — {r.name}",
            fname=f"cm_{dataset_tag}_{r.name}.png",
        )
        scores = decision_scores(r.estimator, X_test)
        if multiclass and scores.shape[1] > 2:
            plot_multiclass_roc(
                y_test,
                scores,
                title=f"{dataset_tag} ROC (OvR) — {r.name}",
                fname=f"roc_{dataset_tag}_{r.name}.png",
            )
        else:
            binary_roc_plot(
                y_test,
                scores[:, 1],
                title=f"{dataset_tag} ROC — {r.name}",
                fname=f"roc_{dataset_tag}_{r.name}.png",
            )
        # 5-fold CV on train+val merged for reporting stability (use indices from caller)
    return summary


def run_for_dataset(ds: TabularDataset, dataset_tag: str, large_tabular: bool) -> Dict[str, Any]:
    proc = preprocess_tabular(ds)
    X_train, y_train = proc.X_train, proc.y_train
    X_val, y_val = proc.X_val, proc.y_val
    X_test, y_test = proc.X_test, proc.y_test
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    trained = train_all_models(X_train, y_train, large_tabular=large_tabular)

    # pick best by validation F1 weighted
    best_name, best_f1 = None, -1.0
    for r in trained:
        yv = r.estimator.predict(X_val)
        f1 = f1_score(y_val, yv, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_name = r.name

    multiclass = len(np.unique(y_train)) > 2
    model_summary = evaluate_trained(trained, X_test, y_test, dataset_tag, multiclass=multiclass)

    cv_block: Dict[str, Any] = {}
    for r in trained:
        cv_block[r.name] = run_stratified_kfold_multi(r.estimator, X_trainval, y_trainval, cv=5)

    for r in trained:
        model_summary[r.name]["cv_5fold"] = cv_block[r.name]
        model_summary[r.name]["cv_accuracy_mean_std"] = cv_block[r.name]["accuracy"]

    classes, counts = np.unique(y_trainval, return_counts=True)
    imbalance_note = (
        "Adult is imbalanced (~3:1 for majority class). We use stratified 70/15/15 splits, "
        "GridSearchCV scoring=f1_weighted, and report both macro and weighted precision/recall/F1 "
        "so minority-class performance remains visible."
    )

    out = {
        "dataset": ds.name,
        "target": ds.target_name,
        "best_model_val_f1_weighted_name": best_name,
        "best_model_val_f1_weighted": best_f1,
        "models": model_summary,
        "class_imbalance": {
            "strategy": imbalance_note,
            "trainval_class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        },
        "preprocessing": {
            "numeric_features": proc.numeric_features,
            "categorical_features": proc.categorical_features,
            "split_ratio": "70/15/15 stratified",
            "notes": "Median imputation + StandardScaler for numeric columns; "
            "most frequent + one-hot for categoricals (unknown categories ignored at transform).",
        },
    }
    save_json(out, f"task2_{dataset_tag}.json")
    return out


def main() -> None:
    run_for_dataset(load_dataset_a_adult(), "dataset_a_adult", large_tabular=True)
    run_for_dataset(load_dataset_b_digits(), "dataset_b_digits", large_tabular=False)


if __name__ == "__main__":
    main()
