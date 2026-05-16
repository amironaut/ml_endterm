"""Metrics, confusion matrices, ROC (OvR), and cross-validation reporting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

from .config import FIGURES_DIR, LOGS_DIR, RANDOM_STATE


def classification_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Accuracy, precision/recall/F1 macro and weighted."""
    labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    fname: str,
    labels: Optional[List[str]] = None,
) -> Path:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    fname: str,
) -> Path:
    """One-vs-rest ROC with macro-average AUC (multiclass)."""
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.5, label=f"class {c} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    mean_auc = float(np.mean(aucs))
    ax.set_title(f"{title}\nMacro-average AUC = {mean_auc:.4f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def binary_roc_plot(y_true: np.ndarray, y_score_pos: np.ndarray, title: str, fname: str) -> Path:
    """ROC for binary problems using positive-class scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score_pos)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_stratified_kfold_scores(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """Return mean ± std for requested scoring metric."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    import os

    n_jobs = int(os.environ.get("SKLEARN_N_JOBS", "1"))
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores.tolist()}


def run_stratified_kfold_multi(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, Dict[str, float]]:
    """5-fold CV for accuracy, macro-F1, and weighted-F1 (assignment reporting)."""
    metrics = ("accuracy", "f1_macro", "f1_weighted")
    return {name: run_stratified_kfold_scores(model, X, y, cv=cv, scoring=name) for name in metrics}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Any, fname: str) -> Path:
    path = LOGS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    return path


def full_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, digits=4)
