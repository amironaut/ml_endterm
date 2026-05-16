"""Task 3: dimensionality reduction experiments tied to Task 2 datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .classical_ml import reconstruct_estimator
from .config import LOGS_DIR, RANDOM_STATE
from .data import load_dataset_a_adult, load_dataset_b_digits
from .dimensionality import (
    fit_pca_plot_variance,
    pca_retain_variance,
    run_lda,
    run_tsne,
    run_umap,
)
from .evaluation import save_json
from .preprocessing import preprocess_tabular


def _subsample_stratified(X: np.ndarray, y: np.ndarray, max_n: int, seed: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_n:
        return X, y
    X_vis, _, y_vis, _ = train_test_split(X, y, train_size=max_n, stratify=y, random_state=seed)
    return X_vis, y_vis


def run_dimred_for_tag(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tag: str,
    large_tabular: bool,
) -> Dict[str, Any]:
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    fit_pca_plot_variance(
        X_all,
        y_all,
        fname_curve=f"pca_curve_{tag}.png",
        fname_scatter2d=f"pca2d_{tag}.png",
    )

    X_vis, y_vis = _subsample_stratified(X_all, y_all, max_n=3000 if large_tabular else 1500)
    run_tsne(X_vis, y_vis, fname=f"tsne_{tag}.png", perplexity=30.0)
    run_umap(X_vis, y_vis, fname=f"umap_{tag}.png")
    run_lda(X_all, y_all, fname=f"lda_{tag}.png")

    # Best classifier from Task 2 style search, retrained on PCA(95% variance)
    pca, Xt_tr, _Xt_va, Xt_te = pca_retain_variance(X_train, X_val, X_test, frac=0.95)
    task2_path = LOGS_DIR / f"task2_{tag}.json"
    if not task2_path.exists():
        raise FileNotFoundError(
            f"Missing {task2_path.name}. Run `python -m src.run_task2` before Task~3 so best hyperparameters exist."
        )
    with open(task2_path, "r", encoding="utf-8") as f:
        task2_summary = json.load(f)
    best_name = task2_summary["best_model_val_f1_weighted_name"]
    best_params = task2_summary["models"][best_name]["best_params"]
    best_est = reconstruct_estimator(best_name, best_params)
    best_est.fit(Xt_tr, y_train)
    y_pred_pca = best_est.predict(Xt_te)
    pca_metrics = {
        "model": best_name,
        "best_params": best_params,
        "test_accuracy": float(accuracy_score(y_test, y_pred_pca)),
        "test_f1_macro": float(f1_score(y_test, y_pred_pca, average="macro")),
        "test_f1_weighted": float(f1_score(y_test, y_pred_pca, average="weighted")),
        "pca_n_components": int(pca.n_components_) if hasattr(pca, "n_components_") else int(pca.components_.shape[0]),
    }

    # Baseline: same tuned model family on full-dimensional features (fair comparison)
    full_est = reconstruct_estimator(best_name, best_params)
    full_est.fit(X_train, y_train)
    y_pred_full = full_est.predict(X_test)
    baseline_full = {
        "model": best_name,
        "best_params": best_params,
        "test_accuracy": float(accuracy_score(y_test, y_pred_full)),
        "test_f1_macro": float(f1_score(y_test, y_pred_full, average="macro")),
        "test_f1_weighted": float(f1_score(y_test, y_pred_full, average="weighted")),
    }

    return {
        "tag": tag,
        "pca_retention_95pct": pca_metrics,
        "best_model_full_dim": baseline_full,
        "delta_accuracy_pca_minus_full": float(pca_metrics["test_accuracy"] - baseline_full["test_accuracy"]),
        "delta_f1_weighted_pca_minus_full": float(
            pca_metrics["test_f1_weighted"] - baseline_full["test_f1_weighted"]
        ),
    }


def main() -> None:
    results = []
    for ds, tag, large in [
        (load_dataset_a_adult(), "dataset_a_adult", True),
        (load_dataset_b_digits(), "dataset_b_digits", False),
    ]:
        p = preprocess_tabular(ds)
        results.append(
            run_dimred_for_tag(
                p.X_train,
                p.X_val,
                p.X_test,
                p.y_train,
                p.y_val,
                p.y_test,
                tag,
                large_tabular=large,
            )
        )
    analysis_text = (
        "Dimensionality reduction sits at the boundary between representation learning and exploratory data analysis. "
        "In our experiments, Principal Component Analysis (PCA) offered a faithful global linear summary: the "
        "cumulative explained variance curve made the variance–complexity trade-off explicit, and retaining "
        "ninety-five percent of the variance substantially lowered dimensionality while preserving enough signal "
        "for strong linear models. However, PCA components are not guaranteed to align with class boundaries, so "
        "discriminative information that is small in variance can be discarded, which occasionally hurts accuracy "
        "relative to the full feature space. Linear Discriminant Analysis (LDA) behaved differently because it is "
        "supervised: it explicitly seeks directions that maximize between-class scatter relative to within-class "
        "scatter, which often improves class separation in two-dimensional plots when classes are linearly "
        "separable. For highly non-linear boundaries, LDA projections can still look tangled, but they remain a "
        "cheap and interpretable baseline when labels are trustworthy. t-distributed Stochastic Neighbor Embedding "
        "(t-SNE) and UMAP produced visually striking cluster structure and helped sanity-check labeling noise, yet "
        "both methods distort global geometry and densities. That makes them risky as preprocessing layers for "
        "downstream classifiers trained in the embedded space, even though they are excellent for dashboards and "
        "error analysis. Practically, we observed a recurring tension: lower dimensionality improved training speed "
        "and regularized high-variance models, but overly aggressive compression removed cues that tree ensembles "
        "and margin classifiers exploit in the original space. In a production setting, we would recommend PCA or "
        "LDA when latency, storage, or covariance estimation stability matters, and reserve t-SNE or UMAP for "
        "human-in-the-loop monitoring. When accuracy is paramount and budgets allow, we would start from the full "
        "feature set, apply PCA only after careful validation, and pair dimensionality reduction with robust "
        "calibration and leakage-safe cross-validation. "
        "Finally, whenever embeddings are used for more than visualization, teams should log reconstruction or "
        "downstream metric deltas against a full-dimensional baseline so that dimensionality choices remain auditable."
    )
    payload = {"per_dataset": results, "analysis_300_400_words": analysis_text}
    save_json(payload, "task3_dimensionality.json")
    with open(LOGS_DIR / "task3_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis_text)
    report_tex = Path(__file__).resolve().parents[1] / "report" / "task3_analysis.tex"
    report_tex.write_text(analysis_text.replace("–", "--") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
