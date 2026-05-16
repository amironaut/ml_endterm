"""Dimensionality reduction: PCA, t-SNE, UMAP, LDA and downstream evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from .config import FIGURES_DIR, RANDOM_STATE


@dataclass
class PCAResult:
    model: PCA
    X_pca: np.ndarray
    cumulative_variance: np.ndarray


def fit_pca_plot_variance(
    X: np.ndarray,
    y: Optional[np.ndarray],
    fname_curve: str,
    fname_scatter2d: str,
    max_components: Optional[int] = None,
) -> PCAResult:
    """Fit PCA on full X, plot explained variance curve and 2D scatter (first two PCs)."""
    n = X.shape[1] if max_components is None else min(max_components, X.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_pca_full = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), cumvar, marker="o", ms=3)
    ax.axhline(0.95, color="r", linestyle="--", label="95% variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA cumulative explained variance (elbow / retention curve)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname_curve, dpi=150)
    plt.close(fig)

    if y is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
        ax.set_title("PCA 2D projection (first two components)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, label="class")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname_scatter2d, dpi=150)
        plt.close(fig)

    return PCAResult(model=pca, X_pca=X_pca_full, cumulative_variance=cumvar)


def pca_retain_variance(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, frac: float = 0.95) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on training data, choose k for `frac` explained variance, transform splits."""
    pca = PCA(n_components=frac, svd_solver="full", random_state=RANDOM_STATE)
    Xt = pca.fit_transform(X_train)
    return pca, Xt, pca.transform(X_val), pca.transform(X_test)


def plot_embedding_2d(Z: np.ndarray, y: np.ndarray, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=10, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    fig.colorbar(sc, ax=ax, label="class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def run_tsne(X: np.ndarray, y: np.ndarray, fname: str, perplexity: float = 30.0) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=RANDOM_STATE)
    Z = tsne.fit_transform(X)
    plot_embedding_2d(Z, y, "t-SNE 2D embedding", fname)
    return Z


def run_umap(X: np.ndarray, y: np.ndarray, fname: str) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    Z = reducer.fit_transform(X)
    plot_embedding_2d(Z, y, "UMAP 2D embedding", fname)
    return Z


def run_lda(X: np.ndarray, y: np.ndarray, fname: str) -> np.ndarray:
    lda = LinearDiscriminantAnalysis()
    Z = lda.fit_transform(X, y)
    if Z.shape[1] == 1:
        Z2 = np.column_stack([Z[:, 0], np.zeros(len(Z))])
    else:
        Z2 = Z[:, :2]
    plot_embedding_2d(Z2, y, "LDA projection (first two discriminants)", fname)
    return Z
