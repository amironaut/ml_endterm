"""Train and evaluate MLP (Task 4) on Dataset A with optimizer comparison."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .config import FIGURES_DIR, LOGS_DIR, MODELS_DIR, RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPClassifier(nn.Module):
    """MLP with configurable activation, BatchNorm, and Dropout after each hidden block."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], num_classes: int, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation.lower()]
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                act(),
                nn.Dropout(0.25),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_run(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
) -> TrainHistory:
    """Standard supervised training loop with cross-entropy."""
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    hist = TrainHistory([], [], [], [])
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = running / max(n_seen, 1)
        train_acc = correct / max(total, 1)
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.long, device=device)
            logits = model(xv)
            vloss = loss_fn(logits, yv).item()
            vacc = _accuracy(logits, yv)
        hist.train_loss.append(train_loss)
        hist.val_loss.append(vloss)
        hist.train_acc.append(train_acc)
        hist.val_acc.append(vacc)
    return hist


def epochs_to_plateau(val_losses: List[float], window: int = 3, rel_tol: float = 0.002) -> int:
    """Heuristic: first epoch after which moving average changes < rel_tol for `window` steps."""
    if len(val_losses) < window + 1:
        return len(val_losses)
    for i in range(window, len(val_losses)):
        w = val_losses[i - window : i]
        if max(w) - min(w) < rel_tol * (abs(np.mean(w)) + 1e-8):
            return i
    return len(val_losses)


def plot_histories(hists: Dict[str, TrainHistory], fname: str, metric: str = "loss") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, h in hists.items():
        if metric == "loss":
            ax.plot(h.val_loss, label=f"{name} (val)")
        else:
            ax.plot(h.val_acc, label=f"{name} (val)")
    ax.set_title(f"Validation {metric} — optimizer comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_history_json(hists: Dict[str, TrainHistory], fname: str) -> Path:
    payload = {k: vars(v) for k, v in hists.items()}
    path = LOGS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def demonstrate_core_concepts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 25,
) -> Dict[str, object]:
    """
    Assignment core-concept demos: Sigmoid activations, softmax (via CE head), MSE loss, L1/L2 penalties.
    Logged to results/logs/mlp_core_concepts.json.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import label_binarize

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    # Sigmoid hidden activations + cross-entropy (binary/multiclass logits)
    sig_model = MLPClassifier(in_dim, (128, 64), num_classes=num_classes, activation="sigmoid").to(device)
    sig_opt = torch.optim.Adam(sig_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sig_hist = train_one_run(sig_model, X_train, y_train, X_val, y_val, sig_opt, None, epochs, device)
    with torch.no_grad():
        pred = sig_model(torch.tensor(X_test, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    sig_acc = float(accuracy_score(y_test, pred))

    # MSE on one-hot targets (regression view of classification — demonstrates MSE loss)
    classes = np.arange(num_classes)
    y_train_oh = label_binarize(y_train, classes=classes)
    y_test_oh = label_binarize(y_test, classes=classes)
    if num_classes == 2 and y_train_oh.shape[1] == 1:
        y_train_oh = np.column_stack([1.0 - y_train_oh, y_train_oh])
        y_test_oh = np.column_stack([1.0 - y_test_oh, y_test_oh])
    mse_model = MLPClassifier(in_dim, (128, 64), num_classes=num_classes, activation="relu").to(device)
    mse_opt = torch.optim.Adam(mse_model.parameters(), lr=1e-3)
    mse_loss_fn = nn.MSELoss()
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_oh, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    for _ in range(epochs):
        mse_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mse_opt.zero_grad(set_to_none=True)
            out = torch.softmax(mse_model(xb), dim=1)
            loss = mse_loss_fn(out, yb)
            loss.backward()
            mse_opt.step()
    with torch.no_grad():
        logits = mse_model(torch.tensor(X_test, dtype=torch.float32, device=device))
        mse_pred = logits.argmax(1).cpu().numpy()
        mse_val = float(mean_squared_error(y_test_oh, torch.softmax(logits, dim=1).cpu().numpy()))
    mse_acc = float(accuracy_score(y_test, mse_pred))

    # L1 vs L2 on logistic regression (classical regularization, same Adult features)
    l1 = LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        C=0.5,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    l2 = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    l1.fit(X_train, y_train)
    l2.fit(X_train, y_train)
    l1_acc = float(accuracy_score(y_test, l1.predict(X_test)))
    l2_acc = float(accuracy_score(y_test, l2.predict(X_test)))
    l1_nnz = int(np.sum(np.abs(l1.coef_) > 1e-6))

    payload = {
        "sigmoid_mlp_test_accuracy": sig_acc,
        "sigmoid_mlp_final_val_loss": float(sig_hist.val_loss[-1]),
        "mse_softmax_head_test_mse": mse_val,
        "mse_softmax_head_test_accuracy": mse_acc,
        "logistic_l1_test_accuracy": l1_acc,
        "logistic_l1_nonzero_coefs": l1_nnz,
        "logistic_l2_test_accuracy": l2_acc,
        "notes": "Softmax is applied at the output for MSE demo; cross-entropy MLP uses linear logits + CrossEntropyLoss (implicit softmax).",
    }
    path = LOGS_DIR / "mlp_core_concepts.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_mlp_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden: Tuple[int, int, int] = (256, 128, 64),
    epochs_main: int = 60,
    epochs_opt: int = 40,
) -> Dict[str, object]:
    """Main Task 4 driver: activation comparison + optimizer study."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    results: Dict[str, object] = {}

    for act in ("relu", "tanh"):
        model = MLPClassifier(in_dim, hidden, num_classes=num_classes, activation=act).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_main)
        hist = train_one_run(model, X_train, y_train, X_val, y_val, opt, sched, epochs_main, device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
            pred = logits.argmax(1).cpu().numpy()
        results[f"mlp_{act}_test_acc"] = float(accuracy_score(y_test, pred))
        results[f"mlp_{act}_test_f1w"] = float(f1_score(y_test, pred, average="weighted"))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist.train_loss, label="train")
        ax[0].plot(hist.val_loss, label="val")
        ax[0].set_title(f"Loss ({act})")
        ax[0].legend()
        ax[1].plot(hist.train_acc, label="train")
        ax[1].plot(hist.val_acc, label="val")
        ax[1].set_title(f"Accuracy ({act})")
        ax[1].legend()
        fig.suptitle("MLP training curves (Task 4)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"mlp_curves_{act}.png", dpi=150)
        plt.close(fig)
        ckpt_path = MODELS_DIR / f"mlp_{act}.pt"
        torch.save(model.state_dict(), ckpt_path)
        meta = {"in_dim": in_dim, "hidden": list(hidden), "num_classes": num_classes, "activation": act}
        with open(MODELS_DIR / f"mlp_{act}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    concepts = demonstrate_core_concepts(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25)
    results["core_concepts"] = concepts

    # Optimizer comparison on ReLU model
    hists: Dict[str, TrainHistory] = {}
    plateau_epochs: Dict[str, int] = {}
    for name, builder in [
        ("SGD", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-4)),
        ("Adam", lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=1e-4)),
        ("RMSprop", lambda p: torch.optim.RMSprop(p, lr=1e-3, weight_decay=1e-4)),
    ]:
        model = MLPClassifier(in_dim, hidden, num_classes=num_classes, activation="relu").to(device)
        opt = builder(model.parameters())
        hist = train_one_run(model, X_train, y_train, X_val, y_val, opt, None, epochs_opt, device)
        hists[name] = hist
        plateau_epochs[name] = epochs_to_plateau(hist.val_loss)

    plot_histories(hists, "mlp_optimizer_val_loss.png", metric="loss")
    plot_histories(hists, "mlp_optimizer_val_acc.png", metric="acc")
    save_history_json(hists, "mlp_optimizer_histories.json")
    results["optimizer_plateau_epochs"] = plateau_epochs
    with open(LOGS_DIR / "mlp_optimizer_plateau.json", "w", encoding="utf-8") as f:
        json.dump(plateau_epochs, f, indent=2)
    return results
