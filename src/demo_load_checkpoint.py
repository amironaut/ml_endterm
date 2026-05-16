"""Demonstrate loading saved PyTorch checkpoints (grading checklist)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .cnn_task5 import CustomCIFAR10CNN
from .config import MODELS_DIR, RANDOM_STATE
from .mlp_train import MLPClassifier


def _safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mlp_example(path: str | None = None, meta_path: str | None = None) -> MLPClassifier:
    """Load MLP checkpoint using saved metadata (in_dim, hidden layers, classes)."""
    meta_p = Path(meta_path) if meta_path else MODELS_DIR / "mlp_relu_meta.json"
    ckpt_p = Path(path) if path else MODELS_DIR / "mlp_relu.pt"
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = MLPClassifier(
        meta["in_dim"],
        tuple(meta["hidden"]),
        num_classes=meta["num_classes"],
        activation=meta.get("activation", "relu"),
    )
    model.load_state_dict(_safe_torch_load(ckpt_p))
    model.eval()
    return model


def load_cnn_example(path: str | None = None) -> CustomCIFAR10CNN:
    p = Path(path) if path else MODELS_DIR / "custom_cnn_cifar10.pt"
    model = CustomCIFAR10CNN()
    model.load_state_dict(_safe_torch_load(p))
    model.eval()
    return model


if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    print("MLP load demo:")
    m = load_mlp_example()
    in_dim = m.net[0].in_features  # type: ignore[attr-defined]
    x = torch.randn(2, in_dim)
    with torch.no_grad():
        print("  output shape:", m(x).shape)

    print("CNN load demo:")
    c = load_cnn_example()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        print("  output shape:", c(x).shape)
    print("Checkpoint loading OK.")
