"""Task 5: custom CNN, augmentation, transfer learning, and visual diagnostics (PyTorch)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

from .config import FIGURES_DIR, LOGS_DIR, MODELS_DIR, RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class CustomCIFAR10CNN(nn.Module):
    """At least three Conv+BN+ReLU+MaxPool blocks and a two-layer FC head with Dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


def cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train with augmentation; val/test use deterministic preprocessing (no augmentation)."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    val_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=eval_tf)
    n = len(full_train)
    n_train = int(0.85 * n)
    n_val = n - n_train
    train_subset, val_subset_idx = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    val_indices = val_subset_idx.indices
    val_subset = torch.utils.data.Subset(val_base, val_indices)

    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def cifar10_loaders_imagenet_norm(
    batch_size: int = 64,
    max_train: int | None = 8000,
    max_val: int | None = None,
    max_test: int | None = None,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders with ImageNet normalization for ResNet-18 transfer learning.

    ``max_train`` / ``max_val`` cap samples for fast CPU runs; use ``img_size=224`` for full quality.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    crop = max(32, int(img_size))
    resize = int(crop * 1.15)
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocess.mean, std=preprocess.std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocess.mean, std=preprocess.std),
        ]
    )
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    val_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=eval_tf)
    n = len(full_train)
    n_train = int(0.85 * n)
    n_val = n - n_train
    train_subset, val_subset_idx = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    if max_train is not None and len(train_subset) > max_train:
        train_subset, _ = random_split(
            train_subset,
            [max_train, len(train_subset) - max_train],
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
    val_indices = val_subset_idx.indices
    val_subset = torch.utils.data.Subset(val_base, val_indices)
    if max_val is not None and len(val_subset) > max_val:
        val_subset, _ = random_split(
            val_subset,
            [max_val, len(val_subset) - max_val],
            generator=torch.Generator().manual_seed(RANDOM_STATE + 1),
        )
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
    if max_test is not None and len(test_ds) > max_test:
        test_ds = torch.utils.data.Subset(test_ds, list(range(max_test)))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    tag: str,
    lr: float = 0.1,
) -> Dict[str, List[float]]:
    """Train with cross-entropy, cosine LR schedule, SGD+momentum."""
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        tl, ta, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            ta += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()
        history["train_loss"].append(tl / max(n, 1))
        history["train_acc"].append(ta / max(n, 1))
        model.eval()
        vl, va, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                vl += loss.item() * x.size(0)
                va += (logits.argmax(1) == y).sum().item()
                m += x.size(0)
        history["val_loss"].append(vl / max(m, 1))
        history["val_acc"].append(va / max(m, 1))
    torch.save(model.state_dict(), MODELS_DIR / f"{tag}.pt")
    return history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, confs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        ys.append(y.numpy())
        ps.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(confs)


def plot_cnn_history(hist: Dict[str, List[float]], fname: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].legend()
    ax[0].set_title("Loss")
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].legend()
    ax[1].set_title("Accuracy")
    fig.suptitle("Custom CNN on CIFAR-10")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def visualize_first_conv_filters(model: CustomCIFAR10CNN, fname: str, k: int = 8) -> None:
    w = model.block1.conv.weight.detach().cpu()[:k]
    grid = make_grid(w, nrow=4, normalize=True, scale_each=True)
    npimg = grid.numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(np.clip(npimg, 0, 1))
    ax.axis("off")
    ax.set_title("First conv layer filters (subset)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def activation_maps(model: CustomCIFAR10CNN, image_tensor: torch.Tensor, device: torch.device, fname: str) -> None:
    """Hook first conv output for one image."""
    acts: Dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):
        acts["feat"] = out.detach()

    h = model.block1.conv.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0).to(device))
    h.remove()
    feat = acts["feat"][0].cpu()
    ch = min(6, feat.shape[0])
    fig, axes = plt.subplots(1, ch, figsize=(2 * ch, 2))
    if ch == 1:
        axes = [axes]
    for i in range(ch):
        axes[i].imshow(feat[i].numpy(), cmap="viridis")
        axes[i].axis("off")
    fig.suptitle("Feature maps after first conv (sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def simple_gradcam(
    model: CustomCIFAR10CNN,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
    fname: str,
) -> None:
    """Grad-CAM using gradients of the target score w.r.t. block3 conv activations."""
    model.eval()
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)

    def full_bwd_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = model.block3.conv.register_forward_hook(fwd_hook)
    h2 = model.block3.conv.register_full_backward_hook(full_bwd_hook)

    x = image.unsqueeze(0).to(device).detach().requires_grad_(True)
    logits = model(x)
    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    h1.remove()
    h2.remove()

    fmap = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(fmap.shape[1:], device=device, dtype=fmap.dtype)
    for i, w in enumerate(weights):
        cam = cam + w * fmap[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(np.clip(img, 0, 1))
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(cam_np, cmap="jet")
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def _load_resnet18_pretrained() -> Tuple[nn.Module, bool]:
    """Load ImageNet weights from project cache or torchvision hub."""
    local = MODELS_DIR / "resnet18_imagenet.pth"
    if local.exists():
        model = models.resnet18(weights=None)
        state = torch.load(local, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model, True
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        torch.save(model.state_dict(), local)
        return model, True
    except Exception:
        # Offline fallback: still demonstrates freeze-head then full fine-tune protocol
        return models.resnet18(weights=None), False


def run_transfer_learning(
    device: torch.device,
    epochs_head: int = 3,
    epochs_full: int = 5,
    batch_size: int = 64,
    max_train: int | None = 8000,
    fast: bool = False,
) -> Dict[str, float]:
    """ResNet-18: train classifier head frozen, then fine-tune all layers (224×224 inputs)."""
    img_size = 224
    max_val: int | None = None
    max_test: int | None = None
    if fast:
        epochs_head, epochs_full = 3, 4
        max_train, max_val, max_test, img_size = 4000, 1000, 3000, 160
    train_loader, val_loader, test_loader = cifar10_loaders_imagenet_norm(
        batch_size=batch_size,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
        img_size=img_size,
    )
    model, used_pretrained = _load_resnet18_pretrained()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model.to(device)
    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    t0 = time.time()
    hist_head = train_model(model, train_loader, val_loader, epochs_head, device, tag="resnet18_head", lr=0.05)
    # unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    hist_full = train_model(model, train_loader, val_loader, epochs_full, device, tag="resnet18_full", lr=0.01)
    train_time = time.time() - t0
    y_true, y_pred, conf = evaluate(model, test_loader, device)
    acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    out = {
        "test_accuracy": acc,
        "train_time_sec": train_time,
        "used_imagenet_pretrained": used_pretrained,
        "fast_mode": fast,
        "max_train": max_train,
        "img_size": img_size,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "history_head": hist_head,
        "history_full": hist_full,
    }
    with open(LOGS_DIR / "cifar10_resnet18_eval.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if k not in {"history_head", "history_full"}}, f, indent=2)
    return out


def _write_cnn_summary_partial(payload: Dict[str, object]) -> None:
    with open(LOGS_DIR / "cnn_task5_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_cnn_task5(
    epochs: int = 30,
    skip_custom: bool = False,
    skip_transfer: bool = False,
    fast_transfer: bool = True,
) -> Dict[str, object]:
    """End-to-end Task 5 pipeline (downloads CIFAR-10 on first run)."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: Dict[str, object] = {}

    if skip_custom:
        prior = LOGS_DIR / "cnn_task5_summary.json"
        if prior.exists():
            with open(prior, "r", encoding="utf-8") as f:
                cached = json.load(f)
            summary.update(
                {
                    "custom_cnn_test_accuracy": cached.get("custom_cnn_test_accuracy"),
                    "custom_cnn_train_time_sec": cached.get("custom_cnn_train_time_sec"),
                }
            )
        if not skip_transfer:
            transfer = run_transfer_learning(device, fast=fast_transfer)
            summary["resnet18_test_accuracy"] = transfer["test_accuracy"]
            summary["resnet18_train_time_sec"] = transfer["train_time_sec"]
            summary["resnet18_used_imagenet_pretrained"] = transfer["used_imagenet_pretrained"]
            _write_cnn_summary_partial(
                {
                    "custom_cnn_test_accuracy": summary.get("custom_cnn_test_accuracy"),
                    "custom_cnn_train_time_sec": summary.get("custom_cnn_train_time_sec"),
                    "resnet18_test_accuracy": transfer["test_accuracy"],
                    "resnet18_train_time_sec": transfer["train_time_sec"],
                    "resnet18_used_imagenet_pretrained": transfer["used_imagenet_pretrained"],
                    "status": "complete",
                }
            )
        return summary

    train_loader, val_loader, test_loader = cifar10_loaders()
    model = CustomCIFAR10CNN()
    t0 = time.time()
    hist = train_model(model, train_loader, val_loader, epochs, device, tag="custom_cnn_cifar10", lr=0.1)
    cnn_time = time.time() - t0
    plot_cnn_history(hist, "cnn_custom_history.png")
    y_true, y_pred, conf = evaluate(model, test_loader, device)
    acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, digits=4)
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("CIFAR-10 confusion matrix (custom CNN)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    visualize_first_conv_filters(model, "cnn_first_conv_filters.png", k=8)

    # sample batch for maps
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
    imgs = [test_ds[i][0] for i in range(3)]
    classes = [test_ds[i][1] for i in range(3)]
    for idx, (im, c) in enumerate(zip(imgs, classes)):
        activation_maps(model, im, device, f"cnn_act_map_{idx}.png")
        pred_cls = int(torch.argmax(model(im.unsqueeze(0).to(device)), dim=1).item())
        simple_gradcam(model, im, pred_cls, device, f"cnn_gradcam_{idx}.png")

    # correct vs incorrect with confidence
    y_true, y_pred, conf = evaluate(model, test_loader, device)
    correct_idx = np.where(y_true == y_pred)[0][:4]
    wrong_idx = np.where(y_true != y_pred)[0][:4]

    def show_samples(idxs: np.ndarray, title: str, fname: str) -> None:
        fig, axes = plt.subplots(1, len(idxs), figsize=(2.2 * len(idxs), 2.5))
        if len(idxs) == 1:
            axes = [axes]
        for ax, i in zip(axes, idxs):
            im, _ = test_ds[int(i)]
            im_disp = im.cpu().numpy().transpose(1, 2, 0)
            im_disp = im_disp * np.array(std) + np.array(mean)
            im_disp = np.clip(im_disp, 0, 1)
            ax.imshow(im_disp)
            ax.axis("off")
            ax.set_title(f"true={y_true[i]} pred={y_pred[i]}\nconf={conf[i]:.2f}")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)

    if len(correct_idx):
        show_samples(correct_idx, "Correctly classified samples", "cnn_correct_samples.png")
    if len(wrong_idx):
        show_samples(wrong_idx, "Misclassified samples", "cnn_wrong_samples.png")

    summary.update(
        {
            "custom_cnn_test_accuracy": acc,
            "custom_cnn_train_time_sec": cnn_time,
            "classification_report": report,
            "per_class_metrics_json": report_dict,
        }
    )
    _write_cnn_summary_partial(
        {
            "custom_cnn_test_accuracy": acc,
            "custom_cnn_train_time_sec": cnn_time,
            "resnet18_test_accuracy": None,
            "resnet18_train_time_sec": None,
            "status": "custom_cnn_complete",
        }
    )

    if not skip_transfer:
        transfer = run_transfer_learning(device, fast=fast_transfer)
        summary["resnet18_test_accuracy"] = transfer["test_accuracy"]
        summary["resnet18_train_time_sec"] = transfer["train_time_sec"]
        summary["resnet18_used_imagenet_pretrained"] = transfer["used_imagenet_pretrained"]
        _write_cnn_summary_partial(
            {
                "custom_cnn_test_accuracy": acc,
                "custom_cnn_train_time_sec": cnn_time,
                "resnet18_test_accuracy": transfer["test_accuracy"],
                "resnet18_train_time_sec": transfer["train_time_sec"],
                "resnet18_used_imagenet_pretrained": transfer["used_imagenet_pretrained"],
                "status": "complete",
            }
        )
    return summary
