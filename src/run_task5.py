"""Task 5: CIFAR-10 CNN pipeline (long-running; requires network first time for download)."""
from __future__ import annotations

import argparse
import json

from .cnn_task5 import run_cnn_task5
from .config import LOGS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 — CIFAR-10 CNN + transfer learning")
    parser.add_argument("--epochs", type=int, default=30, help="Custom CNN training epochs (min 30)")
    parser.add_argument("--transfer-only", action="store_true", help="Skip custom CNN; run ResNet-18 fine-tune only")
    parser.add_argument("--custom-only", action="store_true", help="Skip transfer learning phase")
    parser.add_argument(
        "--full-transfer",
        action="store_true",
        help="Use full CIFAR-10 train set and more ResNet epochs (slow on CPU)",
    )
    import sys
    args = parser.parse_args(args=[] if 'ipykernel' in sys.modules else None)
    epochs = max(30, args.epochs)
    summary = run_cnn_task5(
        epochs=epochs,
        skip_custom=args.transfer_only,
        skip_transfer=args.custom_only,
        fast_transfer=not args.full_transfer,
    )
    with open(LOGS_DIR / "task5_run_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: summary[k] for k in summary if k not in ("classification_report", "per_class_metrics_json")}, f, indent=2)


if __name__ == "__main__":
    main()
