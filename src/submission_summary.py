"""Aggregate experiment JSON logs into a single summary for the report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import LOGS_DIR, PROJECT_ROOT


def _load(name: str) -> Dict[str, Any]:
    path = LOGS_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_integrated_summary() -> Dict[str, Any]:
    t2a = _load("task2_dataset_a_adult.json")
    t2b = _load("task2_dataset_b_digits.json")
    t3 = _load("task3_dimensionality.json")
    t4 = _load("task4_mlp_summary.json")
    t5 = _load("cnn_task5_summary.json")
    resnet_eval = _load("cifar10_resnet18_eval.json")
    if resnet_eval and t5.get("resnet18_test_accuracy") is None:
        t5["resnet18_test_accuracy"] = resnet_eval.get("test_accuracy")
        t5["resnet18_train_time_sec"] = resnet_eval.get("train_time_sec")
        t5["resnet18_used_imagenet_pretrained"] = resnet_eval.get("used_imagenet_pretrained")
    concepts = _load("mlp_core_concepts.json")

    def best_task2(blob: Dict[str, Any]) -> Dict[str, Any]:
        if not blob:
            return {}
        name = blob["best_model_val_f1_weighted_name"]
        m = blob["models"][name]
        cv = m.get("cv_5fold", m.get("cv_accuracy_mean_std", {}))
        acc_cv = cv.get("accuracy", cv) if isinstance(cv, dict) else {}
        return {
            "name": name,
            "test_metrics": m["test_metrics"],
            "cv_accuracy": acc_cv,
            "best_params": m["best_params"],
        }

    adult_best = best_task2(t2a)
    digits_best = best_task2(t2b)

    pca_adult = next((x for x in t3.get("per_dataset", []) if x["tag"] == "dataset_a_adult"), {})
    pca_digits = next((x for x in t3.get("per_dataset", []) if x["tag"] == "dataset_b_digits"), {})

    mlp_act = "tanh" if t4.get("mlp_tanh_test_acc", 0) >= t4.get("mlp_relu_test_acc", 0) else "relu"

    return {
        "task2": {"adult": adult_best, "digits": digits_best, "imbalance": t2a.get("class_imbalance", {})},
        "task3": {"adult": pca_adult, "digits": pca_digits, "analysis": t3.get("analysis_300_400_words", "")},
        "task4": {
            "best_activation": mlp_act,
            "metrics": t4,
            "core_concepts": concepts,
            "optimizer_plateau": _load("mlp_optimizer_plateau.json"),
        },
        "task5": t5,
    }


def write_summary() -> Path:
    payload = build_integrated_summary()
    out = LOGS_DIR / "integrated_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def _fmt_pm(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def write_report_snippets() -> Path:
    s = build_integrated_summary()
    t2_lines = []
    ab = s["task2"]["adult"]
    db = s["task2"]["digits"]
    if ab:
        cv = ab["cv_accuracy"]
        t2_lines.append(
            f"Adult & {ab['name'].replace('_', ' ')} & {ab['test_metrics']['accuracy']:.4f} & "
            f"{ab['test_metrics']['f1_macro']:.4f} & {ab['test_metrics']['f1_weighted']:.4f} & "
            f"{_fmt_pm(cv['mean'], cv['std'])} \\\\"
        )
    if db:
        cv = db["cv_accuracy"]
        t2_lines.append(
            f"Digits & {db['name'].replace('_', ' ')} & {db['test_metrics']['accuracy']:.4f} & "
            f"{db['test_metrics']['f1_macro']:.4f} & {db['test_metrics']['f1_weighted']:.4f} & "
            f"{_fmt_pm(cv['mean'], cv['std'])} \\\\"
        )
    (PROJECT_ROOT / "report" / "task2_results.tex").write_text("\n".join(t2_lines), encoding="utf-8")

    final_lines = []
    pa = s["task3"]["adult"]
    pd = s["task3"]["digits"]
    t4 = s["task4"]["metrics"]
    t5 = s["task5"]
    act = s["task4"]["best_activation"]
    if ab:
        final_lines.append(
            f"Classical & {ab['name'].replace('_', ' ')} & Adult & {ab['test_metrics']['accuracy']:.4f} \\\\"
        )
    if db:
        final_lines.append(
            f"Classical & {db['name'].replace('_', ' ')} & Digits & {db['test_metrics']['accuracy']:.4f} \\\\"
        )
    if pa:
        pca = pa.get("pca_retention_95pct", pa.get("pca_retention", {}))
        final_lines.append(
            f"PCA (95\\%) & {pca.get('model', '')} & Adult & {pca.get('test_accuracy', 0):.4f} \\\\"
        )
    if pd:
        pca = pd.get("pca_retention_95pct", pd.get("pca_retention", {}))
        final_lines.append(
            f"PCA (95\\%) & {pca.get('model', '')} & Digits & {pca.get('test_accuracy', 0):.4f} \\\\"
        )
    if t4:
        acc = max(t4.get("mlp_relu_test_acc", 0), t4.get("mlp_tanh_test_acc", 0))
        final_lines.append(f"MLP & {act} & Adult & {acc:.4f} \\\\")
    if t5.get("custom_cnn_test_accuracy") is not None:
        final_lines.append(f"CNN scratch & Custom CNN & CIFAR-10 & {t5['custom_cnn_test_accuracy']:.4f} \\\\")
    if t5.get("resnet18_test_accuracy") is not None:
        final_lines.append(f"CNN transfer & ResNet-18 & CIFAR-10 & {t5['resnet18_test_accuracy']:.4f} \\\\")

    out = PROJECT_ROOT / "report" / "generated_metrics.tex"
    out.write_text("\n".join(final_lines), encoding="utf-8")
    return out


def main() -> None:
    path = write_summary()
    tex = write_report_snippets()
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {tex.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
