"""Task 4: MLP on Dataset A (Adult) using preprocessed tensors from the shared pipeline."""
from __future__ import annotations

from .data import load_dataset_a_adult
from .evaluation import save_json
from .mlp_train import run_mlp_experiments
from .preprocessing import preprocess_tabular


def main() -> None:
    ds = load_dataset_a_adult()
    p = preprocess_tabular(ds)
    out = run_mlp_experiments(
        p.X_train,
        p.y_train,
        p.X_val,
        p.y_val,
        p.X_test,
        p.y_test,
    )
    serializable = {k: v for k, v in out.items() if isinstance(v, (float, int, str, dict, list))}
    save_json(serializable, "task4_mlp_summary.json")


if __name__ == "__main__":
    main()
