"""Global paths and random seeds for reproducibility."""
import os
from pathlib import Path

# Headless-safe plotting (prevents GUI backend crashes in CI/servers)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"

for _p in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
