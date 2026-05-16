# Machine Learning and Deep Learning — Course Project (Assignment 2)

This repository implements Tasks 2–5 in modular Python under `src/`, saves figures and logs under `results/`, and includes LaTeX source for the full six-task report (`report/report.tex`). Your course requires you to compile `report.pdf` locally and to disclose any AI assistance per the syllabus.

## Environment

Python 3.10+ recommended.

```bash
cd /Users/AmirkhanOral/Downloads/ML_DL_Project_Assignment2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the project root so that `./data` (CIFAR-10) and `results/` paths resolve correctly.

## Task 2 — Classical classification

```bash
python -m src.run_task2
```

Produces metrics, ROC figures, and confusion matrices for Dataset A (Adult Census Income, OpenML) and Dataset B (sklearn Digits, 10 classes).

## Task 3 — Dimensionality reduction

**Run Task 2 first** so `results/logs/task2_*.json` exists (Task 3 reloads the same best model + hyperparameters on PCA features).

```bash
python -m src.run_task3
```

Runs PCA (variance curve + 2D), t-SNE, UMAP, LDA visualizations and retrains tuned classifiers on PCA features retaining 95% variance.

## Task 4 — MLP fundamentals

```bash
python -m src.run_task4
```

Trains depth-3 MLPs with BatchNorm and Dropout on Dataset A, compares ReLU vs Tanh, and compares SGD vs Adam vs RMSprop with logged curves.

## Task 5 — CNN on CIFAR-10

```bash
python -m src.run_task5
```

Expect roughly 30+ minutes on CPU (faster on GPU). Downloads CIFAR-10 on first run, trains the custom CNN for 30 epochs with augmentation and cosine learning-rate schedule, runs ResNet-18 transfer learning (frozen head then full fine-tune), and writes filter maps, activation maps, Grad-CAMs, and qualitative success/failure panels.

If the custom CNN checkpoint already exists but transfer learning was interrupted (fast CPU-friendly defaults: 4k train images, 2+3 epochs):

```bash
python -m src.run_task5 --transfer-only
```

For full-dataset transfer (slow): `python -m src.run_task5 --transfer-only --full-transfer`

Optional ImageNet weights (with internet): `python scripts/download_resnet18_weights.py`

## Finalize tables (no training)

```bash
python scripts/finalize_submission.py
cd report && pdflatex report.tex && pdflatex report.tex
```

## Build integrated summary + LaTeX table snippets

After Tasks 2–5:

```bash
python -m src.submission_summary
```

**Note:** ImageNet weights for ResNet-18 require internet (`python scripts/download_resnet18_weights.py`). Without them, `--transfer-only` still runs the freeze/unfreeze protocol on a random-init backbone (fast CPU profile).

## Report

```bash
cd report
pdflatex report.tex
bibtex report   # optional if you add a .bib file
pdflatex report.tex
pdflatex report.tex
```

## Notebooks

Jupyter notebooks under `notebooks/` import the same `src` modules so you can re-run cells interactively; clear outputs before submission if your instructor requires a clean re-run trace.

## Checkpoint loading (grading checklist)

After training, demonstrate restoring weights:

```bash
python -m src.demo_load_checkpoint
```

## Academic integrity

Your PDF states that undisclosed AI-generated work is prohibited. If you use this repository as a starting point, disclose that clearly in your report or cover page per instructor rules, and rewrite prose in your own words where required.
