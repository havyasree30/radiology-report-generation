# CLAUDE.md

Research project: **"A Framework for AI-Based Preliminary Radiology Report Generation"**.

Research quality, correctness and reproducibility take priority over producing a
quick working demo.

## System overview

- **System A: Classification.** DenseNet121 multi-label chest X-ray classifier trained on CheXpert.
- **System B: Evidence retrieval.** Retrieval over IU X-Ray / Open-I reports.
- **System C: Report generation.** Controlled comparison of **Single-Agent RAG** vs **Multi-Agent RAG** for preliminary radiology report generation.
- A research-grade web application comes last, only after the ML/RAG pipeline is finalized.

## Before starting any new major phase

Inspect the existing code, `configs/` and `results/` to find out what has actually
been done and what previous experiments actually showed. Never assume prior results,
split definitions or chosen policies. Read them from the repository.

## Permanent research rules

**Integrity**
- Build the project from scratch. Do not copy from or assume any prior implementation.
- Do not assume results. Never fabricate metrics. Never hard-code experimental results.
- Every research figure/table must be generated from actual experiment outputs, by code in this repo.
- Preserve reproducibility: fixed and logged seeds, configs saved alongside outputs, pinned dependencies.
- Any heuristic must be explicitly documented and experimentally evaluated.

**Data handling**
- Keep raw datasets outside Git. They are read-only. Access them only via `configs/paths.yaml` (never hard-coded paths).
- Prevent patient-level data leakage: all images of a patient belong to exactly one split.
- Never use the test set for model selection, threshold selection, calibration fitting, or hyperparameter tuning.
- Validation/test distributions must not be artificially balanced. Rebalancing, if any, applies to training only.

**CheXpert labels / classifier**
- CheXpert is **multi-label** classification, not multi-class.
- DenseNet121 outputs **14 independent sigmoid probabilities**. Do **NOT** use softmax across the 14 observations.
- Probabilities do not need to sum to 1. Multiple observations may be positive simultaneously.
- Preserve uncertain (−1) and blank labels as-is until an explicit experimental uncertainty policy is selected. Never silently convert them.

**Evaluation**
- Evaluate classifier, retrieval and report generation separately.
- Semantic similarity (e.g. embedding/text-overlap scores) is not proof of clinical correctness.
- Do not assume Multi-Agent RAG will outperform Single-Agent RAG. Compare under controlled conditions with statistical testing.

**Scope**
- The final application is a research prototype, not an autonomous clinical diagnostic system.

## Environment

- Python 3.11 virtual env at `.venv/` (Windows: `.venv\Scripts\python.exe`). Do not use the global Python.
- Current phase dependencies: `requirements-eda.txt`. Heavy deps (PyTorch/CUDA, FAISS, LLM SDKs, web stack)
  are added only when their phase begins, and only with user approval.
- Jupyter kernel: `radiology-report-generation`.
- Tests: `.venv\Scripts\python.exe -m pytest`.

## Repository conventions

- `configs/`: `paths.example.yaml` is the template. `paths.yaml` is local and git-ignored.
- `data/splits/`: generated split definitions (tracked). `data/cache/`: regenerable (ignored). No raw data in `data/`.
- `src/`: importable code. `scripts/`: entry points. `notebooks/`: exploration only; reusable logic belongs in `src/`.
- `results/eda/{figures,tables,summaries}`: EDA outputs. Checkpoints and caches under `results/` are git-ignored.
- Do not create placeholder modules. Add files only when they have a current purpose.
