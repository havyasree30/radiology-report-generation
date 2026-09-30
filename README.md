# A Framework for AI-Based Preliminary Radiology Report Generation

## Research objective

To investigate whether a pipeline combining multi-label chest X-ray
classification with retrieval of evidence from real radiology reports can
produce useful **preliminary** radiology reports, and to compare
**Single-Agent** and **Multi-Agent** Retrieval-Augmented Generation (RAG) for
that task under controlled, reproducible conditions.

This is a research prototype. It is **not** an autonomous clinical diagnostic
system.

## Planned architecture

| System | Role | Data |
|---|---|---|
| **A: Classification** | DenseNet121 multi-label classifier: 14 independent sigmoid outputs, one per CheXpert observation (no softmax; findings may co-occur). | CheXpert |
| **B: Evidence retrieval** | Retrieves relevant report text to ground generation. | IU X-Ray / Open-I |
| **C: Report generation** | Controlled comparison of Single-Agent RAG vs Multi-Agent RAG for drafting preliminary reports. | Outputs of A + B |

Each system is evaluated separately (classification, retrieval, generation),
followed by ablations, error analysis, robustness analysis and statistical
comparison.

## Current status

**Phase 1: Dataset Exploration and EDA.**

The repository and environment are set up. No models have been trained and
no experimental results exist yet.

## Datasets

- **CheXpert**: chest radiographs with 14 observation labels (positive / negative / uncertain / blank).
- **IU X-Ray / Open-I**: chest radiographs paired with free-text radiology reports.

Datasets are **not** included in this repository. Configure their locations
locally:

```bash
cp configs/paths.example.yaml configs/paths.yaml   # then edit the paths
```

## Environment setup (Phase 1)

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-eda.txt
```

Jupyter kernel: `radiology-report-generation`.

## Repository layout

```
configs/    path & experiment configuration (machine-specific files git-ignored)
data/       split definitions and caches only (no raw data; see data/README.md)
notebooks/  exploratory notebooks
src/        importable project code
scripts/    runnable entry points
tests/      pytest tests
results/    generated outputs (EDA figures/tables/summaries, later experiments)
```
