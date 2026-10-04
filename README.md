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

### Pipeline

```
Chest X-ray
  → Image Validation            (src/validation/image_validator.py)
  → Preprocessing               (src/preprocessing/transforms.py: aspect-preserving 224×224, ImageNet norm)
  → DenseNet121                 (src/classification/model.py: 14 logits)
  → Sigmoid                     (14 independent probabilities; no softmax, no renormalisation)
  → Youden's J thresholds       (src/classification/thresholds.py: fitted on validation only)
  → Structured Findings         (src/classification/findings.py)
  → Query Builder               (src/classification/findings.py: text is built, NOT embedded)
  → MiniLM embedding                          [FUTURE, Phase 3]
  → Qdrant over IU X-Ray TRAIN reports only   [FUTURE, Phase 3]
  → Multi-Agent RAG                           [FUTURE]
  → Meerkat-7B-v1.0                           [FUTURE]
  → Preliminary FINDINGS + IMPRESSION report  [FUTURE]
```

Each system is evaluated separately (classification, retrieval, generation),
followed by ablations, error analysis, robustness analysis and statistical
comparison.

## Current status

**Phase 1: Dataset Exploration and EDA. Analysis complete, awaiting review.**

The EDA, data-integrity audit, leak-free splits and imbalance analysis are
done. See [`results/eda/EDA_REPORT.md`](results/eda/EDA_REPORT.md). No model
has been trained; Phase 2 (DenseNet121) has not started.

## Datasets

- **CheXpert v1.0-small**: chest radiographs with 14 observation labels (positive / negative / uncertain / blank).
- **IU X-Ray** (Kaggle CSV layout of the Indiana University / Open-I collection): chest radiographs paired with free-text reports.

The datasets are stored locally under `data/` but are **never committed**:
`.gitignore` excludes everything in `data/` except `data/README.md` and the
generated `data/splits/`. Source dataset files are treated as read-only.
Configure their locations in a local, git-ignored file:

```bash
cp configs/paths.example.yaml configs/paths.yaml   # then edit the paths
```

## Reproducing Phase 1

Run from the project root with the project environment:

```bash
.venv\Scripts\python.exe -m scripts.eda_01_image_audit
.venv\Scripts\python.exe -m scripts.eda_02_chexpert
.venv\Scripts\python.exe -m scripts.eda_03_chexpert_splits
.venv\Scripts\python.exe -m scripts.eda_04_iu_xray
.venv\Scripts\python.exe -m scripts.eda_05_summary
.venv\Scripts\python.exe -m pytest
```

- **Image audit cache:** stage 1 decodes and hashes every image, which takes several minutes. It caches its results in `data/cache/`; pass `--force` to rescan.
- **Split selection:** thresholds and the pre-declared split-selection rule are in `configs/eda.yaml`.

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
configs/    paths template, EDA settings, concept lexicon, Phase 2 candidate config
data/       local datasets (git-ignored), caches (git-ignored), split manifests (tracked)
notebooks/  exploratory notebooks
src/        importable code: data loading, splits, label policies, losses, analysis
scripts/    runnable pipeline stages (eda_01 ... eda_05)
tests/      pytest tests (label handling, splits, leakage, losses, retrieval invariant)
results/    generated outputs (EDA report, figures, tables, summaries)
```
