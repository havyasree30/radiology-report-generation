# data/

The source datasets are stored here **locally** but are **never committed**.
`.gitignore` excludes everything in `data/` except this README and `splits/`.
Source dataset files are **read-only**: no code in this project writes,
renames, moves, resizes or relabels them.

All code locates the datasets through `configs/paths.yaml` (created from
`configs/paths.example.yaml`, git-ignored), never through hard-coded paths.

| Path | Contents | Tracked in Git |
|---|---|---|
| `CheXpert-v1.0-small/` | CheXpert images (`train/`, `valid/`) and `valid.csv` | No |
| `train.csv` | CheXpert training labels (stored outside the release folder in this download) | No |
| `IU X-ray/` | `indiana_reports.csv`, `indiana_projections.csv`, `images/images_normalized/` | No |
| `IU X-ray.zip` | Original IU X-Ray archive | No |
| `cache/` | Regenerable artifacts, e.g. the full image-audit table | No |
| `splits/chexpert/` | Patient-group split manifests, split metadata, training-only `pos_weight`s | Yes |
| `splits/iu_xray/` | Report-level split manifests and the train-only retrieval-corpus list | Yes |

## Split invariants (enforced by code and tests)

**CheXpert**
- The unit of assignment is a patient.
- Patients who share a byte-identical image are merged into one split group.
- A group never spans partitions.
- The official validation CSV is kept intact as the separate `official_valid` set.

**IU X-Ray**
- The unit of assignment is the report `uid`.
- All images of a report stay in the same partition.
- Only training reports may enter the retrieval corpus.

Split files are produced by `scripts/eda_03_chexpert_splits.py` and
`scripts/eda_04_iu_xray.py`, and can be regenerated deterministically.
