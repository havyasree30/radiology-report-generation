"""Run the image validator over all frontal TRAIN and VAL images (never the locked test split)
to measure how often it rejects or warns on real radiographs.

    .venv\\Scripts\\python.exe -m scripts.audit_image_validator
Outputs: results/classification/validator_audit/{summary.csv, flagged.csv}
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT, load_paths
from src.validation.image_validator import ImageValidator

_V = None


def _check(path: str) -> dict:
    global _V
    if _V is None:
        _V = ImageValidator()
    r = _V.validate(path)
    return {"valid": r.valid, "errors": ";".join(e["code"] for e in r.errors),
            "warnings": ";".join(w["code"] for w in r.warnings)}


def main() -> int:
    paths = load_paths()
    man = pd.read_csv(PROJECT_ROOT / "data/splits/chexpert/chexpert_image_manifest.csv.gz")
    rows = man[man["split"].isin(["train", "val"]) & (man["Frontal/Lateral"] == "Frontal")].reset_index(drop=True)
    files = [str(paths.chexpert_image_base / p) for p in rows["Path"]]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        res = list(tqdm(ex.map(_check, files, chunksize=256), total=len(files), mininterval=10))
    df = pd.concat([rows[["Path", "split"]], pd.DataFrame(res)], axis=1)
    out = PROJECT_ROOT / "results/classification/validator_audit"
    out.mkdir(parents=True, exist_ok=True)
    df[(~df["valid"]) | (df["warnings"] != "")].to_csv(out / "flagged.csv", index=False)
    summ = []
    for split, g in df.groupby("split"):
        summ.append({"split": split, "code": "_images", "kind": "total", "count": len(g)})
        summ.append({"split": split, "code": "_hard_rejected", "kind": "error", "count": int((~g["valid"]).sum())})
        for kind, col in (("error", "errors"), ("warning", "warnings")):
            codes = g[col].str.split(";").explode()
            for c, n in codes[codes != ""].value_counts().items():
                summ.append({"split": split, "code": c, "kind": kind, "count": int(n)})
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
