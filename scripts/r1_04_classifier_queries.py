"""R1 query mode B: run the FROZEN classifier on the frontal image of every retrieval-VALIDATION study and build the
two classifier-driven queries (all frozen positives; precision-aware gated). External / domain-shift application:
nothing is tuned on IU X-Ray, and test-split studies are never opened.

    .venv\\Scripts\\python.exe -m scripts.r1_04_classifier_queries
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.classification import inference_policy
from src.classification.checkpointing import load_checkpoint
from src.classification.final_test import apply_frozen_pipeline, load_frozen, sha256_file, verify_freeze
from src.classification.labels import LABELS
from src.classification.model import predict_proba
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.retrieval.queries import ABNORMAL, classifier_query, gated_query
from src.utils.config import PROJECT_ROOT, load_paths
from src.validation.image_validator import ImageValidator

EXP = PROJECT_ROOT / "results/classification/experiments"
OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
C6 = EXP / "c6_final_test"


def main() -> int:
    manifest_path = C6 / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    verify_freeze(manifest_path, PROJECT_ROOT)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy, cal = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    gates = json.loads((OUT / "precision_aware_query_gates.json").read_text(encoding="utf-8"))["gates"]
    gate_val = {l: gates[l]["gate_calibrated_probability"] for l in ABNORMAL}
    split = json.loads((OUT / "retrieval_split.json").read_text(encoding="utf-8"))
    val_uids = set(split["study_ids"]["validation"])
    sf = pd.read_csv(OUT / "iu_study_findings.csv", dtype={"uid": str})
    vs = sf[(sf.partition == "validation") & (sf.frontal_image.fillna("") != "")].copy()
    assert set(vs.uid) <= val_uids and not (set(vs.uid) & set(split["study_ids"]["locked_test"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(PROJECT_ROOT / man["checkpoint"], device)
    assert sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"] and meta["epoch"] == man["checkpoint_epoch"]
    tf = EvalTransform(PreprocessConfig.from_dict(meta["preprocessing"]))
    assert {k: meta["preprocessing"][k] for k in man["preprocessing"]} == man["preprocessing"]
    inference_policy.apply(inference_policy.CANONICAL)
    images_dir = load_paths().iu_images_dir
    validator = ImageValidator()
    xs, ok, notes = [], [], []
    for fn in vs.frontal_image:
        res = validator.validate(images_dir / fn)
        ok.append(bool(res.valid))
        notes.append(";".join(w["code"] for w in res.warnings) + (";ERROR:" + ";".join(e["code"] for e in res.errors) if not res.valid else ""))
        xs.append(tf(res.image) if res.valid else None)
    good = [i for i, x in enumerate(xs) if x is not None]
    scores = np.full((len(vs), 14), np.nan)
    with torch.no_grad():
        for s in range(0, len(good), 32):
            idx = good[s:s + 32]
            scores[idx] = predict_proba(model, torch.stack([xs[i] for i in idx]).to(device)).cpu().numpy().astype(np.float64)
    keep = ~np.isnan(scores).any(1)
    o = apply_frozen_pipeline(scores[keep], policy, cal)
    rows = []
    for k, (_, r) in enumerate(vs[keep].iterrows()):
        pf = {l: bool(o["pred_final"][k, j]) for j, l in enumerate(LABELS)}
        pb = {l: float(o["prob"][k, j]) for j, l in enumerate(LABELS)}
        qa, qg = classifier_query(pf), gated_query(pf, pb, gate_val)
        row = {"uid": r.uid, "frontal_image": r.frontal_image,
               "query_all_positive": qa["query"], "query_all_positive_status": qa["status"], "query_all_positive_findings": ";".join(qa["findings"]),
               "query_gated": qg["query"], "query_gated_status": qg["status"], "query_gated_findings": ";".join(qg["findings"]),
               "low_confidence_positive_findings": ";".join(qg["low_confidence"])}
        for j, l in enumerate(LABELS):
            row[f"{l}__score"], row[f"{l}__prob"] = float(o["scores"][k, j]), float(o["prob"][k, j])
            row[f"{l}__pred_final"] = int(o["pred_final"][k, j])
            row[f"{l}__state"] = qg["states"][l] if l in qg["states"] else ("positive" if pf[l] else "negative")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "iu_validation_classifier_outputs.csv", index=False, float_format="%.17g")
    run = {"n_validation_studies_with_frontal": int(len(vs)), "n_images_failing_validator": int((~np.array(ok)).sum()), "n_classified": int(len(df)),
           "device": str(device), "inference_policy": "fp32_strict", "split_used": "validation only; locked retrieval-test studies not opened",
           "checkpoint_sha256": man["checkpoint_sha256"], "classifier_tuned_on_iu": False,
           "image_validator_warnings": pd.Series([n for n in notes if n]).value_counts().head(10).to_dict(),
           "note": "External / domain-shift application of the frozen CheXpert classifier to IU X-Ray (PNG 'images_normalized')."}
    (OUT / "iu_classifier_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=1))
    print(df.query_all_positive_status.value_counts().to_dict(), df.query_gated_status.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
