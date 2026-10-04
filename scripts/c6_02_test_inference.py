"""C6 step 3: the ONE locked-test inference with the frozen DenseNet-121 (fp32_strict, no TTA, no ensembling).

    .venv\\Scripts\\python.exe -m scripts.c6_02_test_inference

Refuses to run unless the freeze manifest exists and every frozen file still matches its recorded hash,
and refuses to run a second time (existing test predictions are never overwritten). Only raw scores are
produced here; no metric is computed and nothing is fitted.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.classification import inference_policy
from src.classification.checkpointing import load_checkpoint
from src.classification.dataset import CheXpertDataset, label_policy_from_config, load_split_frame
from src.classification.evaluator import predict
from src.classification.final_test import sha256_file, verify_freeze
from src.classification.serialization import load_predictions_csv, load_predictions_npz, roundtrip_report, save_predictions
from src.classification.trainer import load_config
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT, load_paths

log = logging.getLogger("c6_inference")
OUT = PROJECT_ROOT / "results/classification/experiments/c6_final_test"
C2B = PROJECT_ROOT / "results/classification/experiments/c2b_sqrt_weighted_bce"


def main() -> int:
    setup_logging()
    manifest_path = OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("freeze manifest missing: freeze the classifier before opening the test set")
    if list(OUT.glob("test_raw_predictions*")):
        raise RuntimeError("test predictions already exist: the locked test inference is one-shot and is never repeated")
    verify_freeze(manifest_path, PROJECT_ROOT)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_config(C2B / "config.yaml")
    if cfg["inference"]["policy"] != manifest["inference_policy"] or cfg["preprocessing"] != manifest["preprocessing"]:
        raise RuntimeError("config differs from the frozen manifest")
    policy = inference_policy.get(manifest["inference_policy"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    model, meta = load_checkpoint(PROJECT_ROOT / manifest["checkpoint"], device)
    if meta["epoch"] != manifest["checkpoint_epoch"]:
        raise RuntimeError("checkpoint epoch differs from the manifest")
    paths = load_paths()
    frame = load_split_frame(paths, cfg["data"], "test", allow_locked=True)      # the single sanctioned unlock
    ds = CheXpertDataset(frame, paths.chexpert_image_base, label_policy_from_config(cfg["label_policy"]),
                         EvalTransform(PreprocessConfig.from_dict(cfg["preprocessing"])))
    log.info("test images: %d, patients: %d", len(frame), frame["patient_id"].nunique())
    t0 = time.time()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    pred = predict(model, loader, device, policy=policy)
    elapsed = time.time() - t0
    p, m = pred["probs"], pred["mask"]
    raw = frame[list(manifest["label_order"])].to_numpy(dtype=np.float64)
    if not np.array_equal(pred["index"], np.arange(len(frame))):
        raise RuntimeError("loader returned rows out of order")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise RuntimeError("non-finite or out-of-range scores")
    files = save_predictions(OUT, "test_raw_predictions", frame["Path"], frame["patient_id"], p, raw, m,
                             meta={"policy": policy.name, "checkpoint_epoch": meta["epoch"], "split": "test",
                                   "freeze_manifest_sha256": sha256_file(manifest_path)})
    rt = {"npz": roundtrip_report(p, load_predictions_npz(files["npz"])["probs"], np.full(14, 0.5)),
          "csv_17_digits": roundtrip_report(p, load_predictions_csv(files["csv"])["probs"], np.full(14, 0.5))}
    if not (rt["npz"]["bit_identical"] and rt["csv_17_digits"]["bit_identical"]):
        raise AssertionError(f"serialization changed scores: {rt}")
    run = {"split": "test", "n_images": int(len(frame)), "n_patients": int(frame["patient_id"].nunique()),
           "n_excluded_by_exclusion_list": int(frame.attrs.get("n_excluded", 0)), "inference_policy": policy.as_dict(),
           "backend_flags": inference_policy.current_flags(), "test_time_augmentation": False, "ensembling": False,
           "n_inference_passes": 1, "seconds": round(elapsed, 1), "score_min": float(p.min()), "score_max": float(p.max()),
           "serialization_roundtrip": rt, "checkpoint_sha256_at_inference": sha256_file(PROJECT_ROOT / manifest["checkpoint"]),
           "freeze_manifest_sha256": sha256_file(manifest_path), "gpu": torch.cuda.get_device_name(0),
           "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "test_inference_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    log.info("done: %s", json.dumps({k: run[k] for k in ("n_images", "n_patients", "seconds", "score_min", "score_max")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
