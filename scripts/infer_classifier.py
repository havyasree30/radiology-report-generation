"""Single-image inference: validate -> preprocess -> DenseNet121 -> 14 sigmoids -> thresholds
-> structured findings -> query-ready output. Prints JSON.

    .venv\\Scripts\\python.exe -m scripts.infer_classifier <image> [--config ...]

Fails loudly if the checkpoint or the threshold file is missing.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from src.classification.checkpointing import load_checkpoint
from src.classification.findings import build_query, build_structured_findings, label_predictions
from src.classification.model import predict_proba
from src.classification.thresholds import load_thresholds
from src.classification.trainer import load_config
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.utils.config import PROJECT_ROOT
from src.validation.image_validator import ImageValidator


def infer(image_path: str, cfg: dict, device: torch.device | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = PROJECT_ROOT / cfg["output_dir"]
    model, meta = load_checkpoint(out / "checkpoints" / "best.pt", device)
    # Thresholds fitted under the SAME canonical policy that predict_proba uses below.
    thresholds = load_thresholds(out / "evaluation" / cfg["inference"]["policy"] / "thresholds" / "youden_j_thresholds.json")
    val = ImageValidator().validate(image_path)
    result = {"image": str(image_path), "validation": val.to_dict(), "checkpoint_epoch": meta["epoch"],
              "inference_policy": cfg["inference"]["policy"]}
    if cfg["inference"]["policy"] != "fp32_strict":
        raise ValueError("single-image inference is implemented for the canonical fp32_strict policy only")
    if not val.valid:
        result["predictions"] = None
        return result
    x = EvalTransform(PreprocessConfig.from_dict(meta["preprocessing"]))(val.image).unsqueeze(0).to(device)
    probs = predict_proba(model, x)[0].cpu().numpy().astype(float)
    preds = label_predictions(probs, thresholds)
    structured = build_structured_findings(preds, cfg["findings"]["borderline_margin"])
    result.update(predictions=preds, structured_findings=structured, query=build_query(structured))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--config", default="configs/classifier/densenet121.yaml")
    args = ap.parse_args()
    res = infer(args.image, load_config(PROJECT_ROOT / args.config))
    json.dump(res, sys.stdout, indent=2)
    return 0 if res["validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
