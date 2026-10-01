"""Train the DenseNet121 baseline.

    .venv\\Scripts\\python.exe -m scripts.train_classifier --config configs/classifier/densenet121.yaml
    .venv\\Scripts\\python.exe -m scripts.train_classifier --smoke   # tiny subset, NOT an experiment

The smoke run writes to results/classification/_smoke/ (git-ignored) and must not be reported.
"""

from __future__ import annotations

import argparse
import copy
import logging

from src.classification.trainer import load_config, run_training
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier/densenet121.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(PROJECT_ROOT / args.config)
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["training"].update(max_epochs=2, early_stopping_patience=99)
        out = PROJECT_ROOT / "results" / "classification" / "_smoke"
        summary = run_training(cfg, out, subset={"train": 512, "val": 256})
    else:
        summary = run_training(cfg, PROJECT_ROOT / cfg["output_dir"])
    logging.getLogger("train").info("summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
