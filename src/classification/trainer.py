"""Config-driven DenseNet121 training with validation-only model selection."""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.classification.checkpointing import save_checkpoint
from src.classification.dataset import (CheXpertDataset, label_policy_from_config, load_split_frame, seed_worker)
from src.classification.evaluator import predict
from src.classification.inference_policy import CANONICAL
from src.classification.inference_policy import apply as apply_inference_policy
from src.classification.labels import LABELS
from src.classification.losses import build_loss, temper_pos_weight
from src.classification.metrics import per_class_metrics, summary_metrics
from src.classification.model import build_densenet121
from src.data.label_policies import compute_pos_weight
from src.preprocessing.transforms import AugmentConfig, EvalTransform, PreprocessConfig, TrainTransform
from src.utils.config import PROJECT_ROOT, SPLITS_DIR, load_paths
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("trainer")


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def cuda_environment() -> dict:
    env = {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=torch.cuda.get_device_name(0), gpu_total_memory_gb=p.total_memory / 1e9,
                   compute_capability=f"{p.major}.{p.minor}")
    return env


def train_pos_weight(train_ds: CheXpertDataset) -> tuple[torch.Tensor, pd.DataFrame]:
    """pos_weight = N_neg / N_pos over entries that enter the loss, TRAINING partition only."""
    r = compute_pos_weight(train_ds.targets.numpy(), train_ds.mask.numpy())
    df = pd.DataFrame({"observation": LABELS, "n_positive": r["n_positive"], "n_negative": r["n_negative"],
                       "n_masked": r["n_masked"], "pos_weight_raw": r["pos_weight"], "degenerate": r["degenerate"]})
    return torch.tensor(r["pos_weight"], dtype=torch.float32), df


def run_training(cfg: dict, out_dir: Path, subset: dict | None = None, require_cuda: bool = True) -> dict:
    """Train; returns a summary dict. ``subset={'train': n, 'val': n}`` is for smoke tests only."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to train on CPU (see Phase 2 instructions).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    # TF32 off for the whole run so training and the per-epoch canonical validation share
    # backend settings (C1 itself was trained with PyTorch's default cuDNN TF32 = on).
    apply_inference_policy(CANONICAL)
    paths = load_paths()
    dcfg, tcfg = cfg["data"], cfg["training"]

    policy = label_policy_from_config(cfg["label_policy"])
    pcfg = PreprocessConfig.from_dict(cfg["preprocessing"])
    tr_frame = load_split_frame(paths, dcfg, dcfg["train_split"])
    va_frame = load_split_frame(paths, dcfg, dcfg["val_split"])
    counts = {"train_images": len(tr_frame), "train_excluded": tr_frame.attrs["n_excluded"],
              "train_patients": int(tr_frame["patient_id"].nunique()),
              "val_images": len(va_frame), "val_excluded": va_frame.attrs["n_excluded"],
              "val_patients": int(va_frame["patient_id"].nunique())}
    if set(tr_frame["patient_id"]) & set(va_frame["patient_id"]):
        raise AssertionError("patient overlap between train and val")
    if subset:
        tr_frame = tr_frame.sample(n=min(subset["train"], len(tr_frame)), random_state=cfg["seed"]).reset_index(drop=True)
        va_frame = va_frame.sample(n=min(subset["val"], len(va_frame)), random_state=cfg["seed"]).reset_index(drop=True)
    train_ds = CheXpertDataset(tr_frame, paths.chexpert_image_base, policy,
                               TrainTransform(pcfg, AugmentConfig.from_dict(cfg["augmentation"])))
    val_ds = CheXpertDataset(va_frame, paths.chexpert_image_base, policy, EvalTransform(pcfg))

    raw_pw, pw_df = train_pos_weight(train_ds)
    pw_df["pos_weight_used"] = temper_pos_weight(raw_pw, cfg["loss"].get("pos_weight_tempering", "none")).numpy() \
        if cfg["loss"]["type"] == "weighted_bce" else np.nan
    pw_df.to_csv(out_dir / "train_pos_weights.csv", index=False)
    loss_fn = build_loss(cfg["loss"], raw_pw.to(device) if cfg["loss"]["type"] == "weighted_bce" else None).to(device)

    g = torch.Generator().manual_seed(cfg["seed"])
    nw = dcfg.get("num_workers", 4)
    common = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                  worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"] * 2, shuffle=False, **common)

    model = build_densenet121(len(LABELS), cfg["model"]["pretrained"]).to(device)
    ocfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"],
                                  betas=tuple(ocfg["betas"]))
    scfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scfg["mode"], factor=scfg["factor"],
                                                           patience=scfg["patience"])
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accum = int(tcfg["grad_accumulation"])

    env = {"project": environment_snapshot(cfg["seed"]), "cuda": cuda_environment(),
           "physical_batch": tcfg["batch_size"], "effective_batch": tcfg["batch_size"] * accum}
    meta_base = {
        "architecture": "densenet121", "num_labels": len(LABELS), "label_order": list(LABELS),
        "val_metric_name": tcfg["monitor"], "preprocessing": cfg["preprocessing"], "augmentation": cfg["augmentation"],
        "label_policy": cfg["label_policy"], "loss": cfg["loss"], "optimizer": ocfg, "scheduler": scfg,
        "seed": cfg["seed"], "git_commit": env["project"]["git"]["commit"], "counts": counts,
        "pos_weight_raw": dict(zip(LABELS, map(float, raw_pw))), "subset": subset,
    }

    history, best, bad_epochs = [], -math.inf, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        run_loss, n_batches, grad_norms, nonfinite = 0.0, 0, [], 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, m, _) in enumerate(train_loader, 1):
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y, m)
            if not torch.isfinite(loss):
                nonfinite += 1
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            scaler.scale(loss / accum).backward()
            if step % accum == 0:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))  # measure only
                if torch.isfinite(gn):
                    grad_norms.append(gn.item())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            run_loss += loss.item()
            n_batches += 1
            if step % 500 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), run_loss / n_batches)
        train_time = time.time() - t0
        t1 = time.time()
        pred = predict(model, val_loader, device, loss_fn, CANONICAL)  # model selection on canonical probabilities
        model.train()
        sm = summary_metrics(pred["targets"], pred["probs"], pred["mask"])
        val_time = time.time() - t1
        metric = sm["macro_auroc"]
        row = {"epoch": epoch, "train_loss": run_loss / max(n_batches, 1), "val_loss": pred["loss"],
               "val_macro_auroc": metric, "val_micro_auroc": sm["micro_auroc"], "val_macro_auprc": sm["macro_auprc"],
               "val_micro_auprc": sm["micro_auprc"], "lr": optimizer.param_groups[0]["lr"],
               "grad_norm_median": float(np.median(grad_norms)) if grad_norms else np.nan,
               "grad_norm_p99": float(np.percentile(grad_norms, 99)) if grad_norms else np.nan,
               "grad_norm_max": float(np.max(grad_norms)) if grad_norms else np.nan,
               "amp_skipped_or_nonfinite_steps": n_batches // accum - len(grad_norms),
               "train_seconds": train_time, "val_seconds": val_time,
               "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else np.nan}
        for rr in per_class_metrics(pred["targets"], pred["probs"], pred["mask"]).itertuples():
            row[f"auroc::{rr.observation}"] = rr.auroc
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        log.info("epoch %d | train %.4f | val %.4f | macroAUROC %.4f | %.0fs+%.0fs | peak %.2f GB",
                 epoch, row["train_loss"], row["val_loss"], metric, train_time, val_time, row["peak_gpu_mem_gb"])

        meta = dict(meta_base, epoch=epoch, val_metric=metric, val_loss=pred["loss"], train_loss=row["train_loss"])
        save_checkpoint(out_dir / "checkpoints" / "final.pt", model, meta, optimizer, scheduler)
        if metric > best:
            best, bad_epochs = metric, 0
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, dict(meta, selected_as="best"))
        else:
            bad_epochs += 1
        scheduler.step(metric)
        if bad_epochs >= tcfg["early_stopping_patience"]:
            log.info("early stopping after epoch %d (no improvement for %d epochs)", epoch, bad_epochs)
            break

    hist = pd.DataFrame(history)
    best_row = hist.loc[hist["val_macro_auroc"].idxmax()]
    summary = {"counts": counts, "epochs_run": len(hist), "best_epoch": int(best_row["epoch"]),
               "best_val_macro_auroc": float(best_row["val_macro_auroc"]),
               "best_epoch_train_loss": float(best_row["train_loss"]), "best_epoch_val_loss": float(best_row["val_loss"]),
               "total_seconds": time.time() - t_start, "peak_gpu_mem_gb": float(hist["peak_gpu_mem_gb"].max()),
               "stopped_early": len(hist) < tcfg["max_epochs"]}
    env["training_summary"] = summary
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False), encoding="utf-8")
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def phase1_pos_weight(view: str, policy_name: str) -> dict:
    d = json.loads((SPLITS_DIR / "chexpert" / "training_pos_weights.json").read_text(encoding="utf-8"))
    return d["weights"][view][policy_name]


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


__all__ = ["run_training", "load_config", "PROJECT_ROOT"]
