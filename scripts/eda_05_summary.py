"""Stage 5: assemble machine-readable summaries and the reproducibility record.

Reads ONLY outputs written by stages 1-4; computes no new statistics except
the dataset directory inventory.

Outputs
    results/eda/tables/dataset_summary.csv
    results/eda/summaries/eda_summary.json
    results/eda/summaries/reproducibility.json

Usage:  .venv\\Scripts\\python.exe -m scripts.eda_05_summary
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils.audit import setup_logging
from src.utils.config import EDA_SUMMARIES, EDA_TABLES, SPLITS_DIR, load_paths, load_yaml
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("eda_summary")


def directory_inventory(root: Path, max_depth: int = 2) -> dict:
    """Directory counts and file extensions (depth-limited listing + full extension census)."""
    ext = Counter()
    n_files = n_dirs = 0
    listing: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        n_dirs += len(dirnames)
        n_files += len(filenames)
        for f in filenames:
            ext[Path(f).suffix.lower() or "<none>"] += 1
        if depth < max_depth:
            listing[str(rel).replace("\\", "/") or "."] = len(dirnames) + len(filenames)
    return {"root": str(root), "files": n_files, "directories": n_dirs,
            "extensions": dict(ext.most_common()), f"entries_per_dir_depth_lt_{max_depth}":
                dict(sorted(listing.items())[:50])}


def main() -> int:
    setup_logging()
    cfg = load_yaml("eda.yaml")
    paths = load_paths()
    cx = json.loads((EDA_SUMMARIES / "chexpert_stats.json").read_text(encoding="utf-8"))
    cs = json.loads((EDA_SUMMARIES / "chexpert_split_stats.json").read_text(encoding="utf-8"))
    iu = json.loads((EDA_SUMMARIES / "iu_stats.json").read_text(encoding="utf-8"))
    img = pd.read_csv(EDA_TABLES / "image_audit_summary.csv").set_index("dataset")
    imb = pd.read_csv(EDA_TABLES / "chexpert_imbalance_analysis.csv")
    imb = imb[imb.cohort == "official_train_all_images"].set_index("observation")
    pw = pd.read_csv(EDA_TABLES / "training_pos_weights.csv")
    iu_split = pd.read_csv(EDA_TABLES / "iu_split_sizes.csv")
    iu_meta = json.loads((SPLITS_DIR / "iu_xray" / "split_metadata.json").read_text(encoding="utf-8"))

    p = cx["patients"]
    rows = [
        ("chexpert", "images_total", p["images_total"]), ("chexpert", "images_official_train", p["images_official_train"]),
        ("chexpert", "images_official_valid", p["images_official_valid"]), ("chexpert", "patients_total", p["patients_total"]),
        ("chexpert", "studies_total", p["studies_total"]), ("chexpert", "frontal_images", cx["views"]["Frontal/Lateral:Frontal"]),
        ("chexpert", "lateral_images", cx["views"]["Frontal/Lateral:Lateral"]),
        ("chexpert", "target_observations", len(cx["targets"])),
        ("chexpert", "unreadable_image_files", int(img.loc["chexpert", "unreadable_or_corrupt"])),
        ("chexpert", "near_blank_images", int(img.loc["chexpert", "near_blank"])),
        ("chexpert", "exact_duplicate_groups", int(img.loc["chexpert", "dup_groups_exact_sha256"])),
        ("chexpert", "cross_patient_duplicate_links", cs["links"]),
        ("chexpert", "no_finding_conflicts", cx["no_finding"]["conflicts_with_positive_pathology"]),
        ("iu_xray", "reports", iu["summary"]["reports_rows"]), ("iu_xray", "images_in_projection_csv", iu["summary"]["projection_rows"]),
        ("iu_xray", "image_files_on_disk", iu["summary"]["image_files_on_disk"]),
        ("iu_xray", "frontal_images", iu["summary"]["frontal_images"]), ("iu_xray", "lateral_images", iu["summary"]["lateral_images"]),
        ("iu_xray", "reports_both_sections", iu["summary"].get("section_both", 0)),
        ("iu_xray", "reports_no_text", iu["summary"].get("section_neither", 0)),
        ("iu_xray", "retrieval_corpus_reports", iu["retrieval_corpus_size"]),
    ]
    for s in cs["sizes"]:
        rows += [("chexpert", f"split_{s['split']}_images", s["images"]), ("chexpert", f"split_{s['split']}_patients", s["patients"])]
    for r in iu_split.itertuples():
        rows += [("iu_xray", f"split_{r.split}_reports", r.reports), ("iu_xray", f"split_{r.split}_images", r.images)]
    pd.DataFrame(rows, columns=["dataset", "metric", "value"]).to_csv(EDA_TABLES / "dataset_summary.csv", index=False)

    # Consolidated CANDIDATE list for a Phase 2 decision. Nothing is excluded here.
    manifest = pd.read_csv(SPLITS_DIR / "chexpert" / "chexpert_image_manifest.csv.gz")
    split_of = manifest.set_index("Path")["split"]
    review = pd.read_csv(EDA_TABLES / "manual_image_review.csv")
    flags = pd.read_csv(EDA_TABLES / "image_quality_flags.csv")
    ages = pd.read_csv(EDA_TABLES / "chexpert_age_flags.csv")
    cand = []
    for r in flags[(flags.dataset == "chexpert") & flags.referenced_in_csv].itertuples():
        confirmed = review.loc[review.rel_path == r.rel_path, "finding"]
        cand.append({"Path": r.rel_path, "reason": r.flags, "manual_review": confirmed.iloc[0] if len(confirmed) else "",
                     "recommendation": "exclude" if "near_blank" in r.flags else "keep (no defect)"})
    for r in ages.itertuples():
        cand.append({"Path": r.Path, "reason": f"implausible_age={r.Age}", "manual_review": "",
                     "recommendation": "keep image; treat age as missing in any age-stratified analysis"})
    odd_view = manifest[manifest["AP/PA"].isin(["LL", "RL"])]
    for path, appa in zip(odd_view["Path"], odd_view["AP/PA"]):
        cand.append({"Path": path, "reason": f"frontal image with AP/PA={appa}", "manual_review": "",
                     "recommendation": "keep; exclude from any AP-vs-PA analysis"})
    cand = pd.DataFrame(cand)
    cand["split"] = cand["Path"].map(split_of)
    cand.to_csv(EDA_TABLES / "chexpert_exclusion_candidates.csv", index=False)

    primary_pw = pw[(pw.view_subset == "frontal") & (pw.label_policy == "U-Mask_NaN-Zero")].set_index("observation")
    summary = {
        "generated_from": "scripts/eda_05_summary.py (aggregates stage 1-4 outputs; no hand-entered numbers)",
        "chexpert": {
            "images": p["images_total"], "images_official_train": p["images_official_train"],
            "images_official_valid": p["images_official_valid"], "patients": p["patients_total"],
            "studies": p["studies_total"], "views": cx["views"], "targets": cx["targets"],
            "class_counts_official_train": cx["label_states_train_all"],
            "positive_prevalence_official_train": imb["positive_prevalence_all_images"].to_dict(),
            "neg_to_pos_ratio_explicit": imb["neg_to_pos_ratio_explicit"].to_dict(),
            "neg_to_pos_ratio_nan_as_neg_unc_masked": imb["neg_to_pos_ratio_nan_as_neg_unc_masked"].to_dict(),
            "imbalance_severity": imb["imbalance_severity"].to_dict(),
            "uncertain_rate_pct": imb["uncertain_pct"].to_dict(), "missing_rate_pct": imb["missing_pct"].to_dict(),
            "multilabel": cx["multilabel"], "no_finding": {k: v for k, v in cx["no_finding"].items() if k != "representative_rows"},
            "demographics": cx["demographics"],
            "split_decision": cs["decision"], "split_sizes": cs["sizes"],
            "cross_patient_duplicate_links": cs["links"], "patients_merged_for_splitting": cs["patients_merged"],
            "training_pos_weight_frontal_U-Mask_NaN-Zero": primary_pw["pos_weight"].to_dict(),
            "sampling_kish_ess_fraction": cs["sampling_ess_fraction"],
            "image_integrity": img.loc["chexpert"].to_dict(),
        },
        "iu_xray": {
            "images_on_disk": iu["summary"]["image_files_on_disk"], "images_in_projection_csv": iu["summary"]["projection_rows"],
            "reports": iu["summary"]["reports_rows"],
            "projection_counts": {"Frontal": iu["summary"]["frontal_images"], "Lateral": iu["summary"]["lateral_images"]},
            "report_completeness": {k.replace("section_", ""): v for k, v in iu["summary"].items() if k.startswith("section_")},
            "quality_flags": {k.replace("quality_", ""): v for k, v in iu["summary"].items() if k.startswith("quality_")},
            "split_sizes": iu["split_sizes"], "retrieval_corpus_reports": iu["retrieval_corpus_size"],
            "exact_text_overlap_with_train": iu["text_overlap"],
            "image_integrity": img.loc["iu_xray"].to_dict(),
        },
        "seed": cfg["seed"],
    }
    (EDA_SUMMARIES / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    repro = {
        "environment": environment_snapshot(cfg["seed"]),
        "config_files": ["configs/eda.yaml", "configs/iu_concept_lexicon.yaml", "configs/paths.yaml (local, git-ignored)"],
        "dataset_paths": paths.as_dict(),
        "dataset_structure": {
            "chexpert_root": directory_inventory(paths.chexpert_root),
            "iu_xray_root": directory_inventory(paths.iu_xray_root),
        },
        "dataset_counts": {"chexpert_images": p["images_total"], "chexpert_patients": p["patients_total"],
                           "iu_reports": iu["summary"]["reports_rows"], "iu_images_on_disk": iu["summary"]["image_files_on_disk"]},
        "split_metadata": {"chexpert": "data/splits/chexpert/split_metadata.json",
                           "iu_xray": "data/splits/iu_xray/split_metadata.json"},
        "iu_split_seed": iu_meta["seed"],
        "pipeline": ["scripts.eda_01_image_audit", "scripts.eda_02_chexpert", "scripts.eda_03_chexpert_splits",
                     "scripts.eda_04_iu_xray", "scripts.eda_05_summary"],
    }
    (EDA_SUMMARIES / "reproducibility.json").write_text(json.dumps(repro, indent=2, default=str), encoding="utf-8")
    log.info("wrote dataset_summary.csv, eda_summary.json, reproducibility.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
