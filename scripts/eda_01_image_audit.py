"""Stage 1: read-only integrity audit of every image file in both datasets.

Outputs
    data/cache/image_audit_full.csv.gz         (every file; regenerable, git-ignored)
    results/eda/tables/image_quality_flags.csv (only records with at least one flag)
    results/eda/tables/image_audit_summary.csv
    results/eda/tables/duplicate_analysis.csv  (exact + perceptual candidate groups)

Usage:  .venv\\Scripts\\python.exe -m scripts.eda_01_image_audit
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.chexpert import load_chexpert_csv
from src.data.image_audit import audit_many, walk_files
from src.data.iu_xray import load_projections
from src.utils.audit import AuditLog, setup_logging
from src.utils.config import CACHE_DIR, EDA_TABLES, ensure_output_dirs, load_paths, load_yaml

log = logging.getLogger("image_audit")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_inventory(paths) -> pd.DataFrame:
    """Union of CSV-referenced images and image-like files found on disk."""
    audit = AuditLog("inventory")
    rows = []
    for split, csv in (("train", paths.chexpert_train_csv), ("valid", paths.chexpert_valid_csv)):
        df, _ = load_chexpert_csv(csv, split, audit)
        for p in df["Path"]:
            rows.append({"dataset": "chexpert", "rel_path": p.replace("\\", "/"), "csv_split": split,
                         "abs_path": str(paths.chexpert_image_base / p)})
    proj = load_projections(paths.iu_projections_csv, audit)
    for fn in proj["filename"]:
        rows.append({"dataset": "iu_xray", "rel_path": fn, "csv_split": "all",
                     "abs_path": str(paths.iu_images_dir / fn)})
    inv = pd.DataFrame(rows)
    inv["referenced_in_csv"] = True

    on_disk = []
    for dataset, root, base in (("chexpert", paths.chexpert_root, paths.chexpert_image_base),
                                ("iu_xray", paths.iu_images_dir, paths.iu_images_dir)):
        for f in walk_files(root):
            if f.suffix.lower() not in IMAGE_EXTS and f.suffix.lower() not in {".csv"}:
                on_disk.append({"dataset": dataset, "abs_path": str(f), "kind": "non_image_file"})
            elif f.suffix.lower() in IMAGE_EXTS:
                on_disk.append({"dataset": dataset, "abs_path": str(f),
                                "kind": "appledouble_metadata" if f.name.startswith("._") else "image"})
    disk = pd.DataFrame(on_disk)
    disk["key"] = disk["abs_path"].map(lambda s: str(Path(s).resolve()).lower())
    inv["key"] = inv["abs_path"].map(lambda s: str(Path(s).resolve()).lower())
    orphans = disk[~disk["key"].isin(set(inv["key"]))]
    orph_rows = orphans.assign(rel_path=orphans["abs_path"], csv_split="not_referenced",
                               referenced_in_csv=False)
    inv = pd.concat([inv, orph_rows[["dataset", "rel_path", "csv_split", "abs_path", "referenced_in_csv", "key", "kind"]]],
                    ignore_index=True)
    inv["kind"] = inv["kind"].fillna("image")
    return inv


def flag_records(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    img_cfg = cfg["images"]
    f = pd.DataFrame(index=df.index)
    f["missing_file"] = ~df["exists"]
    f["zero_byte"] = df["exists"] & (df["file_bytes"] == 0)
    f["unreadable_or_corrupt"] = df["exists"] & (df["file_bytes"] > 0) & ~df["readable"]
    f["not_referenced_in_csv"] = ~df["referenced_in_csv"]
    f["appledouble_metadata_file"] = df["kind"].eq("appledouble_metadata")
    ext = df["rel_path"].str.extract(r"\.([A-Za-z0-9]+)$", expand=False).str.lower()
    expected_ext = np.where(df["dataset"] == "chexpert", "jpg", "png")
    f["unusual_extension"] = ext.ne(expected_ext)
    f["non_grayscale_mode"] = df["readable"] & ~df["mode"].isin(["L", "I;16", "I"])
    f["suspiciously_small"] = df["readable"] & (np.minimum(df["width"], df["height"]) < img_cfg["suspicious_min_side_px"])
    sat = img_cfg["near_blank_saturated_fraction"]
    f["near_blank"] = df["readable"] & ((df["pixel_std"] < img_cfg["low_pixel_std"])
                                        | (df["frac_near_white"] >= sat) | (df["frac_near_black"] >= sat))
    ar = df["width"] / df["height"]
    f["unusual_aspect_ratio"] = df["readable"] & ((ar < img_cfg["aspect_ratio_min"]) | (ar > img_cfg["aspect_ratio_max"]))
    return f


def duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = df[df["readable"] & df["kind"].eq("image")]
    for method, key in (("exact_sha256", "sha256"), ("perceptual_dhash256_exact", "dhash256")):
        g = ok.groupby(key)
        sizes = g["rel_path"].transform("size")
        dups = ok[sizes > 1]
        for gid, (h, grp) in enumerate(dups.groupby(key)):
            for r in grp.itertuples():
                rows.append({"method": method, "group_id": f"{method}:{gid}", "hash": h,
                             "group_size": len(grp), "dataset": r.dataset, "rel_path": r.rel_path,
                             "csv_split": r.csv_split,
                             "n_datasets_in_group": grp["dataset"].nunique()})
    # Duplicate CSV references (same image path listed more than once).
    ref = df[df["referenced_in_csv"]]
    dup_ref = ref[ref.duplicated(["dataset", "rel_path"], keep=False)]
    for gid, (p, grp) in enumerate(dup_ref.groupby(["dataset", "rel_path"])):
        rows.append({"method": "duplicate_csv_reference", "group_id": f"ref:{gid}", "hash": None,
                     "group_size": len(grp), "dataset": p[0], "rel_path": p[1],
                     "csv_split": ",".join(grp["csv_split"]), "n_datasets_in_group": 1})
    return pd.DataFrame(rows, columns=["method", "group_id", "hash", "group_size", "dataset", "rel_path",
                                       "csv_split", "n_datasets_in_group"])


def main() -> int:
    setup_logging()
    ensure_output_dirs()
    cfg = load_yaml("eda.yaml")
    paths = load_paths()
    inv = build_inventory(paths)
    log.info("inventory: %s", inv.groupby(["dataset", "kind", "referenced_in_csv"]).size().to_dict())

    cache = CACHE_DIR / "image_audit_full.csv.gz"
    unique_paths = inv["abs_path"].drop_duplicates().tolist()
    if cache.exists() and "--force" not in sys.argv:
        log.info("re-using cached audit %s (pass --force to rescan)", cache)
        res = pd.read_csv(cache)
    else:
        res = pd.DataFrame(audit_many(unique_paths))
        res.to_csv(cache, index=False, compression="gzip")
    df = inv.merge(res, left_on="abs_path", right_on="path", how="left").drop(columns=["path", "key"])
    for c in ("exists", "readable"):
        df[c] = df[c].fillna(False).astype(bool)

    flags = flag_records(df, cfg)
    df = pd.concat([df, flags], axis=1)
    flag_cols = list(flags.columns)
    df["n_flags"] = flags.sum(axis=1)
    df["flags"] = flags.apply(lambda r: ";".join(c for c in flag_cols if r[c]), axis=1)
    out_cols = ["dataset", "rel_path", "csv_split", "referenced_in_csv", "kind", "exists", "file_bytes",
                "readable", "error", "format", "mode", "width", "height", "pixel_mean", "pixel_std", "flags"]
    df.loc[df["n_flags"] > 0, out_cols].to_csv(EDA_TABLES / "image_quality_flags.csv", index=False)

    summ = []
    for ds, g in df.groupby("dataset"):
        r = g[g["readable"]]
        ar = r["width"] / r["height"]
        s = {"dataset": ds, "files_total": len(g),
             "csv_referenced_records": int(g["referenced_in_csv"].sum()),
             "unique_referenced_paths": int(g.loc[g["referenced_in_csv"], "rel_path"].nunique()),
             "readable": int(g["readable"].sum())}
        s.update({c: int(g[c].sum()) for c in flag_cols})
        s.update({
            "width_min": r["width"].min(), "width_median": r["width"].median(), "width_max": r["width"].max(),
            "height_min": r["height"].min(), "height_median": r["height"].median(), "height_max": r["height"].max(),
            "aspect_min": ar.min(), "aspect_median": ar.median(), "aspect_max": ar.max(),
            "file_mb_total": g["file_bytes"].sum() / 1e6,
        })
        for key, label in (("sha256", "exact_sha256"), ("dhash", "dhash64"), ("dhash256", "dhash256")):
            vc = r[r["kind"].eq("image")][key].value_counts()
            s[f"dup_groups_{label}"] = int((vc > 1).sum())
            s[f"dup_images_{label}"] = int(vc[vc > 1].sum())
            s[f"dup_max_group_{label}"] = int(vc.max()) if len(vc) else 0
        for mode, c in r["mode"].value_counts().items():
            s[f"mode_{mode}"] = int(c)
        for fmt, c in r["format"].value_counts().items():
            s[f"format_{fmt}"] = int(c)
        summ.append(s)
    pd.DataFrame(summ).to_csv(EDA_TABLES / "image_audit_summary.csv", index=False)

    dups = duplicate_groups(df)
    dups.to_csv(EDA_TABLES / "duplicate_analysis.csv", index=False)
    log.info("flags: %s", {c: int(df[c].sum()) for c in flag_cols})
    log.info("duplicate rows by method: %s", dups.groupby("method").size().to_dict() if len(dups) else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
