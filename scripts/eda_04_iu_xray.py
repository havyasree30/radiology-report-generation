"""Stage 4: IU X-Ray structural EDA, text quality, concept analysis and report-level splits.

Usage:  .venv\\Scripts\\python.exe -m scripts.eda_04_iu_xray
"""

from __future__ import annotations

import json
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plotting import (INK_2, MISSING_GREY, SERIES, STATE_COLORS, apply_style, save_figure, subtitle)
from src.analysis.text_concepts import NEGATED, NOT_MENTIONED, POSITIVE, UNCERTAIN, ConceptExtractor
from src.data.image_audit import walk_files
from src.data.iu_xray import (ANONYMIZATION_TOKEN, approx_token_count, combined_text, load_projections,
                              load_reports, normalize_for_duplicates, section_status, word_count)
from src.data.retrieval_corpus import assert_train_only, eligible_report_ids
from src.data.splits import assert_no_group_overlap, stratified_simple_group_split
from src.utils.audit import AuditLog, setup_logging
from src.utils.config import EDA_SUMMARIES, EDA_TABLES, SPLITS_DIR, ensure_output_dirs, load_paths, load_yaml
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("eda_iu")
OUT = SPLITS_DIR / "iu_xray"
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def length_stats(s: pd.Series) -> dict:
    return {"n": int(s.size), "mean": float(s.mean()), "std": float(s.std()), "min": int(s.min()),
            "q1": float(s.quantile(0.25)), "median": float(s.median()), "q3": float(s.quantile(0.75)),
            "p99": float(s.quantile(0.99)), "max": int(s.max())}


def main() -> int:
    setup_logging()
    ensure_output_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    cfg = load_yaml("eda.yaml")
    icfg = cfg["iu_xray"]
    seed = int(cfg["seed"])
    paths = load_paths()
    audit = AuditLog("iu_xray")
    rep = load_reports(paths.iu_reports_csv, audit)
    proj = load_projections(paths.iu_projections_csv, audit)

    # ---------------- Part 18: structure ----------------
    disk = {p.name for p in walk_files(paths.iu_images_dir)}
    proj["file_exists"] = proj["filename"].isin(disk)
    for fn in proj.loc[~proj["file_exists"], "filename"]:
        audit.add("missing_image_file", fn, "", severity="error")
    unref = sorted(disk - set(proj["filename"]))
    for fn in unref:
        audit.add("image_not_in_projections_csv", fn, "")
    rep_uids, proj_uids = set(rep["uid"].dropna()), set(proj["uid"].dropna())
    for u in sorted(rep_uids - proj_uids, key=int):
        audit.add("report_without_images", u, "")
    for u in sorted(proj_uids - rep_uids, key=int):
        audit.add("images_without_report", u, "", severity="error")

    per_uid = proj.groupby("uid").agg(n_images=("filename", "size"),
                                      n_frontal=("projection", lambda s: int((s == "Frontal").sum())),
                                      n_lateral=("projection", lambda s: int((s == "Lateral").sum())))
    per_uid["projection_pattern"] = per_uid.apply(
        lambda r: f"{r.n_frontal}F+{r.n_lateral}L", axis=1)
    rep = rep.merge(per_uid, left_on="uid", right_index=True, how="left")
    rep[["n_images", "n_frontal", "n_lateral"]] = rep[["n_images", "n_frontal", "n_lateral"]].fillna(0).astype(int)
    rep["projection_pattern"] = rep["projection_pattern"].fillna("no_images")

    proj_summary = []
    for k, v in proj["projection"].value_counts().items():
        proj_summary.append({"level": "image", "category": k, "count": int(v), "pct": 100 * v / len(proj)})
    for k, v in rep["projection_pattern"].value_counts().items():
        proj_summary.append({"level": "report_projection_pattern", "category": k, "count": int(v), "pct": 100 * v / len(rep)})
    for k, v in rep["n_images"].value_counts().sort_index().items():
        proj_summary.append({"level": "images_per_report", "category": str(k), "count": int(v), "pct": 100 * v / len(rep)})
    proj_summary = pd.DataFrame(proj_summary)
    proj_summary.to_csv(EDA_TABLES / "iu_projection_summary.csv", index=False)

    # ---------------- sections & lengths ----------------
    rep["section_status"] = [section_status(f, i) for f, i in zip(rep["findings"], rep["impression"])]
    rep["combined"] = [combined_text(f, i) for f, i in zip(rep["findings"], rep["impression"])]
    rep["has_any_text"] = rep["combined"].str.len() > 0
    for sec in ("findings", "impression", "combined"):
        rep[f"{sec}_chars"] = rep[sec].str.len()
        rep[f"{sec}_words"] = rep[sec].map(word_count)
        rep[f"{sec}_tokens_approx"] = rep[sec].map(approx_token_count)
    len_rows = []
    for sec in ("findings", "impression", "combined"):
        present = rep[rep[f"{sec}_chars"] > 0]
        for unit in ("chars", "words", "tokens_approx"):
            len_rows.append({"section": sec, "unit": unit, **length_stats(present[f"{sec}_{unit}"])})
    lengths = pd.DataFrame(len_rows)
    lengths.to_csv(EDA_TABLES / "iu_report_lengths.csv", index=False)

    # ---------------- Part 19: quality flags ----------------
    words = rep["combined_words"]
    long_thr = float(words[words > 0].quantile(icfg["extremely_long_quantile"]))
    norm = rep["combined"].map(normalize_for_duplicates)
    dup_counts = norm[rep["has_any_text"]].value_counts()
    rep["normalized_text"] = norm
    rep["duplicate_group_size"] = norm.map(dup_counts).fillna(0).astype(int)
    xx = rep["combined"].str.count(ANONYMIZATION_TOKEN)
    rep["anonymization_tokens"] = xx
    rep["anonymization_fraction"] = np.where(words > 0, xx / words.replace(0, np.nan), 0.0)
    q = pd.DataFrame({"uid": rep["uid"]})
    q["empty_report"] = ~rep["has_any_text"]
    q["near_empty"] = rep["has_any_text"] & (words <= icfg["near_empty_words"])
    q["extremely_short"] = rep["has_any_text"] & (words <= icfg["extremely_short_words"])
    q["extremely_long"] = words > long_thr
    q["missing_findings"] = rep["findings_chars"] == 0
    q["missing_impression"] = rep["impression_chars"] == 0
    q["duplicate_report_text"] = rep["duplicate_group_size"] > 1
    q["findings_equals_impression"] = (rep["findings_chars"] > 0) & (
        rep["findings"].map(normalize_for_duplicates) == rep["impression"].map(normalize_for_duplicates))
    q["high_anonymization"] = rep["anonymization_fraction"] >= icfg["high_anonymization_fraction"]
    q["malformed"] = rep["has_any_text"] & (~rep["combined"].str.contains("[A-Za-z]{2,}", regex=True)
                                            | rep["combined"].map(lambda s: bool(_CTRL.search(s))))
    q["no_images"] = rep["n_images"] == 0
    flag_cols = [c for c in q.columns if c != "uid"]
    q["n_flags"] = q[flag_cols].sum(axis=1)
    q = q.merge(rep[["uid", "section_status", "combined_words", "anonymization_tokens", "duplicate_group_size",
                     "n_images"]], on="uid")
    q.to_csv(EDA_TABLES / "iu_report_quality.csv", index=False)
    quality_counts = {c: int(q[c].sum()) for c in flag_cols}

    # ---------------- Part 20: concept analysis ----------------
    lex = load_yaml("iu_concept_lexicon.yaml")
    ext = ConceptExtractor.from_config(lex)
    statuses = pd.DataFrame([ext.report_status(t) for t in rep["combined"]], index=rep.index)
    n_text = int(rep["has_any_text"].sum())
    conc_rows = []
    for c in statuses.columns:
        vc = statuses.loc[rep["has_any_text"], c].value_counts()
        row = {"concept": c, "reports_with_text": n_text}
        for st in (POSITIVE, UNCERTAIN, NEGATED, NOT_MENTIONED):
            row[f"{st}_count"] = int(vc.get(st, 0))
            row[f"{st}_pct"] = 100 * vc.get(st, 0) / n_text
        conc_rows.append(row)
    conc = pd.DataFrame(conc_rows)
    conc["method"] = "rule-based assertion-aware keyword matching (approximate; not clinical NLP)"
    conc.to_csv(EDA_TABLES / "iu_concept_frequency.csv", index=False)

    mesh_terms = rep["MeSH"].fillna("").map(lambda m: {x.split("/")[0].strip() for x in m.split(";") if x.strip()})
    agree = []
    for c, spec in lex["concepts"].items():
        if not spec.get("mesh"):
            continue
        ref = mesh_terms.map(lambda s: bool(s & set(spec["mesh"])))[rep["has_any_text"]]
        pred_pos = (statuses.loc[rep["has_any_text"], c] == POSITIVE)
        pred_pu = statuses.loc[rep["has_any_text"], c].isin([POSITIVE, UNCERTAIN])
        for name, pred in (("positive", pred_pos), ("positive_or_uncertain", pred_pu)):
            tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
            agree.append({"concept": c, "extractor_definition": name, "mesh_headings": ";".join(spec["mesh"]),
                          "mesh_positive_reports": int(ref.sum()), "extractor_positive_reports": int(pred.sum()),
                          "both": tp, "extractor_only": fp, "mesh_only": fn,
                          "agreement_precision_vs_mesh": tp / (tp + fp) if tp + fp else np.nan,
                          "agreement_recall_vs_mesh": tp / (tp + fn) if tp + fn else np.nan})
    pd.DataFrame(agree).to_csv(EDA_TABLES / "iu_concept_mesh_agreement.csv", index=False)
    rng = np.random.default_rng(seed)
    all_mentions = [dict(m, uid=u) for u, t in zip(rep["uid"], rep["combined"]) for m in ext.mentions(t)]
    mdf = pd.DataFrame(all_mentions)
    sample = mdf.iloc[rng.choice(len(mdf), size=min(300, len(mdf)), replace=False)].sort_values(["concept", "status"])
    sample.to_csv(EDA_TABLES / "iu_concept_mentions_review_sample.csv", index=False)
    mdf.groupby(["concept", "status"]).size().rename("mentions").reset_index().to_csv(
        EDA_TABLES / "iu_concept_mention_counts.csv", index=False)

    # ---------------- Part 21: report-level splits ----------------
    rep["stratum"] = np.where(rep["MeSH"].fillna("").str.strip().str.lower() == "normal", "normal", "not_normal")
    assign = stratified_simple_group_split(rep["uid"], rep["stratum"], icfg["split_ratios"], seed)
    rep["split"] = rep["uid"].map(assign)
    rep_split = rep[["uid", "split", "stratum", "n_images", "n_frontal", "n_lateral", "section_status",
                     "has_any_text"]].copy()
    rep_split["retrieval_corpus_eligible"] = rep_split["uid"].isin(eligible_report_ids(rep_split))
    img_split = proj.merge(rep_split[["uid", "split"]], on="uid", how="left")
    img_split["split"] = img_split["split"].fillna("unassigned_no_report")

    assert_no_group_overlap(rep_split, "uid")
    assert_no_group_overlap(img_split[img_split["split"] != "unassigned_no_report"], "uid")
    assert_no_group_overlap(img_split.assign(fn=img_split["filename"]), "fn")
    corpus = eligible_report_ids(rep_split)
    assert_train_only(corpus, rep_split)

    rep_split.to_csv(OUT / "iu_report_splits.csv", index=False)
    img_split[["filename", "uid", "projection", "split"]].to_csv(OUT / "iu_image_manifest.csv", index=False)
    pd.Series(corpus, name="uid").to_csv(OUT / "retrieval_corpus_uids.csv", index=False)

    # Exact-text overlap across splits: not identity leakage, but it inflates text-similarity metrics.
    norm_by_split = {s: set(rep.loc[(rep["split"] == s) & rep["has_any_text"], "normalized_text"]) for s in ("train", "val", "test")}
    overlap = {}
    for s in ("val", "test"):
        sub = rep[(rep["split"] == s) & rep["has_any_text"]]
        hit = sub["normalized_text"].isin(norm_by_split["train"])
        overlap[s] = {"reports": int(len(sub)), "exact_text_in_train": int(hit.sum()), "pct": 100 * hit.mean()}
    split_sizes = rep.groupby("split").agg(reports=("uid", "size"), images=("n_images", "sum"),
                                           normal_reports=("stratum", lambda s: int((s == "normal").sum())),
                                           with_any_text=("has_any_text", "sum")).reset_index()
    split_sizes["normal_pct"] = 100 * split_sizes["normal_reports"] / split_sizes["reports"]
    split_sizes.to_csv(EDA_TABLES / "iu_split_sizes.csv", index=False)
    (OUT / "split_metadata.json").write_text(json.dumps({
        "unit": "report uid (all images of a report stay together)", "strata": "MeSH == 'normal' vs other",
        "seed": seed, "ratios": icfg["split_ratios"], "sizes": split_sizes.to_dict(orient="records"),
        "retrieval_corpus": {"rule": "split == train and non-empty findings/impression", "n_reports": len(corpus)},
        "exact_text_overlap_with_train": overlap,
        "limitation": "No patient identifier in the IU CSVs; patient-level grouping is impossible.",
        "environment": environment_snapshot(seed)}, indent=2, default=float), encoding="utf-8")

    # ---------------- summary table ----------------
    summ = {
        "reports_rows": len(rep), "unique_report_uids": int(rep["uid"].nunique()),
        "projection_rows": len(proj), "unique_image_filenames": int(proj["filename"].nunique()),
        "image_files_on_disk": len(disk), "image_files_not_in_projections_csv": len(unref),
        "projection_rows_missing_file": int((~proj["file_exists"]).sum()),
        "reports_without_images": len(rep_uids - proj_uids), "image_uids_without_report": len(proj_uids - rep_uids),
        "frontal_images": int((proj["projection"] == "Frontal").sum()),
        "lateral_images": int((proj["projection"] == "Lateral").sum()),
        "other_projection_images": int((~proj["projection"].isin(["Frontal", "Lateral"])).sum()),
        **{f"section_{k}": int(v) for k, v in rep["section_status"].value_counts().items()},
        "reports_mesh_normal": int((rep["stratum"] == "normal").sum()),
        "anonymization_tokens_total": int(xx.sum()), "reports_with_anonymization_token": int((xx > 0).sum()),
        "extremely_long_threshold_words": long_thr,
        **{f"quality_{k}": v for k, v in quality_counts.items()},
        "distinct_normalized_texts": int(dup_counts.size),
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in summ.items()]).to_csv(
        EDA_TABLES / "iu_dataset_summary.csv", index=False)
    audit.save(EDA_TABLES / "iu_audit_log.csv")
    (EDA_SUMMARIES / "iu_stats.json").write_text(json.dumps({
        "summary": summ, "lengths": lengths.to_dict(orient="records"),
        "split_sizes": split_sizes.to_dict(orient="records"), "text_overlap": overlap,
        "retrieval_corpus_size": len(corpus), "concepts": conc.set_index("concept").drop(columns="method").to_dict(orient="index"),
        "audit": audit.summary().to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")

    # ---------------- figures 18-24 ----------------
    order = ["both", "findings_only", "impression_only", "neither"]
    vc = rep["section_status"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(["Findings + Impression", "Findings only", "Impression only", "Neither"][::-1], vc.values[::-1],
            color=SERIES[0], height=0.55)
    for yv, c in enumerate(vc.values[::-1]):
        ax.text(c, yv, f"  {c:,} ({100 * c / len(rep):.1f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, vc.max() * 1.3)
    ax.set_xlabel("Reports")
    ax.set_title("Report-section completeness", pad=22)
    subtitle(ax, f"IU X-Ray, {len(rep):,} reports")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "iu_18_section_completeness")

    for sec, fname, colr in (("findings", "iu_19_findings_length", SERIES[0]), ("impression", "iu_20_impression_length", SERIES[0])):
        w = rep.loc[rep[f"{sec}_words"] > 0, f"{sec}_words"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(w, bins=np.arange(0, w.max() + 3, 2), color=colr)
        ax.axvline(w.median(), color=INK_2, lw=1)
        ax.text(w.median(), ax.get_ylim()[1] * 0.95, f"  median {w.median():.0f}", color=INK_2, fontsize=9, va="top")
        ax.set_xlabel("Words")
        ax.set_ylabel("Reports")
        ax.set_title(f"{sec.capitalize()} length", pad=22)
        subtitle(ax, f"Reports with a non-empty {sec} section (n={w.size:,}); whitespace-delimited words")
        ax.grid(axis="x", visible=False)
        save_figure(fig, fname)

    pv = proj["projection"].value_counts()
    pat = rep["projection_pattern"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(pv.index[::-1], pv.values[::-1], color=SERIES[0], height=0.5)
    for yv, c in enumerate(pv.values[::-1]):
        axes[0].text(c, yv, f"  {c:,}", va="center", fontsize=9, color=INK_2)
    axes[0].set_xlim(0, pv.max() * 1.25)
    axes[0].set_title("Images by projection", pad=22)
    subtitle(axes[0], f"{len(proj):,} projection rows")
    axes[1].barh(pat.index[::-1], pat.values[::-1], color=SERIES[0], height=0.6)
    for yv, c in enumerate(pat.values[::-1]):
        axes[1].text(c, yv, f"  {c:,}", va="center", fontsize=9, color=INK_2)
    axes[1].set_xlim(0, pat.max() * 1.25)
    axes[1].set_title("Projection pattern per report", pad=22)
    subtitle(axes[1], "F = frontal, L = lateral images attached to the report")
    for a in axes:
        a.grid(axis="y", visible=False)
        a.set_xlabel("Count")
    save_figure(fig, "iu_21_projection_distribution")

    ipr = rep["n_images"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(ipr.index.astype(str), ipr.values, color=SERIES[0], width=0.6)
    for xv, c in enumerate(ipr.values):
        ax.text(xv, c, f"{c:,}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xlabel("Images per report")
    ax.set_ylabel("Reports")
    ax.set_title("Images per report", pad=22)
    subtitle(ax, "All images of a report are kept in the same split")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "iu_22_images_per_report")

    co = conc.sort_values("positive_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    left = np.zeros(len(co))
    for st, lab in ((POSITIVE, "Positive"), (UNCERTAIN, "Uncertain"), (NEGATED, "Negated")):
        vals = co[f"{st}_pct"].to_numpy()
        colr = {POSITIVE: STATE_COLORS["positive"], UNCERTAIN: STATE_COLORS["uncertain"], NEGATED: STATE_COLORS["negative"]}[st]
        ax.barh(co["concept"], vals, left=left, color=colr, label=lab, height=0.65, edgecolor="white", linewidth=1)
        left += vals
    ax.set_xlabel("Reports (%); remainder = concept not mentioned")
    ax.set_title("CheXpert-vocabulary concepts in IU reports", pad=22)
    subtitle(ax, f"Rule-based assertion-aware matching, report level (n={n_text:,} reports with text); approximate")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    ax.grid(axis="y", visible=False)
    save_figure(fig, "iu_23_concept_frequency")

    qc = pd.Series(quality_counts).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([c.replace("_", " ") for c in qc.index], qc.values, color=[SERIES[0] if v > 0 else MISSING_GREY for v in qc.values],
            height=0.6)
    for yv, c in enumerate(qc.values):
        ax.text(c, yv, f"  {c:,}", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(qc.max(), 1) * 1.2)
    ax.set_xlabel("Reports flagged (flags overlap; nothing removed)")
    ax.set_title("IU report quality flags", pad=22)
    subtitle(ax, f"{len(rep):,} reports; thresholds in configs/eda.yaml")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "iu_24_report_quality")

    log.info("audit:\n%s", audit.summary())
    log.info("summary: %s", summ)
    log.info("splits:\n%s\noverlap %s", split_sizes, overlap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
