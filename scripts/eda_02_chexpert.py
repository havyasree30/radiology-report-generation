"""Stage 2: CheXpert label-state, imbalance, multi-label, cohort and demographic EDA.

Operates on the ORIGINAL label values (1 / 0 / -1 / NaN). Nothing is converted.
Primary cohort: the official training CSV (all images). The official
validation CSV is summarised separately (different label source).

Usage:  .venv\\Scripts\\python.exe -m scripts.eda_02_chexpert
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from src.analysis import multilabel as ml
from src.analysis.plotting import (INK_2, MISSING_GREY, MUTED, SEQUENTIAL, DIVERGING, SERIES, STATE_COLORS,
                                   apply_style, heatmap, plain_log_ticks, save_figure, subtitle)
from src.data.chexpert import LABEL_STATES, label_state_frame, load_chexpert_csv, positive_matrix
from src.utils.audit import AuditLog, setup_logging
from src.utils.config import EDA_SUMMARIES, EDA_TABLES, ensure_output_dirs, load_paths, load_yaml

log = logging.getLogger("eda_chexpert")
NO_FINDING, SUPPORT = "No Finding", "Support Devices"


def band(value: float, bands: list[dict], key: str) -> str:
    for b in bands:
        if value < b[key]:
            return b["name"]
    return bands[-1]["name"]


def imbalance_table(df: pd.DataFrame, targets: list[str], cfg: dict, cohort: str) -> pd.DataFrame:
    st = label_state_frame(df, targets)
    c = cfg["chexpert"]
    out = pd.DataFrame({"cohort": cohort, "observation": st["observation"], "n_images": st["n_images"],
                        "positive_count": st["positive_count"], "negative_count_explicit": st["negative_count"],
                        "uncertain_count": st["uncertain_count"], "missing_count": st["missing_count"]})
    pos = out["positive_count"].replace(0, np.nan)
    out["positive_prevalence_all_images"] = out["positive_count"] / out["n_images"]
    out["positive_prevalence_explicit"] = out["positive_count"] / (out["positive_count"] + out["negative_count_explicit"])
    out["neg_to_pos_ratio_explicit"] = out["negative_count_explicit"] / pos
    out["pos_to_neg_ratio_explicit"] = out["positive_count"] / out["negative_count_explicit"].replace(0, np.nan)
    # Ratio the loss would actually face if unmentioned labels are treated as
    # negative and uncertain labels are excluded (U-Masked, NaN->0).
    out["neg_to_pos_ratio_nan_as_neg_unc_masked"] = (out["negative_count_explicit"] + out["missing_count"]) / pos
    # ... and under U-Zero with NaN->0 (every non-positive is a negative).
    out["neg_to_pos_ratio_all_nonpositive"] = (out["n_images"] - out["positive_count"]) / pos
    out["prevalence_band"] = [band(v, c["prevalence_bands"], "max") for v in out["positive_prevalence_all_images"]]
    out["imbalance_severity"] = [band(v, c["severity_bands"], "max_ratio") if np.isfinite(v) else "no_positives"
                                 for v in out["neg_to_pos_ratio_nan_as_neg_unc_masked"].fillna(np.inf)]
    out["uncertain_pct"] = 100 * out["uncertain_count"] / out["n_images"]
    out["missing_pct"] = 100 * out["missing_count"] / out["n_images"]
    out["uncertain_share_of_pos_or_unc"] = out["uncertain_count"] / (out["positive_count"] + out["uncertain_count"])
    out["flag_negative_dominated"] = out["neg_to_pos_ratio_explicit"] >= c["negative_dominated_ratio"]
    out["flag_high_uncertainty"] = (out["uncertain_pct"] >= c["high_uncertainty_pct"]) | (
        out["uncertain_share_of_pos_or_unc"] >= c["high_uncertainty_share_of_mentions"])
    out["flag_missing_dominated"] = out["missing_pct"] >= c["missing_dominated_pct"]
    out = out.sort_values("neg_to_pos_ratio_nan_as_neg_unc_masked", kind="stable").reset_index(drop=True)
    out.insert(2, "rank_least_to_most_imbalanced", np.arange(1, len(out) + 1))
    return out


def study_label_consistency(df: pd.DataFrame, targets: list[str]) -> dict:
    multi = df[df.groupby("study_id")["Path"].transform("size") > 1]
    lab = multi[targets].fillna(-9)
    nun = lab.groupby(multi["study_id"]).nunique()
    inconsistent = (nun > 1).any(axis=1)
    return {"studies_with_multiple_images": int(len(nun)),
            "studies_with_identical_labels_across_images": int((~inconsistent).sum()),
            "studies_with_differing_labels": int(inconsistent.sum())}


def main() -> int:
    setup_logging()
    ensure_output_dirs()
    apply_style()
    cfg = load_yaml("eda.yaml")
    paths = load_paths()
    audit = AuditLog("chexpert")
    train, targets = load_chexpert_csv(paths.chexpert_train_csv, "official_train", audit)
    valid, v_targets = load_chexpert_csv(paths.chexpert_valid_csv, "official_valid", audit)
    if v_targets != targets:
        audit.add("target_mismatch_train_valid", None, f"{targets} vs {v_targets}", severity="error")
    full = pd.concat([train, valid], ignore_index=True)
    stats: dict = {"targets": targets}

    # ---------------- Part 3: label states ----------------
    frontal = train[train["Frontal/Lateral"] == "Frontal"]
    cohorts = {"official_train_all_images": train, "official_train_frontal_images": frontal,
               "official_valid_all_images": valid}
    dist = pd.concat([label_state_frame(d, targets).assign(cohort=k) for k, d in cohorts.items()], ignore_index=True)
    dist = dist[["cohort", "observation", "n_images"] + [f"{s}_{m}" for s in LABEL_STATES for m in ("count", "pct")]
                + ["other_count"]]
    dist.to_csv(EDA_TABLES / "chexpert_label_distribution.csv", index=False)
    if (dist["other_count"] != 0).any():
        audit.add("unexpected_label_state", None, "labels outside {1,0,-1,NaN}", severity="error")

    d0 = dist[dist.cohort == "official_train_all_images"].set_index("observation").loc[targets[::-1]]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    left = np.zeros(len(d0))
    for s in LABEL_STATES:
        vals = d0[f"{s}_pct"].to_numpy()
        ax.barh(d0.index, vals, left=left, color=STATE_COLORS[s], label=s.capitalize() if s != "missing" else
                "Missing / unmentioned", height=0.7, edgecolor="white", linewidth=1.0)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of images (%)")
    ax.set_title("CheXpert label states per observation", pad=22)
    subtitle(ax, f"Official training set, all {len(train):,} images; raw labels (1, 0, -1, blank) not converted")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_01_label_state_distribution")

    # ---------------- Part 4: imbalance ----------------
    imb = pd.concat([imbalance_table(train, targets, cfg, "official_train_all_images"),
                     imbalance_table(frontal, targets, cfg, "official_train_frontal_images")], ignore_index=True)
    imb.to_csv(EDA_TABLES / "chexpert_imbalance_analysis.csv", index=False)
    ia = imb[imb.cohort == "official_train_all_images"].set_index("observation")

    order = ia.sort_values("positive_prevalence_all_images").index
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(order, 100 * ia.loc[order, "positive_prevalence_all_images"], color=SERIES[0], height=0.6)
    for y, v in enumerate(100 * ia.loc[order, "positive_prevalence_all_images"]):
        ax.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Images with explicit positive label (%)")
    ax.set_title("Positive prevalence by observation", pad=22)
    subtitle(ax, "Official training set, all images; denominator = all images")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_02_positive_prevalence")

    order = ia.sort_values("neg_to_pos_ratio_nan_as_neg_unc_masked").index
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y = np.arange(len(order))
    explicit = ia.loc[order, "neg_to_pos_ratio_explicit"]
    effective = ia.loc[order, "neg_to_pos_ratio_nan_as_neg_unc_masked"]
    for i, o in enumerate(order):
        if explicit[o] > 0:
            ax.plot([explicit[o], effective[o]], [i, i], color=MISSING_GREY, lw=2, zorder=2)
    pos_mask = (explicit > 0).to_numpy()
    ax.scatter(explicit[pos_mask], y[pos_mask], s=42, color=SERIES[0], label="Explicit negatives : positives",
               zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter(effective, y, s=42, color=SERIES[1], label="(Explicit negatives + unmentioned) : positives",
               zorder=3, edgecolor="white", linewidth=1.5)
    for i, o in enumerate(order):
        if explicit[o] == 0:  # cannot be drawn on a log axis; state it instead
            ax.text(effective[o] * 0.85, i, "0 explicit negatives  ", ha="right", va="center", fontsize=8, color=INK_2)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_yticks(y, order)
    ax.set_xlabel("Negative-to-positive ratio (log scale)")
    ax.set_title("Negative-to-positive imbalance ratio", pad=22)
    subtitle(ax, "Uncertain labels excluded from both ratios; official training set, all images")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_03_neg_to_pos_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    order = ia.sort_values("uncertain_pct").index
    axes[0].barh(order, ia.loc[order, "uncertain_pct"], color=SERIES[1], height=0.6)
    axes[0].set_xlabel("Uncertain (-1) labels, % of all images")
    axes[0].set_title("Uncertain-label prevalence", pad=22)
    subtitle(axes[0], "Official training set, all images")
    axes[1].barh(order, 100 * ia.loc[order, "uncertain_share_of_pos_or_unc"], color=SERIES[1], height=0.6)
    axes[1].set_xlabel("Uncertain / (positive + uncertain), %")
    axes[1].set_title("Uncertain share of positive-leaning mentions", pad=22)
    subtitle(axes[1], "How much U-One would add to the positive class")
    for a in axes:
        a.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_04_uncertain_prevalence")

    order = ia.sort_values("missing_pct").index
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(order, ia.loc[order, "missing_pct"], color=MUTED, height=0.6)
    ax.axvline(cfg["chexpert"]["missing_dominated_pct"], color=INK_2, lw=1, ls="-")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Unmentioned (blank) labels, % of all images")
    ax.set_title("Missing / unmentioned label prevalence", pad=22)
    subtitle(ax, f"Official training set, all images; line = {cfg['chexpert']['missing_dominated_pct']:.0f}% "
                 "'missing-dominated' threshold")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_05_missing_prevalence")

    # Uncertainty and missing detail tables (Parts 16, 17).
    unc = ia[["n_images", "positive_count", "negative_count_explicit", "uncertain_count", "missing_count",
              "uncertain_pct", "uncertain_share_of_pos_or_unc"]].copy()
    mentioned = unc["positive_count"] + unc["negative_count_explicit"] + unc["uncertain_count"]
    unc["uncertain_share_of_mentions"] = unc["uncertain_count"] / mentioned
    unc["positives_if_u_one"] = unc["positive_count"] + unc["uncertain_count"]
    unc["positive_increase_factor_u_one"] = unc["positives_if_u_one"] / unc["positive_count"].replace(0, np.nan)
    unc["valid_uncertain_count"] = [int((valid[o] == -1).sum()) for o in unc.index]
    unc.reset_index().to_csv(EDA_TABLES / "chexpert_uncertainty_analysis.csv", index=False)

    mis = ia[["n_images", "positive_count", "negative_count_explicit", "uncertain_count", "missing_count",
              "missing_pct"]].copy()
    mis["explicit_mention_pct"] = 100 - mis["missing_pct"]
    mis["missing_to_uncertain_ratio"] = mis["missing_count"] / mis["uncertain_count"].replace(0, np.nan)
    mis["share_of_nan_among_implicit_negatives"] = mis["missing_count"] / (mis["missing_count"] + mis["negative_count_explicit"])
    mis["frontal_missing_pct"] = imb[imb.cohort == "official_train_frontal_images"].set_index("observation").loc[mis.index, "missing_pct"]
    mis["valid_missing_count"] = [int(valid[o].isna().sum()) for o in mis.index]
    mis.reset_index().to_csv(EDA_TABLES / "chexpert_missing_analysis.csv", index=False)

    # ---------------- Part 6: multi-label structure (frontal training images) ----------------
    Yf = positive_matrix(frontal, targets)
    path_idx = [i for i, t in enumerate(targets) if t != NO_FINDING]
    pathology_idx = [i for i, t in enumerate(targets) if t not in (NO_FINDING, SUPPORT)]
    card = {
        "cohort": "official_train_frontal_images", "n_images": int(len(Yf)),
        "label_cardinality_14": ml.cardinality(Yf), "label_density_14": ml.density(Yf),
        "label_cardinality_13_excl_no_finding": ml.cardinality(Yf[:, path_idx]),
        "label_density_13_excl_no_finding": ml.density(Yf[:, path_idx]),
        "label_cardinality_12_pathologies": ml.cardinality(Yf[:, pathology_idx]),
    }
    Ya = positive_matrix(train, targets)
    card_all = {"label_cardinality_14_all_images": ml.cardinality(Ya), "label_density_14_all_images": ml.density(Ya)}
    dist13 = ml.count_distribution(Yf[:, path_idx])
    dist14 = ml.count_distribution(Yf)
    kdf = pd.DataFrame({"n_positive_labels": sorted(set(dist13.index) | set(dist14.index))})
    kdf["images_14_labels"] = kdf["n_positive_labels"].map(dist14).fillna(0).astype(int)
    kdf["images_13_labels_excl_no_finding"] = kdf["n_positive_labels"].map(dist13).fillna(0).astype(int)
    kdf.to_csv(EDA_TABLES / "chexpert_label_cardinality.csv", index=False)
    stats["multilabel"] = {**card, **card_all}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, title in ((axes[0], "images_14_labels", "All 14 observations"),
                           (axes[1], "images_13_labels_excl_no_finding", "13 observations (excluding No Finding)")):
        ax.bar(kdf["n_positive_labels"], kdf[col], color=SERIES[0], width=0.7)
        ax.set_xlabel("Explicitly positive observations per image")
        ax.set_title(title, pad=22)
        subtitle(ax, f"Frontal training images (n={len(Yf):,})")
        ax.set_xticks(kdf["n_positive_labels"])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Images")
    save_figure(fig, "chexpert_06_positive_cardinality")

    co = ml.cooccurrence_counts(Yf)
    pairs = ml.pairwise_long(Yf, targets)
    pairs.insert(0, "cohort", "official_train_frontal_images")
    pairs.to_csv(EDA_TABLES / "chexpert_cooccurrence.csv", index=False)
    for name, M in (("counts", co), ("jaccard", ml.jaccard(Yf)), ("conditional_p_col_given_row", ml.conditional_probability(Yf)),
                    ("phi", ml.phi_matrix(Yf))):
        pd.DataFrame(M, index=targets, columns=targets).to_csv(EDA_TABLES / f"chexpert_cooccurrence_matrix_{name}.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    Mco = co.astype(float)
    im = ax.imshow(np.where(Mco > 0, Mco, np.nan), cmap=SEQUENTIAL, norm=LogNorm(vmin=max(1, Mco[Mco > 0].min()), vmax=Mco.max()))
    ax.set_xticks(range(len(targets)), targets, rotation=55, ha="right")
    ax.set_yticks(range(len(targets)), targets)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(targets)):
        for j in range(len(targets)):
            v = Mco[i, j]
            ax.text(j, i, f"{v/1000:.1f}k" if v >= 1000 else f"{int(v)}", ha="center", va="center", fontsize=6,
                    color="white" if v > 0.1 * Mco.max() else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Images with both labels positive (log scale)", fontsize=9, color=INK_2)
    ax.set_title("Positive-positive co-occurrence", pad=22)
    subtitle(ax, "Frontal training images; diagonal = positives per observation")
    save_figure(fig, "chexpert_07_cooccurrence_counts")

    J = ml.jaccard(Yf)
    np.fill_diagonal(J, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    heatmap(ax, J, targets, SEQUENTIAL, vmin=0, vmax=np.nanmax(J), cbar_label="Jaccard index")
    ax.set_title("Jaccard similarity of positive labels", pad=22)
    subtitle(ax, "|A and B| / |A or B|; frontal training images; diagonal omitted")
    save_figure(fig, "chexpert_08_jaccard")

    C = ml.conditional_probability(Yf)
    np.fill_diagonal(C, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    heatmap(ax, C, targets, SEQUENTIAL, vmin=0, vmax=1, cbar_label="P(column positive | row positive)")
    ax.set_title("Conditional co-occurrence P(B | A)", pad=22)
    subtitle(ax, "Row = conditioning label A, column = B; frontal training images")
    ax.set_ylabel("A (given positive)")
    ax.set_xlabel("B")
    save_figure(fig, "chexpert_09_conditional_cooccurrence")

    P = ml.phi_matrix(Yf)
    np.fill_diagonal(P, np.nan)
    lim = np.nanmax(np.abs(P))
    fig, ax = plt.subplots(figsize=(9, 8))
    heatmap(ax, P, targets, DIVERGING, vmin=-lim, vmax=lim, cbar_label="Phi coefficient")
    ax.set_title("Phi correlation between positive indicators", pad=22)
    subtitle(ax, "Positive (1) vs not positive (0, -1, blank); association, not causation")
    save_figure(fig, "chexpert_09b_phi_correlation")

    # ---------------- Part 7: No Finding consistency ----------------
    patho = [t for t in targets if t not in (NO_FINDING, SUPPORT)]
    nf = full[full[NO_FINDING] == 1.0]
    pos_patho = nf[patho] == 1.0
    unc_patho = nf[patho] == -1.0
    conflict = pos_patho.any(axis=1)
    conf_rows = nf[conflict].copy()
    conf_rows["conflicting_positive_observations"] = pd.Series(
        [";".join(np.array(patho)[r]) for r in pos_patho[conflict].to_numpy()], index=conf_rows.index, dtype=object)
    conf_rows["n_conflicting"] = pos_patho[conflict].sum(axis=1)
    conf_rows["support_devices_positive"] = conf_rows[SUPPORT] == 1.0
    conf_rows[["Path", "source_split", "patient_id", "study_id", "Frontal/Lateral", NO_FINDING,
               "conflicting_positive_observations", "n_conflicting", "support_devices_positive"]].to_csv(
        EDA_TABLES / "no_finding_conflicts.csv", index=False)
    by_obs = pos_patho[conflict].sum().sort_values(ascending=False)
    stats["no_finding"] = {
        "no_finding_positive_images": int(len(nf)),
        "conflicts_with_positive_pathology": int(conflict.sum()),
        "conflict_pct_of_no_finding_positive": 100 * conflict.sum() / max(len(nf), 1),
        "conflict_pct_of_all_images": 100 * conflict.sum() / len(full),
        "conflicts_by_observation": {k: int(v) for k, v in by_obs.items() if v > 0},
        "no_finding_with_uncertain_pathology": int(unc_patho.any(axis=1).sum()),
        "no_finding_with_support_devices_positive": int((nf[SUPPORT] == 1.0).sum()),
        "representative_rows": conf_rows["Path"].head(10).tolist(),
        "definition": "No Finding == 1 and any of the 12 pathology observations (excluding Support Devices) == 1",
    }

    # ---------------- Part 8: patient / study structure ----------------
    ids_ok = full.dropna(subset=["patient_id", "study_id"])
    per_patient = ids_ok.groupby("patient_id").agg(images=("Path", "size"), studies=("study_id", "nunique"),
                                                   source=("source_split", lambda s: ",".join(sorted(set(s)))))
    per_study = ids_ok.groupby("study_id").size()
    cross = per_patient[per_patient["source"].str.contains(",")]
    thr = per_patient["studies"].quantile(cfg["chexpert"]["many_studies_quantile"])
    heavy = per_patient[per_patient["studies"] > thr].sort_values("studies", ascending=False)
    heavy.reset_index().to_csv(EDA_TABLES / "chexpert_patients_many_studies.csv", index=False)

    def desc(s: pd.Series) -> dict:
        return {"mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()),
                "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)), "min": int(s.min()),
                "p99": float(s.quantile(0.99)), "max": int(s.max())}

    pat_rows = []
    for name, sub in (("official_train", train), ("official_valid", valid), ("combined", full)):
        sub = sub.dropna(subset=["patient_id"])
        pp = sub.groupby("patient_id").agg(images=("Path", "size"), studies=("study_id", "nunique"))
        ps = sub.groupby("study_id").size()
        for metric, s in (("images_per_patient", pp["images"]), ("studies_per_patient", pp["studies"]),
                          ("images_per_study", ps)):
            pat_rows.append({"cohort": name, "metric": metric, "n_units": int(len(s)), **desc(s)})
    pd.DataFrame(pat_rows).to_csv(EDA_TABLES / "chexpert_patient_study_summary.csv", index=False)
    stats["patients"] = {
        "images_total": int(len(full)), "images_official_train": int(len(train)), "images_official_valid": int(len(valid)),
        "patients_total": int(full["patient_id"].nunique()),
        "patients_official_train": int(train["patient_id"].nunique()),
        "patients_official_valid": int(valid["patient_id"].nunique()),
        "studies_total": int(full["study_id"].nunique()),
        "studies_official_train": int(train["study_id"].nunique()),
        "studies_official_valid": int(valid["study_id"].nunique()),
        "patients_in_both_official_train_and_valid": int(len(cross)),
        "rows_missing_patient_id": int(full["patient_id"].isna().sum()),
        "many_studies_threshold": float(thr), "patients_above_threshold": int(len(heavy)),
        "max_studies_per_patient": int(per_patient["studies"].max()),
        "max_images_per_patient": int(per_patient["images"].max()),
        "study_label_consistency_train": study_label_consistency(train, targets),
    }

    for col, fname, title in (("images", "chexpert_10_images_per_patient", "Images per patient"),
                              ("studies", "chexpert_11_studies_per_patient", "Studies per patient")):
        s = per_patient[col]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        vc = s.value_counts().sort_index()
        ax.bar(vc.index, vc.values, color=SERIES[0], width=0.8)
        ax.set_yscale("log")
        plain_log_ticks(ax, "y")
        ax.set_xlabel(title)
        ax.set_ylabel("Patients (log scale)")
        ax.set_title(f"{title}: distribution", pad=22)
        subtitle(ax, f"Official train + valid, {len(s):,} patients; median {s.median():.0f}, max {s.max()}")
        ax.grid(axis="x", visible=False)
        save_figure(fig, fname)

    # ---------------- Part 9: views ----------------
    view_rows = []
    for name, sub in (("official_train", train), ("official_valid", valid), ("combined", full)):
        n = len(sub)
        for col in ("Frontal/Lateral", "AP/PA"):
            vc = sub[col].fillna("Missing").value_counts()
            for k, v in vc.items():
                view_rows.append({"cohort": name, "field": col, "value": k, "count": int(v), "pct": 100 * v / n})
        mismatch = ((sub["Frontal/Lateral"].str.lower() != sub["view_from_path"]) & sub["view_from_path"].notna()).sum()
        view_rows.append({"cohort": name, "field": "csv_view_vs_filename_mismatch", "value": "mismatch",
                          "count": int(mismatch), "pct": 100 * mismatch / n})
    views = pd.DataFrame(view_rows)
    views.to_csv(EDA_TABLES / "chexpert_view_summary.csv", index=False)
    ctab = pd.crosstab(full["Frontal/Lateral"].fillna("Missing"), full["AP/PA"].fillna("Missing"))
    ctab.to_csv(EDA_TABLES / "chexpert_view_crosstab.csv")
    vc_all = views[views.cohort == "combined"]
    stats["views"] = {f"{f}:{v}": int(c) for f, v, c in zip(vc_all["field"], vc_all["value"], vc_all["count"])}
    frontal_studies = train.groupby("study_id")["Frontal/Lateral"].apply(lambda s: (s == "Frontal").any())
    stats["views"]["train_studies_without_frontal"] = int((~frontal_studies).sum())
    stats["views"]["train_frontal_images"] = int(len(frontal))

    v = views[(views.cohort == "combined") & (views.field == "Frontal/Lateral")]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(v["value"], v["count"], color=SERIES[0], height=0.5)
    for y, (c, p) in enumerate(zip(v["count"], v["pct"])):
        ax.text(c, y, f"  {c:,} ({p:.1f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Images")
    ax.set_title("Frontal vs lateral images", pad=22)
    subtitle(ax, "Official train + valid")
    ax.set_xlim(0, v["count"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_12_view_distribution")

    v = views[(views.cohort == "combined") & (views.field == "AP/PA")].sort_values("count")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(v["value"], v["count"], color=SERIES[0], height=0.55)
    for y, (c, p) in enumerate(zip(v["count"], v["pct"])):
        ax.text(c, y, f"  {c:,} ({p:.2f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_xlabel("Images (log scale)")
    ax.set_title("AP/PA field distribution", pad=22)
    subtitle(ax, "Official train + valid; 'Missing' = field blank (lateral images have no AP/PA value)")
    ax.set_xlim(right=v["count"].max() * 8)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_13_ap_pa_distribution")

    # ---------------- Part 10: demographics ----------------
    ac = cfg["chexpert"]["age"]
    age = full["Age"]
    a = age.dropna()
    demo_rows = [{"variable": "age", "unit": "image", "count": int(a.size), "missing": int(age.isna().sum()),
                  "mean": a.mean(), "median": a.median(), "std": a.std(), "q1": a.quantile(0.25),
                  "q3": a.quantile(0.75), "min": a.min(), "max": a.max()}]
    pat_age = full.dropna(subset=["Age"]).groupby("patient_id")["Age"].median()
    demo_rows.append({"variable": "age_median_per_patient", "unit": "patient", "count": int(pat_age.size),
                      "missing": int(full["patient_id"].nunique() - pat_age.size), "mean": pat_age.mean(),
                      "median": pat_age.median(), "std": pat_age.std(), "q1": pat_age.quantile(0.25),
                      "q3": pat_age.quantile(0.75), "min": pat_age.min(), "max": pat_age.max()})
    sex_img = full["Sex"].fillna("Missing").value_counts()
    sex_pat = full.groupby("patient_id")["Sex"].agg(lambda s: ",".join(sorted(set(s.fillna("Missing")))))
    for k, v in sex_img.items():
        demo_rows.append({"variable": f"sex={k}", "unit": "image", "count": int(v)})
    for k, v in sex_pat.value_counts().items():
        demo_rows.append({"variable": f"sex={k}", "unit": "patient", "count": int(v)})
    pd.DataFrame(demo_rows).to_csv(EDA_TABLES / "chexpert_demographics.csv", index=False)
    age_flag = full[(full["Age"] < ac["plausible_min"]) | (full["Age"] > ac["plausible_max"]) | full["Age"].isna()]
    age_flag[["Path", "source_split", "patient_id", "Age", "Sex"]].to_csv(EDA_TABLES / "chexpert_age_flags.csv", index=False)
    stats["demographics"] = {
        "age_image_level": {k: float(v) for k, v in demo_rows[0].items() if k not in ("variable", "unit")},
        "age_values_below_plausible_min": int((full["Age"] < ac["plausible_min"]).sum()),
        "age_value_counts_below_plausible_min": {str(int(k)): int(v) for k, v in
                                                 full.loc[full["Age"] < ac["plausible_min"], "Age"].value_counts().sort_index().items()},
        "age_values_above_plausible_max": int((full["Age"] > ac["plausible_max"]).sum()),
        "patients_with_implausible_age": int(age_flag["patient_id"].nunique()),
        "sex_images": {k: int(v) for k, v in sex_img.items()},
        "sex_patients": {k: int(v) for k, v in sex_pat.value_counts().items()},
        "patients_with_inconsistent_sex": int(sex_pat.str.contains(",").sum()),
    }

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(a, bins=np.arange(a.min(), a.max() + 2) - 0.5, color=SERIES[0])
    ax.axvline(ac["plausible_min"], color=INK_2, lw=1)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Images")
    ax.set_title("Age distribution", pad=22)
    subtitle(ax, f"Official train + valid, image level (n={a.size:,}); line = {ac['plausible_min']} y "
                 f"plausibility threshold, {int((a < ac['plausible_min']).sum())} images below")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "chexpert_14_age_distribution")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(sex_img.index[::-1], sex_img.values[::-1], color=SERIES[0], height=0.5)
    for y, c in enumerate(sex_img.values[::-1]):
        ax.text(c, y, f"  {c:,} ({100 * c / len(full):.1f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_xlim(right=sex_img.max() * 8)
    ax.set_xlabel("Images (log scale)")
    ax.set_title("Sex distribution", pad=22)
    subtitle(ax, "Official train + valid, image level; descriptive only")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_15_sex_distribution")

    # ---------------- Part 17 figure: imbalance severity ----------------
    order = ia.sort_values("neg_to_pos_ratio_nan_as_neg_unc_masked").index
    sev_colors = {"mild": "#9ec5f4", "moderate": "#5598e7", "severe": "#256abf", "extreme": "#0d366b"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    vals = ia.loc[order, "neg_to_pos_ratio_nan_as_neg_unc_masked"]
    ax.barh(order, vals, color=[sev_colors[s] for s in ia.loc[order, "imbalance_severity"]], height=0.6)
    for yv, (v, s) in enumerate(zip(vals, ia.loc[order, "imbalance_severity"])):
        ax.text(v * 1.08, yv, f"{v:.1f} : 1  ({s})", va="center", fontsize=8, color=INK_2)
    for b in cfg["chexpert"]["severity_bands"][:-1]:
        ax.axvline(b["max_ratio"], color=MUTED, lw=0.8)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_xlim(0.3, vals.max() * 6)
    ax.set_xlabel("(Explicit negatives + unmentioned) : positives (log scale)")
    ax.set_title("Imbalance severity by observation", pad=22)
    subtitle(ax, "Band edges at 3, 10, 50 : 1 (descriptive heuristic, configs/eda.yaml); darker = more severe")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "chexpert_17_imbalance_severity")

    # ---------------- persist ----------------
    stats["label_states_train_all"] = dist[dist.cohort == "official_train_all_images"].set_index("observation")[
        [f"{s}_count" for s in LABEL_STATES]].astype(int).to_dict(orient="index")
    stats["imbalance_train_all"] = ia[["positive_prevalence_all_images", "neg_to_pos_ratio_explicit",
                                       "neg_to_pos_ratio_nan_as_neg_unc_masked", "imbalance_severity",
                                       "prevalence_band", "uncertain_pct", "missing_pct"]].to_dict(orient="index")
    audit.save(EDA_TABLES / "chexpert_audit_log.csv")
    (EDA_SUMMARIES / "chexpert_stats.json").write_text(json.dumps(stats, indent=2, default=float), encoding="utf-8")
    log.info("audit summary:\n%s", audit.summary())
    log.info("done: %s images, %s patients", stats["patients"]["images_total"], stats["patients"]["patients_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
