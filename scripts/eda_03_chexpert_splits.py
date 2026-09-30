"""Stage 3: leak-free CheXpert patient-level splits, pos_weights and sampling analysis.

Split design
    * Custom train/val/test partitions are carved from the OFFICIAL TRAINING
      CSV only. The official validation CSV (radiologist-labelled, no
      uncertain labels) is kept intact as a separate "official_valid" set.
    * The unit of assignment is a SPLIT GROUP: a patient, merged with any
      other patient ID that shares a byte-identical image (exact SHA-256), so
      a possibly duplicated identity can never straddle partitions.
    * Random and group-aware multi-label stratified splits are both computed;
      the pre-declared rule in configs/eda.yaml selects between them.

Requires stage 1 (image audit cache) for duplicate-identity linking.

Usage:  .venv\\Scripts\\python.exe -m scripts.eda_03_chexpert_splits
"""

from __future__ import annotations

import hashlib
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plotting import INK_2, SERIES, SPLIT_COLORS, apply_style, plain_log_ticks, save_figure, subtitle
from src.data.chexpert import load_chexpert_csv, positive_matrix
from src.data.label_policies import LabelPolicy, apply_label_policy, compute_pos_weight
from src.data.splits import (assert_no_group_overlap, max_relative_deviation, prevalence_by_split,
                             random_group_split, stratified_group_split)
from src.imbalance.sampling import (inverse_prevalence_weights, kish_effective_sample_size,
                                    sampler_effect_table, weighted_conditional)
from src.analysis.multilabel import conditional_probability
from src.utils.audit import AuditLog, setup_logging
from src.utils.config import (CACHE_DIR, EDA_SUMMARIES, EDA_TABLES, SPLITS_DIR, ensure_output_dirs, load_paths,
                              load_yaml)
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("chexpert_splits")
OUT = SPLITS_DIR / "chexpert"
POLICIES = [LabelPolicy("mask", "zero"), LabelPolicy("zero", "zero"), LabelPolicy("one", "zero"),
            LabelPolicy("mask", "mask")]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_duplicate_identities(train: pd.DataFrame, image_base) -> tuple[pd.Series, pd.DataFrame]:
    """Union-find over patients that share byte-identical images."""
    cache = CACHE_DIR / "image_audit_full.csv.gz"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run scripts.eda_01_image_audit first")
    aud = pd.read_csv(cache, usecols=["path", "sha256", "readable"], low_memory=False)
    base = str(image_base.resolve()).lower().rstrip("\\/") + "\\"
    aud["rel"] = aud["path"].str.lower().str.replace(base, "", regex=False).str.replace("\\", "/", regex=False)
    key = train["Path"].str.lower()
    m = train.assign(rel=key).merge(aud[["rel", "sha256"]], on="rel", how="left")
    if m["sha256"].isna().any():
        raise RuntimeError(f"{int(m['sha256'].isna().sum())} training images have no hash in the audit cache")
    parent = {p: p for p in train["patient_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = []
    for h, g in m.groupby("sha256"):
        pats = sorted(g["patient_id"].unique())
        if len(pats) > 1:
            links.append({"sha256": h, "patients": ";".join(pats), "paths": ";".join(g["Path"])})
            for p in pats[1:]:
                ra, rb = find(pats[0]), find(p)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    group = pd.Series({p: find(p) for p in parent}, name="split_group_id")
    return group, pd.DataFrame(links, columns=["sha256", "patients", "paths"])


def main() -> int:
    setup_logging()
    ensure_output_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    cfg = load_yaml("eda.yaml")
    ccfg = cfg["chexpert"]
    seed = int(cfg["seed"])
    ratios = ccfg["split_ratios"]
    paths = load_paths()
    audit = AuditLog("chexpert_splits")
    train, targets = load_chexpert_csv(paths.chexpert_train_csv, "official_train", audit)
    valid, _ = load_chexpert_csv(paths.chexpert_valid_csv, "official_valid", audit)

    # ---------------- grouping ----------------
    group_of, links = link_duplicate_identities(train, paths.chexpert_image_base)
    links.to_csv(EDA_TABLES / "chexpert_cross_patient_duplicate_links.csv", index=False)
    train["split_group_id"] = train["patient_id"].map(group_of)
    n_merged = int((group_of != group_of.index).sum())
    log.info("cross-patient exact-duplicate links: %d; patients merged into another group: %d", len(links), n_merged)

    Y = positive_matrix(train, targets)
    gdf = pd.DataFrame(Y.astype(int), columns=targets)
    gdf["split_group_id"] = train["split_group_id"].to_numpy()
    gdf["n_images"] = 1
    agg = gdf.groupby("split_group_id").sum()
    gids, G_lab, G_size = agg.index.to_numpy(), agg[targets].to_numpy(), agg["n_images"].to_numpy()

    # ---------------- candidate splits ----------------
    min_pos = ccfg["split_selection"]["min_positive_images"]
    rand = random_group_split(gids, G_size, ratios, seed)
    strat = stratified_group_split(gids, G_lab, G_size, ratios, seed)
    s_rand = train["split_group_id"].map(rand).to_numpy()
    s_strat = train["split_group_id"].map(strat).to_numpy()
    dev_rand = max_relative_deviation(Y, s_rand, min_pos)
    dev_strat = max_relative_deviation(Y, s_strat, min_pos)
    repeats = ccfg["split_selection"]["random_seed_repeats"]
    seed_devs = []
    for k in range(repeats):
        s_k = train["split_group_id"].map(random_group_split(gids, G_size, ratios, seed + 1000 + k)).to_numpy()
        seed_devs.append(max_relative_deviation(Y, s_k, min_pos))
    p95 = float(np.percentile(seed_devs, 95))
    thr = ccfg["split_selection"]["max_relative_deviation"]
    use_strat = dev_rand > thr or p95 > thr
    chosen_name = "stratified_group" if use_strat else "random_group"
    chosen = strat if use_strat else rand
    decision = {
        "rule": f"stratify if max relative prevalence deviation (labels with >= {min_pos} positives) > {thr} "
                f"for the seeded random split or its 95th percentile over {repeats} seeds",
        "random_seeded_max_rel_dev": dev_rand, "random_p95_max_rel_dev": p95,
        "random_median_max_rel_dev": float(np.median(seed_devs)), "random_max_max_rel_dev": float(np.max(seed_devs)),
        "stratified_max_rel_dev": dev_strat, "chosen": chosen_name,
    }
    log.info("split decision: %s", decision)
    pd.DataFrame({"seed_offset": np.arange(repeats), "max_relative_deviation": seed_devs}).to_csv(
        EDA_TABLES / "chexpert_random_split_seed_deviation.csv", index=False)

    train["split"] = train["split_group_id"].map(chosen)
    valid["split_group_id"] = valid["patient_id"]
    valid["split"] = "official_valid"
    manifest = pd.concat([train, valid], ignore_index=True)

    # ---------------- leakage assertions (fail hard) ----------------
    assert_no_group_overlap(manifest, "patient_id")
    assert_no_group_overlap(manifest, "split_group_id")
    assert manifest["split"].notna().all(), "unassigned images"
    assert len(manifest) == len(train) + len(valid)

    cols = ["Path", "patient_id", "split_group_id", "study_id", "Frontal/Lateral", "AP/PA", "split"]
    # mtime=0: no timestamp in the gzip header, so identical splits give an identical file.
    manifest[cols].to_csv(OUT / "chexpert_image_manifest.csv.gz", index=False,
                          compression={"method": "gzip", "mtime": 0})
    pat = (manifest.groupby("patient_id").agg(split_group_id=("split_group_id", "first"), split=("split", "first"),
                                              n_images=("Path", "size"), n_studies=("study_id", "nunique"))
           .reset_index())
    pat.to_csv(OUT / "chexpert_patient_splits.csv", index=False)

    # ---------------- split quality ----------------
    frontal_mask = (train["Frontal/Lateral"] == "Frontal").to_numpy()
    dist = []
    for name, s in (("chosen:" + chosen_name, train["split"].to_numpy()), ("comparator:random_group", s_rand),
                    ("comparator:stratified_group", s_strat)):
        for view, sel in (("all_images", np.ones(len(train), bool)), ("frontal_images", frontal_mask)):
            d = prevalence_by_split(Y[sel], s[sel], targets)
            d.insert(0, "view_subset", view)
            d.insert(0, "strategy", name)
            dist.append(d)
    Yv = positive_matrix(valid, targets)
    dv = prevalence_by_split(Yv, np.full(len(valid), "official_valid"), targets)
    # A single-cohort call yields only the "all" row; it IS the official validation set.
    dv = dv[dv.split == "all"].assign(split="official_valid", strategy="official", view_subset="all_images",
                                      relative_deviation=np.nan)
    dist.append(dv)
    dist = pd.concat(dist, ignore_index=True)
    # uncertain rate per split for the chosen split (label-noise balance)
    unc_rows = []
    for s in ("train", "val", "test"):
        sub = train[train["split"] == s]
        for t in targets:
            unc_rows.append({"split": s, "observation": t, "uncertain_pct": 100 * (sub[t] == -1).mean(),
                             "missing_pct": 100 * sub[t].isna().mean()})
    unc_split = pd.DataFrame(unc_rows)
    dist = dist.merge(unc_split, on=["split", "observation"], how="left")
    dist.loc[~dist["strategy"].str.startswith("chosen"), ["uncertain_pct", "missing_pct"]] = np.nan
    dist.to_csv(EDA_TABLES / "chexpert_split_distribution.csv", index=False)

    sizes = []
    for s in ("train", "val", "test", "official_valid"):
        sub = manifest[manifest["split"] == s]
        sizes.append({"split": s, "images": len(sub), "frontal_images": int((sub["Frontal/Lateral"] == "Frontal").sum()),
                      "lateral_images": int((sub["Frontal/Lateral"] == "Lateral").sum()),
                      "patients": sub["patient_id"].nunique(), "split_groups": sub["split_group_id"].nunique(),
                      "studies": sub["study_id"].nunique(),
                      "image_share_of_official_train": len(sub) / len(train) if s != "official_valid" else np.nan,
                      "patient_share_of_official_train": sub["patient_id"].nunique() / train["patient_id"].nunique()
                      if s != "official_valid" else np.nan})
    sizes = pd.DataFrame(sizes)
    sizes.to_csv(EDA_TABLES / "chexpert_split_sizes.csv", index=False)

    ch = dist[(dist.strategy == "chosen:" + chosen_name) & (dist.view_subset == "frontal_images") & (dist.split != "all")]
    order = ch[ch.split == "train"].sort_values("prevalence")["observation"].tolist()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = {o: i for i, o in enumerate(order)}
    offsets = {"train": -0.2, "val": 0.0, "test": 0.2}
    for s in ("train", "val", "test"):
        sub = ch[ch.split == s]
        ax.scatter(100 * sub["prevalence"], [y[o] + offsets[s] for o in sub["observation"]], s=40,
                   color=SPLIT_COLORS[s], label=s, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_xlabel("Positive prevalence, % of frontal images (log scale)")
    ax.set_title("Positive prevalence by split", pad=22)
    subtitle(ax, f"Patient-level {chosen_name.replace('_', ' ')} split of the official training set (seed {seed})")
    ax.legend(loc="lower right")
    save_figure(fig, "chexpert_16_split_prevalence")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(100 * np.array(seed_devs), bins=20, color=SERIES[0], label=f"Random patient split, {repeats} seeds")
    ax.axvline(100 * dev_strat, color=SERIES[1], lw=2, label=f"Stratified group split (seed {seed})")
    ax.axvline(100 * thr, color=INK_2, lw=1, ls="-", label=f"Pre-declared {100 * thr:.0f}% threshold")
    ax.set_xlabel("Worst relative prevalence deviation across labels and splits (%)")
    ax.set_ylabel("Seeds")
    ax.set_title("Prevalence stability: random vs stratified", pad=22)
    subtitle(ax, f"Labels with >= {min_pos} positive images; all images of the official training set")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "chexpert_16b_split_stability")

    # ---------------- Part 14: training-only pos_weight ----------------
    tr = train[train["split"] == "train"]
    rows, weights_json = [], {}
    for view, sub in (("frontal", tr[tr["Frontal/Lateral"] == "Frontal"]), ("all_views", tr)):
        raw = sub[targets].to_numpy(dtype=float)
        for pol in POLICIES:
            t, m = apply_label_policy(raw, pol)
            pw = compute_pos_weight(t, m)
            weights_json.setdefault(view, {})[pol.name] = dict(zip(targets, map(float, pw["pos_weight"])))
            for j, lab in enumerate(targets):
                rows.append({"view_subset": view, "label_policy": pol.name, "uncertain_policy": pol.uncertain,
                             "missing_policy": pol.missing, "observation": lab, "n_train_images": len(sub),
                             "n_positive": int(pw["n_positive"][j]), "n_negative": int(pw["n_negative"][j]),
                             "n_masked": int(pw["n_masked"][j]), "pos_weight": float(pw["pos_weight"][j]),
                             "degenerate_fallback": bool(pw["degenerate"][j])})
    pw_df = pd.DataFrame(rows)
    pw_df.to_csv(EDA_TABLES / "training_pos_weights.csv", index=False)
    train_csv_hash = sha256_file(paths.chexpert_train_csv)
    (OUT / "training_pos_weights.json").write_text(json.dumps({
        "description": "pos_weight_c = N_negative,c / N_positive,c over entries that enter the loss, "
                       "computed on the TRAINING partition only. Degenerate classes fall back to 1.0.",
        "split_strategy": chosen_name, "seed": seed, "train_csv_sha256": train_csv_hash,
        "targets": targets, "weights": weights_json}, indent=2), encoding="utf-8")

    # ---------------- Part 15: sampling analysis ----------------
    trf = tr[tr["Frontal/Lateral"] == "Frontal"]
    Yt = positive_matrix(trf, targets)
    schemes = {"none": np.ones(len(Yt)),
               "inverse_prevalence_max": inverse_prevalence_weights(Yt, "max"),
               "inverse_prevalence_mean": inverse_prevalence_weights(Yt, "mean"),
               "inverse_prevalence_sqrt_max": inverse_prevalence_weights(Yt, "sqrt_max")}
    eff = sampler_effect_table(Yt, targets, schemes)
    ess = {k: kish_effective_sample_size(w) / len(w) for k, w in schemes.items()}
    top_share = {k: float(np.sort(w)[::-1][: max(1, len(w) // 100)].sum() / w.sum()) for k, w in schemes.items()}
    eff["kish_ess_fraction"] = eff["scheme"].map(ess)
    eff["weight_share_top_1pct_images"] = eff["scheme"].map(top_share)
    eff.to_csv(EDA_TABLES / "chexpert_sampling_effect.csv", index=False)
    C0 = conditional_probability(Yt)
    cond_rows = []
    for k, w in schemes.items():
        if k == "none":
            continue
        Cw = weighted_conditional(Yt, w)
        for i, a in enumerate(targets):
            for j, b in enumerate(targets):
                if i != j:
                    cond_rows.append({"scheme": k, "label_a": a, "label_b": b, "p_b_given_a_original": C0[i, j],
                                      "p_b_given_a_sampled": Cw[i, j], "abs_change": abs(Cw[i, j] - C0[i, j])})
    cond = pd.DataFrame(cond_rows)
    cond.to_csv(EDA_TABLES / "chexpert_sampling_conditional_shift.csv", index=False)
    write_sampling_report(eff, ess, top_share, cond, targets, len(Yt))

    # ---------------- persist metadata ----------------
    meta = {
        "created_by": "scripts/eda_03_chexpert_splits.py", "seed": seed, "ratios": ratios,
        "unit": "split_group_id (patient merged with patients sharing byte-identical images)",
        "source": "official training CSV; official validation CSV kept as 'official_valid'",
        "train_csv_sha256": train_csv_hash, "valid_csv_sha256": sha256_file(paths.chexpert_valid_csv),
        "decision": decision, "cross_patient_duplicate_links": int(len(links)),
        "patients_merged": n_merged, "split_sizes": sizes.to_dict(orient="records"),
        "leakage_checks": {"patient_overlap": 0, "split_group_overlap": 0},
        "environment": environment_snapshot(seed),
    }
    (OUT / "split_metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    (EDA_SUMMARIES / "chexpert_split_stats.json").write_text(json.dumps(
        {"decision": decision, "sizes": sizes.to_dict(orient="records"), "links": int(len(links)),
         "patients_merged": n_merged, "sampling_ess_fraction": ess, "sampling_top1pct_weight_share": top_share},
        indent=2, default=float), encoding="utf-8")
    audit.save(EDA_TABLES / "chexpert_split_audit_log.csv")
    log.info("sizes:\n%s", sizes.to_string(index=False))
    return 0


def write_sampling_report(eff: pd.DataFrame, ess: dict, top_share: dict, cond: pd.DataFrame,
                          targets: list[str], n: int) -> None:
    """Numbers are inserted from the computed tables; the decision rule is stated explicitly."""
    piv = eff.pivot(index="observation", columns="scheme", values="expected_prevalence").loc[targets]
    orig = eff[eff.scheme == "none"].set_index("observation")["original_prevalence"].loc[targets]
    rare = orig.sort_values().index[:4].tolist()
    common = orig.sort_values().index[-4:].tolist()
    lines = [
        "# Sampling analysis (CheXpert training partition, frontal images)",
        "",
        "_Generated by `scripts/eda_03_chexpert_splits.py` from the actual training partition "
        f"(n = {n:,} frontal images). Positives = label == 1; uncertain and blank are not positive._",
        "",
        "## Why multi-label sampling is not a free lunch",
        "",
        "A sampler draws **images**, not labels. Upweighting an image because it carries a rare label also "
        "upweights every other label on that image. Because rare findings co-occur with common ones "
        "(see `chexpert_cooccurrence.csv`), rare-label oversampling also inflates common labels, and the "
        "*effective* label distribution seen by the loss changes in ways that per-label reasoning does not predict.",
        "",
        "## Schemes evaluated (expected effect, no training performed)",
        "",
        "* `inverse_prevalence_max` - weight = 1 / prevalence of the image's rarest positive label",
        "* `inverse_prevalence_mean` - mean inverse prevalence over the image's positive labels",
        "* `inverse_prevalence_sqrt_max` - square root of the max scheme (tempered)",
        "* images with no positive label form their own group weighted by its inverse prevalence",
        "",
        "## Expected positive prevalence under each scheme (%)",
        "",
        "| Observation | none | inv_max | inv_mean | inv_sqrt_max |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in targets:
        lines.append(f"| {t} | {100 * piv.loc[t, 'none']:.2f} | {100 * piv.loc[t, 'inverse_prevalence_max']:.2f} | "
                     f"{100 * piv.loc[t, 'inverse_prevalence_mean']:.2f} | "
                     f"{100 * piv.loc[t, 'inverse_prevalence_sqrt_max']:.2f} |")
    lines += ["", "## Cost in effective sample size", "",
              "| Scheme | Kish ESS / N | Weight share of top 1% images |", "|---|---:|---:|"]
    for k in ess:
        lines.append(f"| {k} | {ess[k]:.3f} | {100 * top_share[k]:.1f}% |")
    worst = cond.sort_values("abs_change", ascending=False).groupby("scheme").head(3)
    lines += ["", "## Largest distortions of P(B | A)", "",
              "| Scheme | A | B | original | sampled |", "|---|---|---|---:|---:|"]
    for r in worst.itertuples():
        lines.append(f"| {r.scheme} | {r.label_a} | {r.label_b} | {r.p_b_given_a_original:.3f} | {r.p_b_given_a_sampled:.3f} |")

    m = eff[eff.scheme == "inverse_prevalence_max"].set_index("observation")
    common_infl = m.loc[common, "inflation_factor"]
    rare_infl = m.loc[rare, "inflation_factor"]
    lines += ["", "## Evidence-based recommendation", "",
              f"* Under `inverse_prevalence_max`, the four rarest labels ({', '.join(rare)}) are inflated by "
              f"{rare_infl.min():.1f}x-{rare_infl.max():.1f}x, while the four most common "
              f"({', '.join(common)}) change by {common_infl.min():.2f}x-{common_infl.max():.2f}x.",
              f"* The same scheme reduces the Kish effective sample size to {100 * ess['inverse_prevalence_max']:.1f}% "
              f"of N, and the top 1% of images carry {100 * top_share['inverse_prevalence_max']:.1f}% of the sampling mass "
              "(repeated exposure of few images -> overfitting risk).",
              f"* The largest shift in any P(B | A) is {cond['abs_change'].max():.3f} "
              f"(scheme `{cond.loc[cond['abs_change'].idxmax(), 'scheme']}`).",
              ""]
    # Pre-stated rule: a sampler is "costly" if it halves the effective sample size or
    # shifts any conditional co-occurrence probability by more than 0.10.
    costly = ess["inverse_prevalence_max"] < 0.5 or cond.loc[cond.scheme == "inverse_prevalence_max", "abs_change"].max() > 0.10
    lines.append("_Decision rule (stated before reading the numbers): a sampler is considered costly if it "
                 "reduces Kish ESS below 50% of N or shifts any P(B | A) by more than 0.10._")
    lines.append("")
    if costly:
        lines.append("**Recommendation.** The rule is met, so do not enable a sampler in the primary experiments. "
                     "Loss-level mechanisms (positive-weighted BCE, focal, asymmetric loss) act per label and leave "
                     "the image distribution, co-occurrence structure and effective sample size intact, so they "
                     "should be compared first. The tempered `inverse_prevalence_sqrt_max` sampler is the only "
                     "variant worth keeping as an optional later ablation, and only if loss-level methods leave "
                     "rare labels under-served.")
    else:
        lines.append("**Recommendation.** The rule is not met: the sampler's side effects are modest on this data. "
                     "A sampler may be included as one controlled ablation after the loss-level comparison.")
    lines += ["", "Validation and test data are never resampled.", ""]
    (EDA_SUMMARIES / "sampling_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
