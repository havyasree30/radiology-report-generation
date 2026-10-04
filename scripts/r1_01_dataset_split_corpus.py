"""R1 step 1-5: IU X-Ray inventory, leakage-safe study-level split (reused from Phase 1), retrieval corpus and the
IU finding mapping. The locked retrieval-test studies are listed by id only; no text, label or finding of a test study
is read, mapped or analysed here.

    .venv\\Scripts\\python.exe -m scripts.r1_01_dataset_split_corpus
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd
import yaml

from src.classification.final_test import sha256_file
from src.classification.labels import LABELS
from src.data.retrieval_corpus import assert_train_only
from src.retrieval.findings import ABNORMAL, eval_set, map_study, parse_mesh
from src.retrieval.split_checks import images_in_multiple_splits, pairwise_overlaps, studies_spanning_splits
from src.retrieval.text import build_retrieval_text, clean_section
from src.utils.config import PROJECT_ROOT, load_paths

OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
SPL = PROJECT_ROOT / "data/splits/iu_xray"

# Decisions fixed BEFORE any retrieval result exists.
UNCERTAIN_REASONS = {
    "Pulmonary Congestion": "vascular congestion is not equivalent to oedema in this vocabulary (the Phase 1 EDA lexicon mapped it to Edema; not adopted here)",
    "Density": "unspecific radiographic density; may or may not be an opacity",
    "Mediastinum": "'prominent mediastinum' is not clearly 'enlarged cardiomediastinum'; lymph-node qualifiers are a different finding",
    "Cardiac Shadow": "mapped to Cardiomegaly only with the qualifier 'enlarged'; other qualifiers (e.g. borderline, prominent) are ambiguous",
    "Thickening": "mapped to Pleural Other only with the qualifier 'pleura'; other thickening (bronchovascular, lung) is ambiguous"}
CONDITIONAL = [{"term": "Cardiac Shadow", "any_qualifier": ["enlarged"], "class": "Cardiomegaly",
                "reason": "an enlarged cardiac shadow is cardiomegaly"},
               {"term": "Thickening", "any_qualifier": ["pleura"], "class": "Pleural Other",
                "reason": "pleural thickening is the classifier's 'Pleural Other' concept"}]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    rep = pd.read_csv(paths.iu_reports_csv)
    proj = pd.read_csv(paths.iu_projections_csv)
    splits = pd.read_csv(SPL / "iu_report_splits.csv")
    imman = pd.read_csv(SPL / "iu_image_manifest.csv")
    corpus_ids = pd.read_csv(SPL / "retrieval_corpus_uids.csv")["uid"].astype(str).tolist()
    rep["uid"] = rep["uid"].astype(str)
    proj["uid"] = proj["uid"].astype(str)
    splits["uid"] = splits["uid"].astype(str)
    imman["uid"] = imman["uid"].astype(str)

    # ------------------------------------------------------------------ 1. inventory (counts only; no per-test analysis)
    on_disk = {p.name for p in paths.iu_images_dir.iterdir() if p.is_file()}
    listed = set(proj["filename"])
    fr = proj.groupby("uid")["projection"].apply(lambda s: (s == "Frontal").sum())
    la = proj.groupby("uid")["projection"].apply(lambda s: (s == "Lateral").sum())
    f_clean, i_clean = rep["findings"].map(clean_section), rep["impression"].map(clean_section)
    has_f, has_i = f_clean != "", i_clean != ""
    mesh_terms, prob_terms = Counter(), Counter()
    for m in rep["MeSH"].dropna():
        mesh_terms.update(t for t, _ in parse_mesh(m))
    for m in rep["Problems"].dropna():
        prob_terms.update(t.strip() for t in str(m).split(";") if t.strip())
    n_img_per = proj.groupby("uid").size()
    inventory = {
        "source_files": {"reports_csv": str(paths.iu_reports_csv.relative_to(PROJECT_ROOT)), "projections_csv": str(paths.iu_projections_csv.relative_to(PROJECT_ROOT)),
                         "images_dir": str(paths.iu_images_dir.relative_to(PROJECT_ROOT)), "downloaded_by_r1": False},
        "reports_studies": int(rep["uid"].nunique()), "report_columns": rep.columns.tolist(),
        "images_listed_in_projections_csv": int(len(proj)), "image_files_on_disk": len(on_disk),
        "listed_images_missing_on_disk": len(listed - on_disk), "disk_images_not_listed": len(on_disk - listed),
        "frontal_images": int((proj["projection"] == "Frontal").sum()), "lateral_images": int((proj["projection"] == "Lateral").sum()),
        "studies_with_multiple_images": int((n_img_per > 1).sum()), "studies_with_one_image": int((n_img_per == 1).sum()),
        "images_per_study_distribution": {str(k): int(v) for k, v in n_img_per.value_counts().sort_index().items()},
        "studies_without_images": int(len(set(rep["uid"]) - set(proj["uid"]))), "image_uids_without_report": int(len(set(proj["uid"]) - set(rep["uid"]))),
        "studies_with_at_least_one_frontal": int((fr > 0).sum()), "studies_without_frontal": int(rep["uid"].map(fr).fillna(0).eq(0).sum()),
        "studies_with_more_than_one_frontal": int((fr > 1).sum()), "studies_with_lateral": int((la > 0).sum()),
        "sections_after_cleaning": {"with_findings": int(has_f.sum()), "with_impression": int(has_i.sum()), "with_both": int((has_f & has_i).sum()),
                                    "with_neither": int((~has_f & ~has_i).sum()), "findings_only": int((has_f & ~has_i).sum()),
                                    "impression_only": int((~has_f & has_i).sum())},
        "other_text_fields": {"indication_present": int(rep["indication"].notna().sum()), "comparison_present": int(rep["comparison"].notna().sum()),
                              "note": "indication/comparison are metadata and are NOT part of the retrieval text"},
        "terminology_available": {"MeSH": {"studies_with_MeSH": int(rep["MeSH"].notna().sum()), "distinct_terms": len(mesh_terms),
                                           "studies_indexed_normal": int(sum(any(t.lower() == "normal" for t, _ in parse_mesh(m)) for m in rep["MeSH"].dropna())),
                                           "studies_no_indexing": int(sum(any(t.lower() == "no indexing" for t, _ in parse_mesh(m)) for m in rep["MeSH"].dropna())),
                                           "top_terms": mesh_terms.most_common(15)},
                                  "Problems": {"studies_with_Problems": int(rep["Problems"].notna().sum()), "distinct_terms": len(prob_terms),
                                               "identical_to_MeSH_terms_in_this_export": bool((rep["Problems"].fillna("") == rep["MeSH"].fillna("").map(lambda s: ";".join(t for t, _ in parse_mesh(s)))).mean() > 0.99)}},
        "frontal_view_rule_for_classifier_query": "first Frontal image of the study in sorted filename order (deterministic)"}
    (OUT / "iu_xray_dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 2/3. study-level split (reused, never re-split)
    sp = splits.set_index("uid")["split"]
    train_u, val_u, test_u = (sorted(sp.index[sp == s], key=int) for s in ("train", "val", "test"))
    corpus_u = sorted(corpus_ids, key=int)
    assert_train_only(corpus_u, splits)
    per_uid_splits = imman.groupby("uid")["split"].nunique()
    spanning, multi_img = studies_spanning_splits(imman), images_in_multiple_splits(imman)
    ov = pairwise_overlaps(corpus_u, val_u, test_u)
    text_key = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower())  # noqa: E731
    combo = pd.Series({u: " ".join(text_key(build_retrieval_text(f, i)[0]).split()) for u, f, i in zip(rep["uid"], rep["findings"], rep["impression"])})
    ctext = set(combo[corpus_u]) - {""}
    dup_val = sum(combo[u] in ctext for u in val_u if combo[u])
    integrity = {
        **ov,
        "train_and_validation": len(set(train_u) & set(val_u)), "train_and_test": len(set(train_u) & set(test_u)),
        "corpus_is_train_only_assert_train_only": True,
        "studies_whose_images_span_more_than_one_split": len(spanning),
        "image_files_in_more_than_one_split": len(multi_img),
        "studies_in_split_table": int(len(splits)), "studies_in_report_csv": int(rep["uid"].nunique()),
        "split_table_matches_report_csv": set(splits["uid"]) == set(rep["uid"]),
        "image_manifest_matches_projections_csv": set(imman["filename"]) == listed,
        "duplicate_study_ids": int(rep["uid"].duplicated().sum()),
        "all_images_of_every_study_share_the_study_split": bool((per_uid_splits == 1).all())}
    ok = (integrity["corpus_and_validation"] == integrity["corpus_and_test"] == integrity["validation_and_test"] == 0
          and integrity["studies_whose_images_span_more_than_one_split"] == 0 and integrity["image_files_in_more_than_one_split"] == 0
          and integrity["split_table_matches_report_csv"] and integrity["duplicate_study_ids"] == 0)
    meta = json.loads((SPL / "split_metadata.json").read_text(encoding="utf-8"))
    split_json = {
        "status": "PASS" if ok else "FAIL",
        "decision": ("The Phase 1 study-level split is REUSED unchanged (project rule: never re-split ad hoc). It already is the preferred "
                     "70/15/15 split with a fixed seed, its unit is the report uid so every image of a study stays together, and it "
                     "stratifies on MeSH normal vs other. No other ratio is materially better for 3,851 studies, so no new split was created."),
        "provenance": {"created_by": meta["created_by"] if "created_by" in meta else "scripts/eda_04_iu_xray.py", "seed": meta["seed"], "ratios": meta["ratios"],
                       "unit": meta["unit"], "strata": meta["strata"], "iu_report_splits_sha256": sha256_file(SPL / "iu_report_splits.csv"),
                       "retrieval_corpus_uids_sha256": sha256_file(SPL / "retrieval_corpus_uids.csv"),
                       "iu_image_manifest_sha256": sha256_file(SPL / "iu_image_manifest.csv")},
        "roles": {"reference_corpus": "split == train with non-empty Findings/Impression (the ONLY studies that may be retrieved)",
                  "retrieval_validation_queries": "split == val (R1 and any retrieval optimisation may use these)",
                  "locked_retrieval_test": "split == test; LOCKED - ids only, no text, finding or retrieval result read in R1"},
        "sizes": {"train_studies": len(train_u), "reference_corpus_studies": len(corpus_u), "validation_studies": len(val_u), "locked_test_studies": len(test_u),
                  "train_images": int(imman[imman.split == "train"].shape[0]), "validation_images": int(imman[imman.split == "val"].shape[0]),
                  "locked_test_images": int(imman[imman.split == "test"].shape[0])},
        "integrity": integrity,
        "informational": {"validation_studies_whose_cleaned_retrieval_text_is_an_exact_copy_of_a_corpus_report": int(dup_val),
                          "note": "identical text across different studies is generic templated reporting (e.g. normal chest), not self-retrieval; each study id is unique and absent from the corpus"},
        "study_ids": {"reference_corpus": corpus_u, "validation": val_u, "locked_test": test_u}}
    (OUT / "retrieval_split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    if not ok:
        raise SystemExit("STOP: retrieval split integrity FAILED")

    # ------------------------------------------------------------------ 5. finding mapping (corpus + validation only)
    lex = yaml.safe_load((PROJECT_ROOT / "configs/iu_concept_lexicon.yaml").read_text(encoding="utf-8"))["concepts"]
    direct = {t: cls for cls, v in lex.items() for t in v["mesh"] if t not in UNCERTAIN_REASONS}
    work = rep.set_index("uid").loc[corpus_u + val_u]
    freq = Counter()
    for m in work["MeSH"].dropna():
        freq.update(t for t, _ in parse_mesh(m))
    mapping = {"version": "r1-v1", "classifier_vocabulary": list(LABELS),
               "normal_term": {"normal": "No Finding (only when no abnormal finding is mapped)"},
               "no_indexing_term": "No Indexing -> no representation",
               "direct": dict(sorted(direct.items())), "conditional": CONDITIONAL,
               "uncertain": {t: {"reason": r, "occurrences_in_corpus_and_validation": int(freq.get(t, 0))} for t, r in UNCERTAIN_REASONS.items()},
               "unmapped_out_of_vocabulary": {t: int(n) for t, n in sorted(freq.items(), key=lambda kv: -kv[1])
                                              if t not in direct and t not in UNCERTAIN_REASONS and t.lower() not in ("normal", "no indexing")},
               "source": "configs/iu_concept_lexicon.yaml (mesh lists, Phase 1) with the explicit changes listed under 'uncertain' / 'conditional'",
               "note": "Terms outside the 14-class vocabulary (e.g. spine, aorta, granuloma, emphysema) are NOT mapped and never invented as findings."}
    (OUT / "iu_finding_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 4/5. corpus + study findings
    front = proj[proj["projection"] == "Frontal"].sort_values(["uid", "filename"]).groupby("uid")["filename"].first()
    rows, corp = [], []
    for u in corpus_u + val_u:
        r = work.loc[u]
        ms = map_study(r["MeSH"], mapping)
        es = eval_set(ms["findings"], ms["normal"])
        part = "reference_corpus" if u in set(corpus_u) else "validation"
        rows.append({"uid": u, "partition": part, "mesh_raw": r["MeSH"], "mapped_findings": ";".join(ms["findings"]), "indexed_normal": ms["normal"],
                     "eval_set": ";".join(sorted(es, key=LABELS.index)) if es else "", "has_eval_set": es is not None,
                     "uncertain_terms": ";".join(ms["uncertain_terms"]), "unmapped_terms": ";".join(ms["unmapped_terms"]),
                     "frontal_image": front.get(u, "")})
        if part == "reference_corpus":
            text, src = build_retrieval_text(r["findings"], r["impression"])
            corp.append({"uid": u, "findings_original": r["findings"], "impression_original": r["impression"],
                         "findings_clean": clean_section(r["findings"]), "impression_clean": clean_section(r["impression"]),
                         "retrieval_text": text, "text_source": src, "mesh_tags": r["MeSH"], "problems_tags": r["Problems"],
                         "mapped_findings": ";".join(ms["findings"]), "eval_set": rows[-1]["eval_set"]})
    sf = pd.DataFrame(rows)
    cdf = pd.DataFrame(corp)
    sf.to_csv(OUT / "iu_study_findings.csv", index=False)
    cdf.to_csv(OUT / "retrieval_corpus.csv", index=False)
    vt = work.loc[val_u, ["findings", "impression"]].copy()
    vt["retrieval_text"] = [build_retrieval_text(f, i)[0] for f, i in zip(vt["findings"], vt["impression"])]
    vt.reset_index().to_csv(OUT / "validation_report_text_for_error_analysis.csv", index=False)
    assert (cdf["retrieval_text"].str.len() > 0).all() and len(cdf) == len(corpus_u)

    vs, cs = sf[sf.partition == "validation"], sf[sf.partition == "reference_corpus"]
    mapping_stats = {"corpus_studies": len(cs), "validation_studies": len(vs),
                     "corpus_with_eval_set": int(cs.has_eval_set.sum()), "validation_with_eval_set": int(vs.has_eval_set.sum()),
                     "validation_without_eval_set_unusable_as_queries": int((~vs.has_eval_set).sum()),
                     "validation_indexed_normal": int(vs.indexed_normal.sum()),
                     "validation_with_abnormal_mapped_finding": int((vs.mapped_findings != "").sum()),
                     "validation_normal_and_abnormal_conflict": int((vs.indexed_normal & (vs.mapped_findings != "")).sum()),
                     "validation_studies_with_uncertain_terms": int((vs.uncertain_terms != "").sum()),
                     "validation_studies_with_unmapped_terms": int((vs.unmapped_terms != "").sum()),
                     "validation_studies_with_frontal_image": int((vs.frontal_image != "").sum()),
                     "mapped_class_frequency_validation": {l: int(sum(l in s.split(";") for s in vs.mapped_findings)) for l in ABNORMAL},
                     "mapped_class_frequency_corpus": {l: int(sum(l in s.split(";") for s in cs.mapped_findings)) for l in ABNORMAL},
                     "corpus_text_source": cdf["text_source"].value_counts().to_dict()}
    (OUT / "r1_mapping_and_corpus_stats.json").write_text(json.dumps(mapping_stats, indent=2), encoding="utf-8")
    print(json.dumps({"split": split_json["status"], "sizes": split_json["sizes"], "integrity": {k: v for k, v in integrity.items() if isinstance(v, int)},
                      "inventory": {k: inventory[k] for k in ("reports_studies", "images_listed_in_projections_csv", "image_files_on_disk", "frontal_images", "lateral_images", "studies_with_multiple_images", "studies_without_frontal")},
                      "sections": inventory["sections_after_cleaning"], "mapping": mapping_stats}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
