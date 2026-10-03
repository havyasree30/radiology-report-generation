"""G1 step 1: evaluation populations, frozen classifier outputs, frozen R2 retrieval (hybrid RRF, Top-5), evidence tables,
generator payloads (classifier + retrieval ONLY), the B0 rule-based baseline and leakage checks. No API call.

    .venv\\Scripts\\python.exe -m scripts.g1_01_build_cases
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.generation.baseline import b0_report
from src.generation.metrics import norm_sentence, sentences
from src.generation.prompts import assert_payload_clean, build_payload, prompt_hash, render_user_message
from src.retrieval.r2_context import R1, R2
from src.retrieval.text import clean_section
from src.utils.config import PROJECT_ROOT
from scripts.r2_02_experiments import Engine

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
EXP = PROJECT_ROOT / "results/classification/experiments"
TOPK = 5


def anon(uid: str) -> str:
    return "case_" + hashlib.sha256(f"g1-anon-v1|{uid}".encode()).hexdigest()[:10]


def main() -> int:
    G1.mkdir(parents=True, exist_ok=True)
    verify_freeze(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", PROJECT_ROOT)
    cfg = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    assert cfg["provisional_top_k"] == TOPK and cfg["retriever"]["selected"] == "hybrid" and cfg["query_construction"]["top_n_findings"] == 3
    assert not cfg["query_construction"]["weighting_selected"] and not cfg["query_construction"]["clinical_phrase_expansion_selected"] and not cfg["diversity_reranking"]["mmr_selected"]
    eng = Engine()
    ctx = eng.ctx
    ctx._cache_path = G1 / "query_embedding_cache_g1.npz"          # R2 artifacts are read-only: new query vectors are cached under G1
    spec = {"phrases": "names", "normal": cfg["query_construction"]["no_finding_query_phrase"], "top_n": 3, "weighting": None, "oracle": False}
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("").set_index("uid")
    vt = pd.read_csv(R1 / "validation_report_text_for_error_analysis.csv", dtype={"uid": str})
    vt["findings_clean"], vt["impression_clean"] = vt["findings"].map(clean_section), vt["impression"].map(clean_section)
    vt = vt.set_index("uid")
    r2_top = pd.read_csv(R2 / "final_pipeline_top10.csv.gz", dtype={"uid": str, "retrieved_uid": str})
    r2_top = r2_top[r2_top.config == "FINAL_pipeline"]

    val = ctx.val
    frontal = [u for u in val.index if u in ctx.cls.index]          # studies with a frontal image (classifier output exists)
    has_ref = lambda u: bool(str(vt.loc[u, "retrieval_text"]).strip()) and str(vt.loc[u, "retrieval_text"]) != "nan"  # noqa: E731
    primary = [u for u in frontal if has_ref(u) and u in ctx.cls.index]
    cases, leak_rows, integrity = [], [], {"r2_primary_studies_checked": 0, "top5_identical_to_r2": 0, "mismatches": []}
    corpus_ids, test_ids = set(ctx.ids), ctx.test_ids
    for u in primary:
        positives = ctx.positives(u)
        probs = ctx.probs(u)
        nf = ctx.no_finding_positive(u)
        q = eng.query(u, spec)
        retrieved = []
        if q is not None:
            ranked, info = eng.retrieve(q, "hybrid")
            for r, d in enumerate(ranked[:TOPK], 1):
                dr, br, rs = info["prov"][d]
                retrieved.append({"rank": r, "study_id": d, "dense_rank": dr if dr is not None else -1, "bm25_rank": br if br is not None else -1, "rrf_score": float(rs),
                                  "findings": corpus.loc[d, "findings_clean"], "impression": corpus.loc[d, "impression_clean"], "mapped_findings": sorted(ctx.doc_set[d])})
        if u in set(ctx.primary):                                 # R2 Top-5 integrity
            integrity["r2_primary_studies_checked"] += 1
            ref5 = r2_top[r2_top.uid == u].sort_values("rank").retrieved_uid.tolist()[:TOPK]
            ok = ref5 == [r["study_id"] for r in retrieved]
            integrity["top5_identical_to_r2"] += int(ok)
            if not ok:
                integrity["mismatches"].append(u)
        payload = build_payload([{"finding": l, "calibrated_probability": probs[l]} for l in positives], nf, retrieved)
        assert_payload_clean(payload, u, corpus_ids, test_ids)
        msg = render_user_message(payload)
        blank = [dict(r, findings="", impression="") for r in payload["retrieved"]]
        msg_no_retrieved = render_user_message({**payload, "retrieved": blank})
        ref_f, ref_i, ref_c = vt.loc[u, "findings_clean"], vt.loc[u, "impression_clean"], str(vt.loc[u, "retrieval_text"])
        ref_sents = [norm_sentence(s) for s in sentences(ref_c) if len(norm_sentence(s).split()) >= 6]
        norm_msg = norm_sentence(msg_no_retrieved)
        in_nonret = sum(s in norm_msg for s in ref_sents)
        in_ret = bool(ref_sents) and all(s in norm_sentence(" ".join(r["findings"] + " " + r["impression"] for r in retrieved)) for s in ref_sents)
        leak_rows.append({"uid": u, "reference_sentences_in_non_retrieval_payload": in_nonret, "reference_text_fully_present_in_retrieved_reports": bool(in_ret),
                          "own_study_in_retrieved": u in {r["study_id"] for r in retrieved}})
        truth = sorted(ctx.truth[u]) if u in ctx.truth else None
        cases.append({"uid": u, "anon_id": anon(u), "image_id": val.loc[u, "frontal_image"], "classifier_positive_findings": positives,
                      "classifier_probabilities": {l: probs[l] for l in LABELS}, "no_finding_positive": bool(nf), "query": q["text"] if q else "", "query_status": q["status"] if q else "empty",
                      "retrieved": retrieved, "evidence_rows": payload["evidence_table"], "payload": payload, "user_message": msg, "user_message_sha256": hashlib.sha256(msg.encode()).hexdigest(),
                      "b0": b0_report(positives, nf), "in_clinical_subset": truth is not None, "in_r2_primary_358": u in set(ctx.primary),
                      "reference": {"findings": ref_f, "impression": ref_i, "combined": ref_c, "truth_findings": truth}})
    with open(G1 / "g1_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    pd.DataFrame(leak_rows).to_csv(G1 / "payload_leakage_check_per_case.csv", index=False)
    clin = [c for c in cases if c["in_clinical_subset"]]
    status = pd.Series([c["query_status"] for c in cases]).value_counts().to_dict()
    pops = {"validation_studies": int(len(val)), "locked_retrieval_test_studies_opened": 0,
            "definition_primary_report_generation_set": "retrieval-validation studies with a frontal image (and a classifier output) and a non-empty reference report; classifier-query empty or imperfect cases are KEPT",
            "validation_studies_with_frontal_image": len(frontal), "frontal_without_usable_reference_report": len([u for u in frontal if not has_ref(u)]),
            "primary_report_generation_set": len(cases), "clinical_finding_evaluation_subset": len(clin),
            "definition_clinical_subset": "primary set studies that also have a usable mapped IU finding set (R1 MeSH mapping)",
            "primary_set_classifier_query_status": status, "primary_set_no_retrieval_context_studies": int(sum(not c["retrieved"] for c in cases)),
            "in_clinical_subset_reference_normal": int(sum(c["reference"]["truth_findings"] == [NO_FINDING] for c in clin)),
            "r2_primary_358_studies_inside_g1_primary": int(sum(c["in_r2_primary_358"] for c in cases)),
            "paired_model_comparison_set": "decided after generation: the studies for which BOTH B0 and G1 reports exist (all studies unless an API failure is reported); B0 and G1 are always compared on identical studies",
            "studies_not_in_clinical_subset_reason": "no usable mapped IU finding set (only out-of-vocabulary or ambiguous MeSH terms)"}
    (G1 / "g1_populations.json").write_text(json.dumps(pops, indent=2), encoding="utf-8")
    leak = pd.DataFrame(leak_rows)
    lk = {"n_cases": len(leak), "payload_whitelist_enforced": True, "reference_sentences_in_non_retrieval_payload_total": int(leak.reference_sentences_in_non_retrieval_payload.sum()),
          "cases_with_reference_sentence_outside_retrieved_reports": int((leak.reference_sentences_in_non_retrieval_payload > 0).sum()),
          "cases_with_own_study_in_retrieved": int(leak.own_study_in_retrieved.sum()),
          "informational_reference_text_fully_present_in_retrieved_reports_templated_duplicates": int(leak.reference_text_fully_present_in_retrieved_reports.sum()),
          "note": "the last count is generic templated reporting by OTHER studies that happens to equal the reference wording; it is legitimate retrieval, not leakage",
          "prompt_sha256": prompt_hash(), "truth_findings_in_payload": False}
    (G1 / "payload_leakage_check.json").write_text(json.dumps(lk, indent=2), encoding="utf-8")
    integrity["top5_integrity_pass"] = integrity["top5_identical_to_r2"] == integrity["r2_primary_studies_checked"]
    integrity["retrieval_config"] = {"sha256_of_r2_candidate_config": sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json"), "k": TOPK}
    (G1 / "retrieval_top5_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({"populations": pops, "leakage": lk, "top5_integrity": {k: v for k, v in integrity.items() if k != "mismatches"}}, indent=1))
    ctx.save_cache()
    return 0 if integrity["top5_integrity_pass"] and lk["cases_with_reference_sentence_outside_retrieved_reports"] == 0 and lk["cases_with_own_study_in_retrieved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
