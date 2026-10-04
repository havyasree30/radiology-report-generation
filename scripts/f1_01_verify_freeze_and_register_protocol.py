"""F1 step 1: (a) verify every freeze identifier of the final system BEFORE any locked-test data is touched (STOP on any difference);
(b) write and hash the F1 locked-test protocol (JSON + Markdown). No test report content, label, image or result is read: the locked test split
is identified only by the hash of its study-id list (ids only, as in R1/G1F).

    .venv\\Scripts\\python.exe -m scripts.f1_01_verify_freeze_and_register_protocol <snapshot.sha256>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

from scripts.g1f_05_integrity import run_checks
from src.classification.final_test import sha256_file
from src.utils.config import PROJECT_ROOT

G1F = PROJECT_ROOT / "results/report_generation/experiments/g1f_final_system"
F1 = PROJECT_ROOT / "results/final_test/f1_locked_end_to_end"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
EXPECTED = {"config_sha256": "cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5", "g1_prompt_sha256": "476b05326f85652b0cdb2faf79a4868a6e839660eb43e34d46cdae625e4b5688",
            "guard_sha256": "1f0b340cc71cf261de84f4b19f290c99568a69760dc757d57c62d2c68c9397ee"}
FROZEN_CODE = ["src/system/guard.py", "src/generation/prompts.py", "src/generation/client.py", "src/generation/parse.py", "src/generation/extraction.py", "src/generation/metrics.py", "src/generation/study_eval.py",
               "src/generation/evidence.py", "src/retrieval/evaluation.py", "src/retrieval/metrics.py", "src/retrieval/queries.py", "src/retrieval/fusion.py", "src/retrieval/r2_context.py", "src/classification/final_test.py",
               "src/classification/metrics.py", "src/classification/calibration.py", "scripts/c6_03_evaluate.py", "scripts/r2_02_experiments.py", "scripts/g1_01_build_cases.py", "scripts/g1f_02_evaluate.py"]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    F1.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((G1F / "FINAL_SYSTEM_CONFIG.json").read_text(encoding="utf-8"))
    s = cfg["settings"]
    cfg_hash = hashlib.sha256(json.dumps(s, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    checks = {
        "config_hash_equals_expected_and_stored": cfg_hash == cfg["config_sha256"] == EXPECTED["config_sha256"],
        "g1_prompt_hash": s["generation"]["g1_prompt_sha256"] == EXPECTED["g1_prompt_sha256"],
        "guard_hash_in_config_and_on_disk": s["guard"]["guard_module_sha256"] == EXPECTED["guard_sha256"] == sha256_file(PROJECT_ROOT / "src/system/guard.py"),
        "classification_checkpoint_loss_thresholds_rule_calibration": (s["classification"]["loss"] == "sqrt_weighted_bce" and "densenet121" in s["classification"]["architecture"].lower() and len(s["classification"]["f1_thresholds_raw_score"]) == 14
                                                                       and s["classification"]["no_finding_rule"] == "suppress_if_any_abnormal_positive" and s["classification"]["calibration"]["method"].startswith("platt")),
        "retrieval_top3_minilm_bm25_rrf60_depth100_k5_no_mmr_no_expansion_no_weighting": (s["retrieval"]["query_construction"]["top_n_findings"] == 3 and s["retrieval"]["embedding_model"].endswith("all-MiniLM-L6-v2") and s["retrieval"]["bm25"]["k1"] == 1.5
                                                                                           and s["retrieval"]["fusion"]["rrf_k"] == 60 and s["retrieval"]["fusion"]["candidate_depth_per_ranker"] == 100 and s["retrieval"]["top_k"] == 5
                                                                                           and s["retrieval"]["mmr"] is False and s["retrieval"]["phrase_expansion"] is False and s["retrieval"]["confidence_weighting"] is False),
        "generation_g1_single_agent_medgemma_digest_decoding": (s["generation"]["selected_architecture"] == "G1 Single-Agent RAG" and s["generation"]["model_tag"] == "medgemma1.5:4b"
                                                                and s["generation"]["model_digest"] == "433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c" and s["generation"]["generation_options"]["temperature"] == 0.0
                                                                and s["generation"]["generation_options"]["top_k"] == 1 and s["generation"]["generation_options"]["seed"] == 42 and s["generation"]["external_api"] is False),
        "guard_paths_A_B_C_and_exact_indeterminate_text": (s["guard"]["states"] == ["normal", "abnormal", "indeterminate"] and s["guard"]["indeterminate_findings"] == "Model output is indeterminate for this study."
                                                           and s["guard"]["indeterminate_impression"] == "Automated preliminary interpretation could not be established. Radiologist review is required."),
        "locked_test_unopened_and_no_test_results_in_config": s["evaluation"]["locked_test_split"]["opened"] is False and cfg["locked_test_results_included"] is False}
    pre = run_checks(snapshot, require_config=True)            # classifier / calibration / thresholds / retrieval / generator (live digest) / earlier artifacts / code unchanged
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    ids = sorted(str(i) for i in split["study_ids"]["locked_test"])
    ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    checks["locked_test_id_list_hash_equals_frozen_hash_and_has_578_ids"] = ids_hash == s["evaluation"]["locked_test_split"]["study_ids_sha256"] and len(ids) == 578 == s["evaluation"]["locked_test_split"]["n_studies"]
    checks["g1f_integrity_run_checks_pass"] = pre["status"] == "PASS"
    verdict = {"all_identifiers_verified": all(checks.values()), "checks": checks, "g1f_integrity_status": pre["status"]}
    (F1 / "f1_freeze_verification.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if not all(checks.values()):
        print(json.dumps({"STOP": "freeze identifier differs; the locked test split is NOT opened", "failed": [k for k, v in checks.items() if not v]}, indent=1), file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proto = {
        "protocol_version": "f1-v1", "created_utc": now, "git_branch": git("branch", "--show-current"), "git_commit_head": git("rev-parse", "HEAD"),
        "final_system_config_sha256": cfg_hash, "g1_prompt_sha256": EXPECTED["g1_prompt_sha256"], "guard_module_sha256": EXPECTED["guard_sha256"],
        "locked_test": {"n_study_ids": len(ids), "study_ids_sha256": ids_hash, "opened_before_this_protocol": False, "split_file": "results/retrieval/experiments/r1_baseline/retrieval_split.json",
                        "ids_only_read_before_protocol": True, "content_read_before_protocol": "none (no report text, MeSH term, label, image, classifier output, retrieval result or metric)"},
        "frozen_system": {"summary": "frozen DenseNet-121 (C2-B sqrt-weighted BCE, C4 F1 thresholds, C4 No Finding rule, C5 Platt) -> deterministic routing guard (G1F) -> Top-3 classifier query -> MiniLM dense + BM25 RRF (k=60, depth 100) -> Top-5 -> frozen G1 MedGemma 1.5 4B Single-Agent RAG; empty classifier output -> deterministic INDETERMINATE",
                          "configuration_file": "results/report_generation/experiments/g1f_final_system/FINAL_SYSTEM_CONFIG.json", "freeze_document": "results/report_generation/experiments/g1f_final_system/FINAL_SYSTEM_FREEZE.md",
                          "frozen_code_sha256": {p: sha256_file(PROJECT_ROOT / p) for p in FROZEN_CODE}},
        "evaluation_populations": {
            "P0_locked_studies": "all 578 locked test studies (report uid is the unit)",
            "P1_with_frontal_image": "studies with at least one Frontal image; the frontal image is the first Frontal file in sorted filename order (R1 rule)",
            "P2_classified": "P1 studies whose image passes the frozen ImageValidator and is classified; studies failing the validator are listed with the reason, never silently dropped",
            "P3_primary_end_to_end_set": "P2 studies with a non-empty cleaned reference report (build_retrieval_text) = the G1 'primary report-generation set'; routing counts are reported on P3 and on P2",
            "P4_clinical_finding_subset": "P3 studies with a usable mapped truth (has_eval_set under the frozen r1-v1 MeSH mapping); all finding-level, report-state and abstention metrics use P4",
            "classification_population": "P2 studies with a usable mapped truth (R1 'eligible and classified'); label = class in the mapped truth set, No Finding = truth is exactly {No Finding}; unmapped MeSH terms are NOT verified negatives (precision is a lower bound)",
            "retrieval_population": "P2 studies with a usable mapped truth and a non-empty classifier query (R1/R2 'primary set')", "inclusion_rules_changed_after_seeing_test": False,
            "reported_without_silent_exclusion": ["total locked studies", "with frontal image", "classified", "with usable reference report", "with usable mapped truth (P4)", "routed to Path A / B / C (on P3 and on P2)", "excluded studies with reasons"]},
        "classification": {"definitions": "identical to C6: scripts/c6_03_evaluate.evaluate (per-class AUROC/AUPRC/precision/recall/specificity/F1/balanced accuracy, macro and micro averages over defined classes, Brier, log loss, adaptive-bin ECE with 10 bins)",
                           "metrics": ["macro AUROC", "micro AUROC", "macro AUPRC", "micro AUPRC", "macro precision", "macro recall", "macro specificity", "macro F1", "macro balanced accuracy", "Brier score", "log loss", "ECE"],
                           "per_class": ["AUROC", "AUPRC", "precision", "recall", "F1", "support"], "undefined_classes": "reported as undefined (no positives or no negatives), never replaced by 0",
                           "saved": ["raw sigmoid outputs", "calibrated probabilities", "binary predictions after the No Finding rule", "No Finding rule result", "routing state"], "preprocessing_and_inference": "frozen: fp32_strict, no TTA, no ensembling, image size 224"},
        "routing": {"rule": "src/system/guard.route (frozen); state never derived from prose", "states": ["normal", "abnormal", "indeterminate"]},
        "retrieval": {"configuration": "frozen R2 (hybrid RRF, Top-5 for generation, Top-10 retained for the K analysis)", "K": [1, 3, 5, 10], "primary_K": 5, "metrics": ["Jaccard (truth)", "nDCG", "union finding coverage", "Hit and reciprocal rank (exact finding-set match)", "duplicate-text rate", "zero-overlap rate"],
                      "definitions": "src/retrieval/evaluation.evaluate_ranking (unchanged)", "oracle_diagnostic": "oracle query (the study's true findings) through the same hybrid pipeline on the same population; classifier-query minus oracle gap at K=5; diagnostic only, may not alter the system",
                      "path_C": "no retrieval", "path_A_normal_query": "no acute abnormality (frozen R2 phrase)"},
        "generation": {"system": "frozen G1 (prompt hash above); Paths A and B each generated exactly once; Path C returns the frozen indeterminate text without any call", "model_options": s["generation"]["generation_options"], "cache": "every response cached per study with the request hash and model digest",
                       "retry_policy": "generate_single: up to 5 attempts for transient errors with exponential back-off (2^attempt s, max 60 s); malformed requests are not retried",
                       "persistent_failure_handling": "recorded explicitly and listed; never replaced by fabricated or hand-edited text; report metrics are computed on studies with a generated report and the failures are reported as a count with ids; no re-prompting"},
        "report_metrics": {"set": "P4", "metrics": ["finding precision", "finding recall", "finding F1 (micro)", "macro finding F1", "hallucination rate", "omission rate", "classifier FP propagation", "classifier TP retention", "ROUGE-L", "BLEU-4", "METEOR exact-match variant (not standard METEOR)"],
                           "text_metric_set": "P3 (all studies with a final text, including the indeterminate message)", "definitions": "src/generation/study_eval.study_row (unchanged); finding extractor = frozen Phase 1 lexicon + finding_extraction_changes.yaml; 'stated' = affirmative or hedged mention"},
        "three_state": {"set": "P4", "outputs": ["state distribution (P3 and P4)", "routing by reference state", "normal and abnormal recall among decided studies", "decision coverage = decided / eligible (P4)"], "indeterminate_not_forced_into_normal_or_abnormal": True},
        "abstention": {"descriptive": True, "outputs": ["number and share", "reference-normal and reference-abnormal fractions", "findings in reference-abnormal abstentions", "rare-finding representation (Fisher exact, descriptive)"]},
        "routing_prose_consistency": {"definition": "among studies routed abnormal, the share whose final report prose is read by the frozen extractor as normal / abnormal / indeterminate; abnormal-routing -> normal-prose rate; reports are not repaired; the validation rate (31.0%) is a descriptive comparator and not a target or threshold"},
        "provenance": {"classes": ["A classifier only", "B retrieval only", "C both", "D neither"], "definition": "src/generation/metrics.provenance with the study's own Top-5 mapped findings as the retrieval-supported set", "reported": ["counts", "reference support within each class", "retrieval-only supported vs unsupported"]},
        "copying": {"definition": "src/generation/metrics.copy_stats (min 6 words) against the study's own Top-5 text + repeated-sentence rate; unchanged", "reported": ["sentence copying", "whole-report copying", "repeated-sentence rate"]},
        "rare_findings": {"findings": ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"], "reported": ["test support", "precision", "recall", "F1", "TP retention"], "caution": "no strong claim when counts are very small"},
        "bootstrap": {"resamples": 1000, "seed": 42, "unit": "study (IU X-Ray has no patient identifier; one frontal image per study); patient-level resampling is not possible for this population", "interval": "95% percentile", "significance_testing": "none beyond descriptive intervals and the descriptive Fisher test above"},
        "validation_to_test": {"method": "the same analysis code is run on the validation partition (dry run; stored G1/G1F/R1/R2 results must be reproduced) and test-minus-validation absolute differences are reported", "classification_comparator": "IU validation (same pipeline and labels); CheXpert C6 validation/test values are listed as context only",
                               "use": "descriptive; differences are not used for tuning"},
        "failure_taxonomy_frozen": {"unit": "study (P4); a study may fall in several categories; 'no failure' if none", "categories": {
            "1_classifier_error": "classifier non-No-Finding positive set differs from the mapped truth findings",
            "2_normal_abnormal_classifier_mismatch": "classifier state (normal/abnormal) differs from the reference state (indeterminate studies are category 8)",
            "3_retrieval_query_mismatch": "studies with retrieved context whose Top-5 union contains none of the reference findings (union coverage@5 = 0)",
            "4_retrieval_only_unsupported_finding": "the report states a finding with provenance 'retrieval only' that is not in the reference",
            "5_report_generator_omission": "the report omits at least one reference finding",
            "6_report_generator_hallucination": "the report states at least one finding that is not in the reference",
            "7_abnormal_routing_normal_prose": "routed abnormal but the final prose is read as normal",
            "8_indeterminate_abstention": "routed indeterminate",
            "9_corpus_limitation": "no corpus report has exactly the reference finding set (exact match impossible)",
            "10_reference_label_limitation": "the study has unmapped or uncertain MeSH terms (reference possibly incomplete)"}, "new_taxonomy_after_inspection": "not allowed except clearly marked exploratory"},
        "figures_planned": ["1 per-class classifier AUROC/AUPRC", "2 calibration / reliability", "3 retrieval K analysis", "4 validation vs test classification metrics", "5 report precision/recall/F1", "6 hallucination, omission, FP propagation, TP retention", "7 three-state results", "8 end-to-end failure-category summary"],
        "tables_planned": ["1 locked-test population", "2 final classifier metrics", "3 per-class classifier metrics", "4 final retrieval metrics", "5 final report-generation metrics", "6 three-state system metrics", "7 rare-finding analysis", "8 validation-to-test comparison", "9 failure analysis", "10 runtime/compute"],
        "runtime": ["classifier inference runtime", "retrieval runtime", "G1 generation runtime (per call and total)", "total runtime", "Ollama/model metadata", "sampled peak GPU memory (nvidia-smi)"],
        "analysis_code": {"development": "all F1 analysis code is developed and debugged on the VALIDATION partition only (dry run) and must reproduce the stored validation results before the test is opened; the SHA-256 of every F1 script is recorded in f1_code_freeze.json immediately before the first locked-test access",
                          "bug_rule": "a code bug found after the test is opened is documented with the corrected code hash and may be fixed only if it invalidates a metric; values are never edited and no scientific choice is changed"},
        "no_post_test_tuning": "No model, threshold, calibration, query, retrieval setting, Top-K, generator, prompt, routing rule, verifier or postprocessor may be changed after any locked-test reference content or metric is inspected; no study may be excluded because of poor performance; any change would require a new independent test set."}
    pj = F1 / "F1_LOCKED_TEST_PROTOCOL.json"
    pj.write_text(json.dumps(proto, indent=2, ensure_ascii=False), encoding="utf-8")
    pj_hash = sha256_file(pj)
    lines = ["# F1 locked end-to-end test protocol", "", f"_Registered {now} UTC on branch `{proto['git_branch']}` at commit `{proto['git_commit_head']}`; written BEFORE any locked-test report content, image, classifier output or result was read. The JSON version is authoritative; its SHA-256 is `{pj_hash}`._", "",
             f"- Final system configuration SHA-256: `{cfg_hash}`", f"- G1 prompt SHA-256: `{EXPECTED['g1_prompt_sha256']}`", f"- Guard SHA-256: `{EXPECTED['guard_sha256']}`",
             f"- Locked-test study-ID list SHA-256: `{ids_hash}` ({len(ids)} ids; opened before this protocol: no)", "", "## Statement", "", f"> {proto['no_post_test_tuning']}", ""]
    for k in ("evaluation_populations", "classification", "routing", "retrieval", "generation", "report_metrics", "three_state", "abstention", "routing_prose_consistency", "provenance", "copying", "rare_findings", "bootstrap", "validation_to_test", "analysis_code"):
        lines += [f"## {k.replace('_', ' ').capitalize()}", ""]
        for kk, vv in proto[k].items():
            lines.append(f"- **{kk.replace('_', ' ')}**: " + (json.dumps(vv, ensure_ascii=False) if not isinstance(vv, str) else vv))
        lines.append("")
    lines += ["## Frozen failure taxonomy", ""] + [f"- **{k}**: {v}" for k, v in proto["failure_taxonomy_frozen"]["categories"].items()] + ["", "## Planned figures", ""] + [f"- {x}" for x in proto["figures_planned"]] + ["", "## Planned tables", ""] + [f"- {x}" for x in proto["tables_planned"]] + ["", "## Runtime", ""] + [f"- {x}" for x in proto["runtime"]] + [""]
    pm = F1 / "F1_LOCKED_TEST_PROTOCOL.md"
    pm.write_text("\n".join(lines), encoding="utf-8")
    (F1 / "F1_LOCKED_TEST_PROTOCOL.sha256").write_text(f"{pj_hash} *F1_LOCKED_TEST_PROTOCOL.json\n{sha256_file(pm)} *F1_LOCKED_TEST_PROTOCOL.md\n", encoding="utf-8")
    print(json.dumps({"freeze_verified": True, "protocol_json_sha256": pj_hash, "protocol_md_sha256": sha256_file(pm), "created_utc": now, "locked_test_ids_sha256": ids_hash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
