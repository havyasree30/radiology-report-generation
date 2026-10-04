"""G1F step 3: write FINAL_SYSTEM_CONFIG.json (immutable identifiers and settings, with a configuration hash) and FINAL_SYSTEM_FREEZE.md, but ONLY
if the pre-freeze implementation and integrity checks pass. No model is contacted other than the read-only version/digest check inside the integrity module;
no locked-test file is read: the locked retrieval-test split is identified by the hash of its study-id list (ids only, as in R1).

    .venv\\Scripts\\python.exe -m scripts.g1f_03_freeze <snapshot.sha256>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

from scripts.g1f_05_integrity import CONFIG_PATH, COMMITS, EXP, G1, G1F, R1, R2, config_hash, run_checks
from src.classification.final_test import sha256_file
from src.system import guard
from src.utils.config import PROJECT_ROOT

COMMIT_PREFIXES = {"C5": "Add C5", "C6": "Add C6", "R1": "Add R1", "R2": "Add R2"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def commit_of(prefix: str) -> str:
    for line in git("log", "--format=%H%x09%s").splitlines():
        h, s = line.split("\t", 1)
        if s.startswith(prefix):
            return h
    raise SystemExit(f"commit not found: {prefix}")


def main(snapshot: str) -> int:
    pre = run_checks(snapshot, require_config=False)
    if pre["status"] != "PASS":
        print(json.dumps({"STOP": "pre-freeze checks failed", "report": pre}, indent=1, default=str), file=sys.stderr)
        return 1
    man = json.loads((EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    r2 = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    gf = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    gp = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    pops = json.loads((G1 / "g1_evaluation_populations.json").read_text(encoding="utf-8"))
    emb = r2["embedding_model"]
    ids_hash = lambda ids: hashlib.sha256(json.dumps(sorted(str(i) for i in ids)).encode()).hexdigest()  # noqa: E731
    settings = {
        "classification": {
            "architecture": man["model_architecture"], "checkpoint": man["checkpoint"], "checkpoint_sha256": man["checkpoint_sha256"], "checkpoint_epoch": man["checkpoint_epoch"], "loss": man["loss"], "loss_config": man["loss_config"],
            "label_order": man["label_order"], "preprocessing": man["preprocessing"], "view": man["view"], "inference_policy": man["inference_policy"], "test_time_augmentation": man["test_time_augmentation"], "ensembling": man["ensembling"],
            "decision_rule": man["binary_rule"], "f1_thresholds_raw_score": man["f1_thresholds_raw_score"], "operating_policy_file": "results/classification/experiments/c4_operating_policy/final_operating_policy.json",
            "operating_policy_file_sha256": sha256_file(EXP / "c4_operating_policy/final_operating_policy.json"), "no_finding_rule": man["no_finding_rule"], "no_finding_rule_abnormal_labels": man["no_finding_rule_abnormal_labels"],
            "support_devices_suppresses_no_finding": man["support_devices_suppresses_no_finding"],
            "calibration": {"method": man["calibration"]["method"], "version": man["calibration"]["version"], "fitted_on": man["calibration"]["fitted_on"], "file": "results/classification/experiments/c5_calibration/final_calibrators.json",
                            "file_sha256": sha256_file(EXP / "c5_calibration/final_calibrators.json"), "parameters": {k: {"a": v["a"], "b": v["b"]} for k, v in man["calibration"]["parameters"].items()}},
            "c6_freeze_manifest": "results/classification/experiments/c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", "c6_freeze_manifest_sha256": sha256_file(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json")},
        "retrieval": {
            "r2_candidate_config": "results/retrieval/experiments/r2_optimization/R2_RETRIEVAL_CANDIDATE_CONFIG.json", "r2_candidate_config_sha256": sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json"),
            "embedding_model": emb["identifier"], "embedding_snapshot_revision": emb["snapshot_revision"], "embedding_dimension": emb["dimension"], "embedding_normalised": emb["normalised"], "embedding_fine_tuned": emb["fine_tuned"], "embedding_weights_sha256": emb["weights_sha256"],
            "query_construction": {k: v for k, v in r2["query_construction"].items() if k not in ("clinical_expansion_mapping",)}, "bm25": r2["lexical"],
            "fusion": {"method": "Reciprocal Rank Fusion", "rrf_k": 60, "candidate_depth_per_ranker": 100, "rankers": ["dense_minilm", "bm25"]}, "mmr": False, "phrase_expansion": False, "confidence_weighting": False, "top_k": 5,
            "corpus": {"ids_file": "results/retrieval/experiments/r1_baseline/corpus_study_ids.json", "ids_file_sha256": sha256_file(R1 / "corpus_study_ids.json"), "n_reports": split["sizes"]["reference_corpus_studies"],
                       "corpus_uids_file": "data/splits/iu_xray/retrieval_corpus_uids.csv", "corpus_uids_file_sha256": sha256_file(PROJECT_ROOT / "data/splits/iu_xray/retrieval_corpus_uids.csv"), "rule": "IU training reports only (assert_train_only)"},
            "split_identifiers": {"retrieval_split_file": "results/retrieval/experiments/r1_baseline/retrieval_split.json", "retrieval_split_file_sha256": sha256_file(R1 / "retrieval_split.json"), "iu_report_splits_sha256": sha256_file(PROJECT_ROOT / "data/splits/iu_xray/iu_report_splits.csv"),
                                  "sizes": split["sizes"], "seed": 42}},
        "generation": {
            "selected_architecture": "G1 Single-Agent RAG", "not_selected": "G2 Structured Multi-Agent RAG (negative result; see FINAL_SYSTEM_SELECTION.md)", "model_tag": gf["model"]["model_tag"], "model_digest": gf["model"]["digest"],
            "quantization": gf["model"]["quantization_level"], "parameter_size": gf["model"]["parameter_size"], "ollama_version": gf["ollama_version"], "endpoint": gf["endpoint"], "generation_options": gf["generation_options"],
            "context_size_num_ctx": gf["context_size_num_ctx"], "max_new_tokens": gf["max_new_tokens"], "tools": "none", "web_access": False, "external_api": False, "g1_prompt_version": gp["prompt_version"], "g1_prompt_sha256": gp["prompt_sha256"],
            "g1_prompt_file": "results/report_generation/experiments/g1_single_agent/g1_prompt_frozen.json", "g1_prompt_file_sha256": sha256_file(G1 / "g1_prompt_frozen.json"), "generator_freeze_file_sha256": sha256_file(G1 / "GENERATOR_FREEZE.json"),
            "parser": "src/generation/parse.py", "parser_sha256": sha256_file(PROJECT_ROOT / "src/generation/parse.py"), "stored_validation_reports_sha256": sha256_file(G1 / "g1_reports_raw.jsonl")},
        "guard": {
            "implementation": "src/system/guard.py", "guard_module_sha256": sha256_file(PROJECT_ROOT / "src/system/guard.py"), "states": list(guard.STATES), "pathology_labels": list(guard.PATHOLOGY_LABELS), "support_devices_is_pathology": False,
            "path_A_normal": "No Finding positive AND no pathology label positive (Support Devices is not a pathology label, as in the frozen C4 rule): frozen G1 normal behaviour (query 'no acute abnormality', G1 prompt, G1 report)",
            "path_B_abnormal": "any positive label that does not make the study an explicit normal one: frozen G1 Single-Agent RAG report (retrieval query = top-3 positive findings by calibrated probability)",
            "path_C_indeterminate": "no label positive AND No Finding not positive (empty classifier output): no retrieval, no language-model call, not declared normal",
            "indeterminate_findings": guard.INDETERMINATE_FINDINGS, "indeterminate_impression": guard.INDETERMINATE_IMPRESSION, "indeterminate_report_text": guard.INDETERMINATE_REPORT_TEXT, "ui_message_indeterminate": guard.UI_MESSAGE_INDETERMINATE,
            "ui_must_not_show_for_indeterminate": list(guard.UI_FORBIDDEN_FOR_INDETERMINATE) + ["a fabricated preliminary diagnosis"], "state_field": "system_interpretation_state", "state_source": "deterministic routing of the classifier output; never inferred from generated prose",
            "validation_counts": {"A_normal": 269, "B_abnormal": 229, "C_indeterminate": 49}, "support_devices_decision": "chosen by the project owner before the freeze: empty output = no positive label at all"},
        "evaluation": {
            "finding_extraction": {"lexicon": "configs/iu_concept_lexicon.yaml", "lexicon_sha256": sha256_file(PROJECT_ROOT / "configs/iu_concept_lexicon.yaml"), "changes": "configs/generation/finding_extraction_changes.yaml",
                                   "changes_sha256": sha256_file(PROJECT_ROOT / "configs/generation/finding_extraction_changes.yaml"), "stated_definition": "affirmative (definite or hedged) mention", "reference": "MeSH-mapped IU findings (R1)"},
            "validation_populations_file_sha256": sha256_file(G1 / "g1_evaluation_populations.json"), "primary_set_n": pops["primary_report_generation_set"], "clinical_subset_n": pops["clinical_finding_evaluation_subset"],
            "validation_split": {"iu_validation_studies": split["sizes"]["validation_studies"], "validation_study_ids_sha256": ids_hash(split["study_ids"]["validation"]), "chexpert_patient_splits_sha256": sha256_file(PROJECT_ROOT / "data/splits/chexpert/chexpert_patient_splits.csv")},
            "locked_test_split": {"identity": "IU report-level locked retrieval/end-to-end test split", "n_studies": split["sizes"]["locked_test_studies"], "study_ids_sha256": ids_hash(split["study_ids"]["locked_test"]), "opened": False,
                                  "classifier_test_note": "the C6 one-shot classifier test is documented in C6 and is not part of this freeze; no retrieval or end-to-end test result exists"},
            "bootstrap": {"resamples": 1000, "seed": 42, "unit": "study"}},
    }
    cfg_hash = config_hash(settings)
    freeze_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commits = {"C1-C5": man["commits"], "C6": commit_of(COMMIT_PREFIXES["C6"]), "R1": commit_of(COMMIT_PREFIXES["R1"]), "R2": commit_of(COMMIT_PREFIXES["R2"]), **COMMITS, "base_head_at_freeze": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
               "note": "the G1F files are uncommitted at freeze time; the base head is the G2 commit"}
    doc = {"frozen_utc": freeze_utc, "config_sha256": cfg_hash, "config_hash_definition": "SHA-256 of the canonical JSON (sorted keys, compact separators) of the `settings` object", "settings": settings, "commits": commits, "locked_test_results_included": False}
    CONFIG_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    freeze = f"""# Final system freeze (G1F)

**Freeze timestamp (UTC):** {freeze_utc}

> Classification, calibration, operating thresholds, retrieval, query construction, Top-K, report-generation architecture, generator model, generation prompt and deterministic empty-output handling are frozen before opening the locked test set.

> No parameter, prompt, threshold, routing rule or model selection may be changed after inspecting the locked-test outputs.

## Identifiers

| Item | Value |
|---|---|
| Final configuration | `results/report_generation/experiments/g1f_final_system/FINAL_SYSTEM_CONFIG.json` |
| Configuration hash (SHA-256 of `settings`) | `{cfg_hash}` |
| G1 prompt hash | `{gp['prompt_sha256']}` |
| G1F guard hash (`src/system/guard.py`) | `{settings['guard']['guard_module_sha256']}` |
| Classifier checkpoint SHA-256 | `{man['checkpoint_sha256']}` |
| Calibration file SHA-256 | `{settings['classification']['calibration']['file_sha256']}` |
| R2 retrieval configuration SHA-256 | `{settings['retrieval']['r2_candidate_config_sha256']}` |
| Generator | `{gf['model']['model_tag']}` digest `{gf['model']['digest']}`, {gf['model']['quantization_level']}, Ollama {gf['ollama_version']} |
| Locked retrieval-test study-id list SHA-256 | `{settings['evaluation']['locked_test_split']['study_ids_sha256']}` ({settings['evaluation']['locked_test_split']['n_studies']} studies, opened: no) |

## Commits (recorded, not pushed)

C1: `{man['commits']['C1']}`, C2 results: `{man['commits']['C2_results']}`, C3: `{man['commits']['C3']}`, C4: `{man['commits']['C4']}`, C5: `{man['commits']['C5']}`, C6: `{commits['C6']}`, R1: `{commits['R1']}`, R2: `{commits['R2']}`, G1: `{COMMITS['G1']}`, G1A: `{COMMITS['G1A']}`, G1B: `{COMMITS['G1B']}`, G2: `{COMMITS['G2']}` (base head at the freeze, branch `{commits['branch']}`). The G1F files themselves are uncommitted at the time of the freeze.

## Selected architecture

G1 Single-Agent RAG (G2 is a negative-result experiment and is not part of the final system; see `FINAL_SYSTEM_SELECTION.md`).

## Frozen end-to-end pipeline

```
Chest X-ray
  -> preprocessing
  -> frozen DenseNet-121
  -> calibrated 14-label outputs
  -> C4 binary decisions
  -> deterministic routing guard (src/system/guard.py)

  Explicit No Finding (Path A, state normal)
    -> frozen G1 normal path

  >= 1 pathology positive (Path B, state abnormal)
    -> Top-3 classifier query findings
    -> Hybrid Dense + BM25 RRF
    -> Top-5 IU evidence reports
    -> frozen G1 MedGemma Single-Agent RAG
    -> preliminary Findings + Impression

  Empty classifier output (Path C, state indeterminate)
    -> deterministic INDETERMINATE output
    -> no retrieval
    -> no LLM generation
```

Routing details: Path A requires No Finding positive and no pathology label positive; Support Devices is not a pathology label, as in the frozen C4 rule (a device does not suppress No Finding). Path C requires that no label at all is positive and that No Finding is not positive. Path B is every other study, including a study whose only positive label is a support device without No Finding. The exact indeterminate output is:

`FINDINGS: {guard.INDETERMINATE_FINDINGS}`

`IMPRESSION: {guard.INDETERMINATE_IMPRESSION}`

## Intended application behaviour for indeterminate studies

The interface must show **{guard.UI_MESSAGE_INDETERMINATE}** and must not show "No abnormality detected", "Normal X-ray" or any fabricated preliminary diagnosis for an indeterminate study. The displayed state comes from `system_interpretation_state` (normal, abnormal or indeterminate), never from the generated prose. The application is a research prototype, not an autonomous diagnostic system, and has not been built yet.

## Not included

No locked-test result of any kind. The retrieval and end-to-end test split is unopened.
"""
    (G1F / "FINAL_SYSTEM_FREEZE.md").write_text(freeze, encoding="utf-8")
    print(json.dumps({"status": "frozen", "config_sha256": cfg_hash, "frozen_utc": freeze_utc, "guard_sha256": settings["guard"]["guard_module_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
