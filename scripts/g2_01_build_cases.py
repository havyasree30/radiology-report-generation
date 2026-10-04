"""G2 step 1: build the generator inputs for the same 547 studies as G1 from the FROZEN classifier outputs and the FROZEN G1/R2 Top-5 (read from
the stored G1 cases; nothing is re-retrieved or re-classified), and run leakage checks on the Agent 1 messages. The case file contains NO reference
report, reference finding or truth. No model call.

    .venv\Scripts\python.exe -m scripts.g2_01_build_cases
"""

from __future__ import annotations

import hashlib
import json

from src.classification.final_test import sha256_file, verify_freeze
from src.generation.g2_agents import build_agent1_message
from src.generation.g2_pipeline import case_inputs
from src.generation.metrics import norm_sentence, sentences
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"
EXP = PROJECT_ROOT / "results/classification/experiments"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"


def main() -> int:
    G2.mkdir(parents=True, exist_ok=True)
    verify_freeze(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", PROJECT_ROOT)
    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    split = json.loads((PROJECT_ROOT / "results/retrieval/experiments/r1_baseline/retrieval_split.json").read_text(encoding="utf-8"))
    test_ids = set(split["study_ids"]["locked_test"])
    out, chk = [], {"n": 0, "n_with_retrieval": 0, "n_empty_query": 0, "reference_sentences_outside_retrieved_text": 0, "study_ids_in_agent1_message": 0, "locked_test_ids_in_case_file": 0, "reference_keys_in_case_file": 0}
    for c in g1_cases:
        chk["n"] += 1
        retrieved = [{"rank": r["rank"], "findings": r["findings"], "impression": r["impression"]} for r in c["retrieved"]]
        case = {"uid": c["uid"], "anon_id": c["anon_id"], "image_id": c["image_id"], "classifier_positive_findings": c["classifier_positive_findings"], "classifier_probabilities": c["classifier_probabilities"],
                "no_finding_positive": c["no_finding_positive"], "query_status": c["query_status"], "retrieved_study_ids": [r["study_id"] for r in c["retrieved"]], "retrieved": retrieved,
                "in_clinical_subset": c["in_clinical_subset"]}
        positives, nf, retr = case_inputs(case)
        msg = build_agent1_message(positives, nf, retr)
        case["agent1_message_sha256"] = hashlib.sha256(msg.encode()).hexdigest()
        chk["n_with_retrieval"] += int(bool(retrieved))
        chk["n_empty_query"] += int(not retrieved)
        chk["study_ids_in_agent1_message"] += sum(str(i) in msg.replace("\n", " ").split() for i in case["retrieved_study_ids"] + [c["uid"]])
        ctx = norm_sentence(" ".join(r["findings"] + " " + r["impression"] for r in retrieved))
        nm = norm_sentence(msg)
        chk["reference_sentences_outside_retrieved_text"] += sum(s in nm and s not in ctx for s in [norm_sentence(x) for x in sentences(c["reference"]["combined"]) if len(norm_sentence(x).split()) >= 6])
        chk["locked_test_ids_in_case_file"] += int(c["uid"] in test_ids) + sum(i in test_ids for i in case["retrieved_study_ids"])
        chk["reference_keys_in_case_file"] += int(any(k in case for k in ("reference", "truth", "truth_findings")))
        out.append(case)
    with open(G2 / "g2_cases.jsonl", "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    same = [c["uid"] for c in out] == [c["uid"] for c in g1_cases]
    rep = {**chk, "same_547_studies_and_order_as_g1": same, "sha256_g1_cases_source": sha256_file(G1 / "g1_cases.jsonl"), "sha256_r2_candidate_config": sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json"),
           "retrieved_ids_equal_g1_top5": all(c["retrieved_study_ids"] == [r["study_id"] for r in g["retrieved"]] for c, g in zip(out, g1_cases)),
           "note": "retrieval is read from the stored G1 cases (frozen R2 Top-5); the case file holds no reference information"}
    (G2 / "g2_input_checks.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if same and rep["retrieved_ids_equal_g1_top5"] and chk["reference_sentences_outside_retrieved_text"] == 0 and chk["study_ids_in_agent1_message"] == 0 and chk["locked_test_ids_in_case_file"] == 0 and chk["reference_keys_in_case_file"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
