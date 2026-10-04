"""G1A step 1: build the G1A generator inputs (classifier output ONLY) for the exact same 547 studies as G1, and verify that the
classifier block is byte-identical to G1's and that no retrieval content, reference or truth can reach the generator. No model call.

    .venv\\Scripts\\python.exe -m scripts.g1a_01_build_cases
"""

from __future__ import annotations

import hashlib
import json
import re

from src.generation.prompts_g1a import (SYSTEM_PROMPT, assert_payload_clean_g1a, build_payload_g1a, classifier_block, prompt_diff_vs_g1, prompt_hash, render_user_message_g1a,
                                        user_message_diff_vs_g1)
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
FORBIDDEN_IN_MESSAGE = [r"retriev", r"study \d+", r"dense rank", r"BM25", r"fusion score", r"Retrieved support", r"EVIDENCE SUMMARY", r"supporting", r"/5\b"]


def main() -> int:
    G1A.mkdir(parents=True, exist_ok=True)
    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    out, checks = [], {"classifier_block_identical_to_g1": 0, "forbidden_pattern_hits": 0, "reference_sentences_in_message": 0, "n": 0}
    from src.generation.metrics import norm_sentence, sentences
    for c in g1_cases:
        payload = build_payload_g1a([{"finding": f, "calibrated_probability": c["classifier_probabilities"][f]} for f in c["classifier_positive_findings"]], c["no_finding_positive"])
        assert_payload_clean_g1a(payload)
        msg = render_user_message_g1a(payload)
        checks["n"] += 1
        checks["classifier_block_identical_to_g1"] += int(classifier_block(msg) == classifier_block(c["user_message"]))
        checks["forbidden_pattern_hits"] += sum(bool(re.search(p, msg, re.IGNORECASE)) for p in FORBIDDEN_IN_MESSAGE)
        ref_sents = [norm_sentence(s) for s in sentences(c["reference"]["combined"]) if len(norm_sentence(s).split()) >= 6]
        nm = norm_sentence(msg)
        checks["reference_sentences_in_message"] += sum(s in nm for s in ref_sents)
        out.append({"uid": c["uid"], "anon_id": c["anon_id"], "image_id": c["image_id"], "classifier_positive_findings": c["classifier_positive_findings"], "no_finding_positive": c["no_finding_positive"],
                    "query_status": c["query_status"], "payload": payload, "user_message": msg, "user_message_sha256": hashlib.sha256(msg.encode()).hexdigest()})
    with open(G1A / "g1a_cases.jsonl", "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (G1A / "g1a_prompt_diff_vs_g1.txt").write_text("== SYSTEM PROMPT DIFF (G1 -> G1A) ==\n" + prompt_diff_vs_g1() + "\n\n== USER MESSAGE DIFF, example study with findings and retrieval ==\n"
                                                    + user_message_diff_vs_g1(next(c for c in g1_cases if c["retrieved"] and c["classifier_positive_findings"])["payload"]) + "\n", encoding="utf-8")
    same_studies = [c["uid"] for c in out] == [c["uid"] for c in g1_cases]
    report = {**checks, "same_547_studies_and_order_as_g1": same_studies, "prompt_sha256_g1a": prompt_hash(), "payload_whitelist": ["classifier"],
              "retrieval_content_in_any_message": checks["forbidden_pattern_hits"], "note": "empty-classifier studies keep the G1 behaviour (rule 9 / the same classifier block); G1 only differed by an explicit 'no retrieval query' line"}
    (G1A / "g1a_input_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if checks["classifier_block_identical_to_g1"] == checks["n"] and checks["forbidden_pattern_hits"] == 0 and checks["reference_sentences_in_message"] == 0 and same_studies else 1


if __name__ == "__main__":
    raise SystemExit(main())
