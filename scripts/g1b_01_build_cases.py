"""G1B step 1: build the sham-retrieval inputs for the same 547 studies as G1 and FREEZE the sham mapping (with its hash) before any
generation. The G1 prompt, payload builder, message format and evidence-table function are reused unchanged; the ONLY change is the identity of
the five context reports: deterministic sham reports sampled from the R2 retrieval corpus (training reports only), never the query study, never
the study's own true Top-5, and never chosen from reference or classifier information. No model call.

    .venv\\Scripts\\python.exe -m scripts.g1b_01_build_cases
"""

from __future__ import annotations

import hashlib
import json
import random

import pandas as pd

from src.classification.final_test import sha256_file
from src.generation.metrics import norm_sentence, sentences
from src.generation.prompts import assert_payload_clean, build_payload, prompt_hash, render_user_message
from src.retrieval.r2_context import FS, R1, R2
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
SEED, K, SCHEME = 42, 5, "g1b-sham-v1"


def sham_ids(uid: str, pool: list[str], exclude: set[str]) -> list[str]:
    """Deterministic, per-study sampling: independent of study order and of every study attribute except its id."""
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{SCHEME}|seed={SEED}|{uid}".encode()).digest()[:8], "big"))
    cand = [d for d in pool if d not in exclude]
    return rng.sample(cand, K)


def main() -> int:
    G1B.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("").set_index("uid")
    pool = json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8"))          # the R2 retrieval corpus (IU training reports only)
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    test_ids, corpus_ids = set(split["study_ids"]["locked_test"]), set(pool)
    assert not (corpus_ids & test_ids) and sorted(pool) == sorted(corpus.index)
    pool = sorted(pool, key=int)
    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]

    mapping, out, checks = [], [], {"n": 0, "n_with_context": 0, "n_empty_query_identical_to_g1": 0, "n_empty_query": 0, "query_study_in_sham": 0, "true_top5_overlap": 0, "sham_not_in_corpus": 0,
                                     "sham_in_locked_test": 0, "duplicates_within_study": 0, "reference_sentences_outside_context": 0, "format_identical_skeleton": 0}
    for c in g1_cases:
        u = c["uid"]
        checks["n"] += 1
        true5 = [r["study_id"] for r in c["retrieved"]]
        if not true5:                                                       # empty-query handling identical to G1: same payload, same message
            msg = c["user_message"]
            c2 = dict(c, retrieved=[], sham_context=False)
            checks["n_empty_query"] += 1
            checks["n_empty_query_identical_to_g1"] += int(msg == c["user_message"])
            out.append(c2)
            continue
        checks["n_with_context"] += 1
        sh = sham_ids(u, pool, {u} | set(true5))
        retrieved = []
        for r, d in zip(c["retrieved"], sh):                                 # structure kept: same rank slot, same displayed rank/score fields as the G1 slot it replaces
            retrieved.append({"rank": r["rank"], "study_id": d, "dense_rank": r["dense_rank"], "bm25_rank": r["bm25_rank"], "rrf_score": r["rrf_score"], "findings": corpus.loc[d, "findings_clean"],
                              "impression": corpus.loc[d, "impression_clean"], "mapped_findings": sorted(FS(corpus.loc[d, "eval_set"]))})
            mapping.append({"uid": u, "rank": r["rank"], "sham_study_id": d, "replaced_true_study_id": r["study_id"]})
        probs = c["classifier_probabilities"]
        payload = build_payload([{"finding": f, "calibrated_probability": probs[f]} for f in c["classifier_positive_findings"]], c["no_finding_positive"], retrieved)
        assert_payload_clean(payload, u, corpus_ids, test_ids)
        msg = render_user_message(payload)
        checks["query_study_in_sham"] += int(u in sh)
        checks["true_top5_overlap"] += len(set(sh) & set(true5))
        checks["sham_not_in_corpus"] += sum(d not in corpus_ids for d in sh)
        checks["sham_in_locked_test"] += sum(d in test_ids for d in sh)
        checks["duplicates_within_study"] += int(len(set(sh)) != K)
        checks["format_identical_skeleton"] += int([l.split(" study ")[0] if l.startswith("[") else l for l in msg.splitlines() if l.startswith(("[", "A.", "B.", "C.", "Write"))]
                                                  == [l.split(" study ")[0] if l.startswith("[") else l for l in c["user_message"].splitlines() if l.startswith(("[", "A.", "B.", "C.", "Write"))])
        ctx_norm = norm_sentence(" ".join(r["findings"] + " " + r["impression"] for r in retrieved))
        nm = norm_sentence(msg)
        checks["reference_sentences_outside_context"] += sum(s in nm and s not in ctx_norm for s in [norm_sentence(x) for x in sentences(c["reference"]["combined"]) if len(norm_sentence(x).split()) >= 6])
        out.append(dict(c, retrieved=retrieved, evidence_rows=payload["evidence_table"], payload=payload, user_message=msg, user_message_sha256=hashlib.sha256(msg.encode()).hexdigest(), sham_context=True,
                        g1_true_retrieved_ids=true5))
    mdf = pd.DataFrame(mapping)
    mdf.to_csv(G1B / "g1b_sham_mapping.csv", index=False)
    with open(G1B / "g1b_cases.jsonl", "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    frozen = {"scheme": SCHEME, "seed": SEED, "k": K, "pool": "R2 retrieval corpus = IU training reports (corpus_study_ids.json)", "pool_size": len(pool),
              "exclusions_per_study": ["the query study itself", "the study's own true R2 Top-5 reports"], "selection_uses": "study id only (no reference, finding, classifier or image information)",
              "displayed_rank_fields": "dense rank, BM25 rank and fusion score are carried over from the G1 slot the sham report replaces, so the message skeleton is byte-structure-identical",
              "evidence_table": "computed by the unchanged G1 function from the sham context reports (as G1 computes it from its context)", "empty_query_studies": "unchanged from G1",
              "n_mapping_rows": len(mdf), "mapping_sha256": sha256_file(G1B / "g1b_sham_mapping.csv"), "g1_prompt_sha256": prompt_hash(),
              "r2_candidate_config_sha256": sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json"), "frozen_before_generation": True}
    (G1B / "g1b_sham_mapping_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    same_prompt = g1_prompt["prompt_sha256"] == prompt_hash()
    rep = {**checks, "same_order_and_studies_as_g1": [c["uid"] for c in out] == [c["uid"] for c in g1_cases], "g1_prompt_sha256_equals_current_prompt": same_prompt,
           "mapping_sha256": frozen["mapping_sha256"], "distinct_sham_reports_used": int(mdf.sham_study_id.nunique()), "pool_size": len(pool)}
    (G1B / "g1b_input_checks.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    ok = (same_prompt and rep["same_order_and_studies_as_g1"] and checks["query_study_in_sham"] == 0 and checks["true_top5_overlap"] == 0 and checks["sham_not_in_corpus"] == 0 and checks["sham_in_locked_test"] == 0
          and checks["duplicates_within_study"] == 0 and checks["n_empty_query_identical_to_g1"] == checks["n_empty_query"] and checks["format_identical_skeleton"] == checks["n_with_context"] and checks["n_with_context"] * K == len(mdf))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
