"""G1F step 1: apply the deterministic routing guard to the 547 validation studies and write the final validation outputs.
NO new retrieval and NO Ollama call: Path A / Path B reuse the stored frozen G1 reports byte for byte; only Path C (empty classifier output) is replaced by
the fixed indeterminate message. The original G1 files are read only.

    .venv\\Scripts\\python.exe -m scripts.g1f_01_build_outputs
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.classification.labels import LABELS
from src.generation.parse import parse_report
from src.system.guard import (INDETERMINATE_FINDINGS, INDETERMINATE_IMPRESSION, PATH_OF_STATE, PATHOLOGY_LABELS, STATE_INDETERMINATE, STATES, SUPPORT_DEVICES, route, run_guarded)
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1F = PROJECT_ROOT / "results/report_generation/experiments/g1f_final_system"
sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()  # noqa: E731


def main() -> int:
    G1F.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    stored_csv = pd.read_csv(G1 / "generated_reports.csv", dtype={"study_id": str}).fillna("").set_index("study_id")
    rows, calls = [], {"retrieve": 0, "generate": 0}

    def retrieve_stub():            # the frozen G1 retrieval was run once and stored; the final system would call it here (counted, never performed in this run)
        calls["retrieve"] += 1
        return None

    for c in cases:
        u = c["uid"]
        pos, nf = c["classifier_positive_findings"], c["no_finding_positive"]
        parsed = parse_report(raw[u]["text"])

        def generate_stub(_retrieved, parsed=parsed):
            calls["generate"] += 1
            return {"findings": parsed["findings"], "impression": parsed["impression"]}

        before = dict(calls)
        res = run_guarded(pos, nf, retrieve_stub, generate_stub)
        invoked_in_g1f_run_stub = {k: calls[k] - before[k] for k in calls}                  # the stored G1 report is replayed through the same call path; no model is contacted
        state = route(pos, nf)
        assert res.system_interpretation_state == state and res.retrieval_invoked == res.llm_invoked == (state != STATE_INDETERMINATE)
        assert (state == STATE_INDETERMINATE) == (not c["retrieved"])          # the guard abstains exactly where G1 had no retrieval query
        g1_text_sha = sha(raw[u]["text"])
        rows.append({"study_id": u, "anon_id": c["anon_id"], "classifier_positive_findings": "; ".join(pos), "calibrated_probabilities": json.dumps({l: round(c["classifier_probabilities"][l], 4) for l in LABELS}),
                     "no_finding_positive": nf, "system_interpretation_state": state, "routing_path": PATH_OF_STATE[state], "retrieval_invoked_in_final_pipeline": res.retrieval_invoked, "llm_invoked_in_final_pipeline": res.llm_invoked,
                     "new_llm_calls_in_g1f_run": 0, "new_retrieval_calls_in_g1f_run": 0, "final_findings": res.findings, "final_impression": res.impression,
                     "source_g1_study_id": "" if state == STATE_INDETERMINATE else u, "source_g1_raw_report_sha256": "" if state == STATE_INDETERMINATE else g1_text_sha,
                     "replaced_g1_raw_report_sha256": g1_text_sha if state == STATE_INDETERMINATE else "", "g1_retrieval_was_performed_in_g1": bool(c["retrieved"]), "stub_calls_through_guard": json.dumps(invoked_in_g1f_run_stub)})
    df = pd.DataFrame(rows)
    df.to_csv(G1F / "g1f_final_validation_outputs.csv", index=False)

    ab = df[df.routing_path.isin(["A", "B"])]
    identical = int(sum(r.final_findings == parse_report(raw[r.study_id]["text"])["findings"] and r.final_impression == parse_report(raw[r.study_id]["text"])["impression"] and r.source_g1_raw_report_sha256 == sha(raw[r.study_id]["text"]) for r in ab.itertuples()))
    in_csv = int(sum(f"FINDINGS: {r.final_findings} IMPRESSION: {r.final_impression}".split() == str(stored_csv.loc[r.study_id, "G1_report"]).split() for r in ab.itertuples()))
    c_ids = set(df[df.routing_path == "C"].study_id)
    empty_ids = {c["uid"] for c in cases if not c["retrieved"]}
    pos_only_device = [c for c in cases if c["classifier_positive_findings"] == [SUPPORT_DEVICES]]
    counts = {"n_studies": len(df), "path_counts": df.routing_path.value_counts().sort_index().to_dict(), "state_counts": df.system_interpretation_state.value_counts().to_dict(),
              "path_C_equals_the_49_empty_classifier_outputs": c_ids == empty_ids and len(c_ids) == 49, "n_path_C": len(c_ids), "n_empty_classifier_outputs_in_g1": len(empty_ids),
              "n_path_C_in_clinical_subset": int(sum(c["in_clinical_subset"] for c in cases if c["uid"] in c_ids)),
              "path_A_B_reports_byte_identical_to_stored_g1": f"{identical} of {len(ab)}", "path_A_B_equal_to_g1_generated_reports_csv_text": f"{in_csv} of {len(ab)}",
              "retrieval_invoked_in_final_pipeline": int(df.retrieval_invoked_in_final_pipeline.sum()), "llm_invoked_in_final_pipeline": int(df.llm_invoked_in_final_pipeline.sum()),
              "new_llm_calls_in_g1f_run": int(df.new_llm_calls_in_g1f_run.sum()), "new_retrieval_calls_in_g1f_run": int(df.new_retrieval_calls_in_g1f_run.sum()), "path_C_retrieval_or_llm_calls_through_guard": sum(1 for r in df[df.routing_path == "C"].itertuples() if r.stub_calls_through_guard != '{"retrieve": 0, "generate": 0}'),
              "support_devices_edge_cases": {"studies_with_support_devices_as_only_positive_label": len(pos_only_device), "of_which_no_finding_positive_routed_normal_path_A": sum(c["no_finding_positive"] for c in pos_only_device),
                                             "of_which_no_finding_negative_routed_abnormal_path_B": sum(not c["no_finding_positive"] for c in pos_only_device)},
              "alternative_readings_for_reference_only": {"strict_c4_pathology_definition (Support Devices not a trigger for Path B)": {"A": 269, "B": 228, "C": 50}, "any_positive_label_is_pathology": {"A": 261, "B": 237, "C": 49}},
              "pathology_labels_that_make_a_study_not_explicit_normal": list(PATHOLOGY_LABELS), "indeterminate_wording": {"findings": INDETERMINATE_FINDINGS, "impression": INDETERMINATE_IMPRESSION}}
    (G1F / "g1f_routing_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    ok = (counts["path_C_equals_the_49_empty_classifier_outputs"] and identical == len(ab) and counts["new_llm_calls_in_g1f_run"] == 0 and counts["path_C_retrieval_or_llm_calls_through_guard"] == 0 and set(df.system_interpretation_state) <= set(STATES))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
