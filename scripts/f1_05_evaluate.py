"""F1 step 5: all analyses for one partition (validation dry run or the locked test). The validation dry run must reproduce the stored G1F validation metrics.

    .venv\\Scripts\\python.exe -m scripts.f1_05_evaluate validation
    .venv\\Scripts\\python.exe -m scripts.f1_05_evaluate test
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.f1.analysis import classification_eval, report_eval, retrieval_eval
from src.f1.pipeline import out_dir, require_open
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1F = PROJECT_ROOT / "results/report_generation/experiments/g1f_final_system"


def main(partition: str) -> int:
    require_open(partition)
    od = out_dir(partition)
    pop = pd.read_csv(od / "population.csv", dtype={"uid": str}).fillna({"eval_set": "", "mapped_findings": "", "uncertain_terms": "", "unmapped_terms": "", "frontal_image": ""}).set_index("uid")
    pop["has_eval_set"] = pop["has_eval_set"].astype(bool)
    cls_df = pd.read_csv(od / "classifier_outputs.csv", dtype={"uid": str}).fillna("")
    cases = [json.loads(l) for l in open(od / "cases.jsonl", encoding="utf-8")]
    raw_path = G1 / "g1_reports_raw.jsonl" if partition == "validation" else od / "g1_reports_raw.jsonl"
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(raw_path, encoding="utf-8")}
    clf = classification_eval(partition, pop, cls_df, od)
    ret = retrieval_eval(od)
    rep = report_eval(partition, cases, raw, od)
    flow = json.loads((od / "population_flow.json").read_text(encoding="utf-8"))
    summary = {"partition": partition, "population_flow": flow, "classification": {k: clf[k] for k in ("n_studies", "point", "ci95", "n_classes_defined", "undefined_classes", "pooled_ece")}, "retrieval_n_queries": ret["n_queries"], "report": rep}
    # runtime (test run only has generation timings of its own)
    rt = {"classifier": json.loads((od / "classifier_run.json").read_text(encoding="utf-8"))["seconds"], "prepare": json.loads((od / "prepare_runtime.json").read_text(encoding="utf-8"))}
    if partition == "test" and (od / "generation_run_log.json").exists():
        gl = json.loads((od / "generation_run_log.json").read_text(encoding="utf-8"))
        secs = np.array([raw[c["uid"]]["wall_seconds"] for c in cases if c["uid"] in raw], float)
        rt["generation"] = {"n_generated": int(len(secs)), "mean_seconds_per_report": float(secs.mean()), "median_seconds_per_report": float(np.median(secs)), "total_call_seconds": float(secs.sum()), "wall_seconds_this_run": gl["wall_seconds_this_run"],
                            "mean_output_tokens": float(np.mean([raw[c["uid"]]["output_tokens"] for c in cases if c["uid"] in raw])), "mean_prompt_tokens": float(np.mean([raw[c["uid"]]["prompt_tokens"] for c in cases if c["uid"] in raw])),
                            "gpu_memory_used_mib_peak_sampled": gl.get("gpu_memory_used_mib_peak_sampled"), "ollama_version": gl["ollama_version"], "model_tag": gl["model_tag"], "model_digest": gl["model_digest"], "hardware": gl.get("hardware_at_end")}
        rt["total_end_to_end_seconds"] = rt["classifier"]["total"] + rt["prepare"]["case_building_incl_query_embedding_and_hybrid_retrieval_seconds"] + gl["wall_seconds_this_run"]
    summary["runtime"] = rt
    (od / "f1_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    ok = True
    if partition == "validation":
        stored = pd.read_csv(G1F / "g1f_validation_metrics.csv", float_precision="round_trip").set_index("metric")
        diffs = {}
        for k in ("precision", "recall", "f1", "macro_f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "rouge_l", "bleu4", "meteor_exact", "mean_words"):
            diffs[k] = abs(rep["metrics"][k]["value"] - stored.loc[k, "G1F_final_system"])
        chk = {"max_abs_diff_vs_stored_g1f_validation_metrics": max(diffs.values()), "per_metric": diffs, "all_reproduced": max(diffs.values()) < 1e-9}
        # stored abstention / three-state numbers
        sa = json.loads((G1F / "g1f_abstention_analysis.json").read_text(encoding="utf-8"))
        chk["abstention_counts_reproduced"] = (rep["abstention"]["n_abstained_P3"] == sa["n_abstained_all_studies"] and rep["abstention"]["n_abstained_P4"] == sa["n_abstained_with_reference"] and rep["abstention"]["reference_abnormal_among_abstained"] == sa["reference_abnormal_among_abstained"])
        sdm = pd.read_csv(G1F / "g1f_decided_case_metrics.csv").set_index("metric")
        chk["decided_metrics_reproduced"] = bool(abs(rep["three_state"]["decided"]["normal_recall_among_decided"]["value"] - sdm.loc["normal_recall_among_decided", "value"]) < 1e-9 and abs(rep["three_state"]["decided"]["abnormal_recall_among_decided"]["value"] - sdm.loc["abnormal_recall_among_decided", "value"]) < 1e-9
                                                 and abs(rep["three_state"]["decided"]["decision_coverage"]["value"] - sdm.loc["decision_coverage", "value"]) < 1e-9)
        pr = pd.read_csv(G1F / "g1f_routing_state_vs_report_prose.csv").set_index("routing_state")
        chk["prose_consistency_reproduced"] = bool(rep["routing_prose_consistency"]["abnormal_routed_P3"]["prose_normal"] == pr.loc["abnormal", "final_report_prose_normal"] and rep["routing_prose_consistency"]["abnormal_routed_P3"]["n"] == pr.loc["abnormal", "n"])
        chk["all_reproduced"] = bool(chk["all_reproduced"] and chk["abstention_counts_reproduced"] and chk["decided_metrics_reproduced"] and chk["prose_consistency_reproduced"])
        (od / "evaluation_reproduction_checks.json").write_text(json.dumps(chk, indent=2, default=float), encoding="utf-8")
        print(json.dumps(chk, indent=1, default=float))
        ok = chk["all_reproduced"]
    pd.set_option("display.width", 200)
    print(json.dumps({"classification_point": {k: round(v, 4) if isinstance(v, float) else v for k, v in clf["point"].items() if k.startswith(("macro", "micro"))}, "n_classification": clf["n_studies"],
                      "report_metrics": {k: round(v["value"], 4) for k, v in rep["metrics"].items()}, "three_state_P3": rep["three_state"]["state_counts_P3"], "abstention": {k: rep["abstention"][k] for k in ("n_abstained_P3", "n_abstained_P4")},
                      "prose_rate": rep["routing_prose_consistency"]["abnormal_routing_to_normal_prose_rate_P3"], "n_generation_failures": rep["n_generation_failures"]}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
