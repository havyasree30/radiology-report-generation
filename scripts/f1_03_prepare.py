"""F1 step 3: population -> frozen classifier -> routing guard -> frozen R2 retrieval -> G1 cases -> retrieval metrics (K = 1, 3, 5, 10; oracle diagnostic).
Run with `validation` (dry run: must reproduce the stored R1 / R2 / G1 results) or `test` (the locked run; requires test_opened.json).

    .venv\\Scripts\\python.exe -m scripts.f1_03_prepare validation
    .venv\\Scripts\\python.exe -m scripts.f1_03_prepare test
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from src.classification.labels import LABELS
from src.f1.pipeline import R1, SPEC, build_cases, classify, load_population, make_engine, out_dir
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
ORACLE_SPEC = {"phrases": "names", "normal": SPEC["normal"]}


def main(partition: str) -> int:
    od = out_dir(partition)
    t_all = time.time()
    pop = load_population(partition)
    pop.to_csv(od / "population.csv")
    cls_df, cinfo = classify(pop, partition)
    cls_df.to_csv(od / "classifier_outputs.csv", index=False, float_format="%.17g")
    (od / "classifier_run.json").write_text(json.dumps(cinfo, indent=2), encoding="utf-8")
    eng = make_engine(pop, cls_df, partition)
    t0 = time.time()
    cases, ctiming = build_cases(eng, pop, cls_df)
    t_cases = time.time() - t0
    with open(od / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    c = eng.ctx
    t0 = time.time()
    eng.run("FINAL_pipeline", SPEC, "hybrid", bm25_spec=SPEC, population=c.primary, store=True)
    t_ret = time.time() - t0
    eng.run("O_final_pipeline", ORACLE_SPEC, "hybrid", oracle=True, bm25_spec=ORACLE_SPEC, population=c.primary, store=False)
    eng.mat["FINAL_pipeline"].reset_index().to_csv(od / "retrieval_classifier_query_per_study.csv", index=False, float_format="%.17g")
    eng.mat["O_final_pipeline"].reset_index().to_csv(od / "retrieval_oracle_query_per_study.csv", index=False, float_format="%.17g")
    pd.DataFrame(eng.top10).to_csv(od / "retrieval_top10.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    c.save_cache()
    st = pd.Series([x["system_interpretation_state"] for x in cases]).value_counts().to_dict()
    flow = {"partition": partition, "locked_or_validation_studies": len(pop), "with_frontal_image": cinfo["n_with_frontal_image"], "without_frontal_image": cinfo["n_without_frontal_image"], "failing_image_validator": cinfo["n_failing_image_validator"],
            "classified_P2": cinfo["n_classified"], "primary_end_to_end_set_P3_(classified_and_usable_reference_report)": len(cases), "classified_without_usable_reference_report": cinfo["n_classified"] - len(cases),
            "clinical_finding_subset_P4": int(sum(x["in_clinical_subset"] for x in cases)), "classification_population_(classified_with_mapped_truth)": len(c.with_cls), "retrieval_primary_population_(non_empty_classifier_query)": len(c.primary),
            "routing_counts_on_P3": st, "routing_counts_on_P2": cls_df.system_interpretation_state.value_counts().to_dict(), "studies_with_mapped_truth_in_partition": int(pop.has_eval_set.sum()),
            "path_C_equals_studies_without_query": True, "reference_normal_in_P4": int(sum(x["reference"]["truth_findings"] == ["No Finding"] for x in cases if x["in_clinical_subset"]))}
    (od / "population_flow.json").write_text(json.dumps(flow, indent=2), encoding="utf-8")
    runtime = {"classifier": cinfo["seconds"], "case_building_incl_query_embedding_and_hybrid_retrieval_seconds": round(t_cases, 2), "mean_query_and_retrieval_seconds_per_study": float(np.mean(ctiming["query_and_retrieval_seconds"])) if ctiming["query_and_retrieval_seconds"] else None,
               "retrieval_evaluation_run_seconds": round(t_ret, 2), "n_retrieval_queries_in_case_building": ctiming["n_queries"], "total_prepare_seconds": round(time.time() - t_all, 2)}
    (od / "prepare_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps({"flow": flow, "runtime": runtime}, indent=1))

    if partition == "validation":                                                          # ---------------- dry-run reproduction of the stored results
        chk: dict = {}
        sf = pd.read_csv(R1 / "iu_study_findings.csv", dtype={"uid": str}).fillna({"eval_set": "", "mapped_findings": "", "uncertain_terms": "", "unmapped_terms": ""}).set_index("uid")
        sf = sf[sf.partition == "validation"]
        vt = pd.read_csv(R1 / "validation_report_text_for_error_analysis.csv", dtype={"uid": str}).set_index("uid")
        chk["population_eval_set_identical"] = bool((pop.eval_set == sf.eval_set.loc[pop.index]).all())
        chk["population_frontal_image_identical"] = bool((pop.frontal_image.fillna("") == sf.frontal_image.fillna("").loc[pop.index]).all())
        chk["population_has_eval_set_identical"] = bool((pop.has_eval_set.astype(bool) == sf.has_eval_set.astype(bool).loc[pop.index]).all())
        chk["population_retrieval_text_identical"] = bool((pop.retrieval_text.fillna("") == vt.retrieval_text.fillna("").loc[pop.index]).all())
        old = pd.read_csv(R1 / "iu_validation_classifier_outputs.csv", dtype={"uid": str}).fillna("").set_index("uid")
        new = cls_df.set_index("uid")
        chk["classifier_same_studies"] = list(old.index) == list(new.index)
        sc = [f"{l}__score" for l in LABELS]
        chk["classifier_score_max_abs_diff"] = float(np.abs(old[sc].astype(float).to_numpy() - new.loc[old.index, sc].astype(float).to_numpy()).max())
        chk["classifier_pred_final_identical"] = bool((old[[f"{l}__pred_final" for l in LABELS]].astype(int).to_numpy() == new.loc[old.index, [f"{l}__pred_final" for l in LABELS]].astype(int).to_numpy()).all())
        g1 = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
        chk["cases_same_studies_and_order"] = [x["uid"] for x in cases] == list(g1)
        chk["cases_user_message_sha256_identical"] = f"{sum(x['user_message_sha256'] == g1[x['uid']]['user_message_sha256'] for x in cases)} of {len(cases)}"
        chk["cases_retrieved_ids_identical"] = bool(all([r['study_id'] for r in x['retrieved']] == [r['study_id'] for r in g1[x['uid']]['retrieved']] for x in cases))
        chk["cases_reference_equal"] = bool(all(x["reference"] == g1[x["uid"]]["reference"] for x in cases))
        stored = pd.read_csv(R2 / "per_query_metrics_all_configs.csv", dtype={"uid": str})
        for name, stored_name in (("FINAL_pipeline", "FINAL_pipeline"), ("O_final_pipeline", "O_final_pipeline")):
            a = stored[stored.config == stored_name].set_index("uid")
            b = eng.mat[name]
            cols = [x for x in b.columns if x in a.columns and pd.api.types.is_numeric_dtype(b[x])]
            chk[f"retrieval_{name}_same_queries"] = sorted(a.index) == sorted(b.index)
            chk[f"retrieval_{name}_max_abs_metric_diff"] = float(np.nanmax(np.abs(a.loc[b.index, cols].to_numpy(float) - b[cols].to_numpy(float))))
        ok = (all(v for k, v in chk.items() if isinstance(v, bool)) and chk["classifier_score_max_abs_diff"] < 1e-6 and chk["cases_user_message_sha256_identical"] == f"{len(cases)} of {len(cases)}"
              and chk["retrieval_FINAL_pipeline_max_abs_metric_diff"] < 1e-9 and chk["retrieval_O_final_pipeline_max_abs_metric_diff"] < 1e-9)
        chk["all_reproduced"] = bool(ok)
        (od / "reproduction_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
        print(json.dumps(chk, indent=1))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
