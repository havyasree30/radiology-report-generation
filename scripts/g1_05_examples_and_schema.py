"""G1 step 5: representative validation examples (deterministic selection, anonymised ids, generated text never edited) and the
REPORT_GENERATION_OUTPUT_SCHEMA.json for the future application.

    .venv\\Scripts\\python.exe -m scripts.g1_05_examples_and_schema
"""

from __future__ import annotations

import json

import pandas as pd

from src.classification.labels import NO_FINDING
from src.generation.parse import parse_report
from src.generation.prompts import prompt_hash
from src.generation.schema import build_record, output_schema, validate_record
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1S, B0S = "G1_single_agent_rag", "B0_rule_based"
FS = lambda x: frozenset(x if x is not None else [])  # noqa: E731


def main() -> int:
    cases = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    df = pd.read_csv(G1 / "g1_per_study_results.csv", dtype={"uid": str})
    for col in ("stated", "definite", "clf_pos", "truth"):
        df[col] = df[col].map(lambda v: json.loads(v.replace("'", '"')) if isinstance(v, str) and v.startswith("[") else (None if pd.isna(v) else v))
    g = df[df.system == G1S].set_index("uid")
    b = df[df.system == B0S].set_index("uid")
    clin = [u for u in sorted(g.index, key=int) if bool(g.loc[u, "in_clinical"])]

    def support(u, f):
        return next((e["n_supporting"] for e in cases[u]["evidence_rows"] if e["finding"] == f), 0)

    def pick(pred, key=None):
        c = [u for u in clin if pred(u)]
        if key:
            c = sorted(c, key=key)
        return c[0] if c else None

    truth = lambda u: FS(g.loc[u, "truth"])      # noqa: E731
    stated = lambda u: FS(g.loc[u, "stated"])    # noqa: E731
    clf = lambda u: FS(g.loc[u, "clf_pos"])      # noqa: E731
    sel = {
        "1_correct_normal": pick(lambda u: g.loc[u, "reference_state"] == "normal" and g.loc[u, "classifier_state"] == "normal" and g.loc[u, "report_state"] == "normal" and not stated(u)),
        "2_correct_abnormal": pick(lambda u: truth(u) and stated(u) == truth(u) and clf(u) >= truth(u), key=lambda u: (-len(truth(u)), int(u))),
        "3_classifier_fp_suppressed": pick(lambda u: any(f not in truth(u) and f not in stated(u) for f in clf(u)), key=lambda u: (-sum(support(u, f) == 0 for f in clf(u) - truth(u) - stated(u)), int(u))),
        "4_classifier_fp_propagated": pick(lambda u: any(f not in truth(u) and f in stated(u) for f in clf(u)), key=lambda u: (support(u, sorted(clf(u) - truth(u) & stated(u))[0]), int(u))),
        "5_important_finding_omitted": pick(lambda u: bool(truth(u) - stated(u)) and bool((truth(u) & clf(u)) - stated(u)), key=lambda u: int(u)),
        "6_retrieval_mismatch": pick(lambda u: bool(cases[u]["retrieved"]) and not any(truth(u) & FS(r["mapped_findings"]) for r in cases[u]["retrieved"]) and bool(truth(u)) and truth(u) != {NO_FINDING}, key=lambda u: int(u)),
    }
    md = ["# G1 representative validation examples", "", "_Deterministic selection (first study by id satisfying each criterion; no manual choice, no editing of generated text). Identifiers are anonymised. "
          "IU 'truth' = MeSH-mapped findings of the study; reference text is shown for comparison only and was never part of the generator input._", ""]
    rows = []
    for name, u in sel.items():
        if u is None:
            md += [f"## {name}", "", "No study in the clinical subset satisfies this criterion.", ""]
            rows.append({"category": name, "anon_id": None})
            continue
        c = cases[u]
        gp = parse_report(raw[u]["text"])
        t, s_, k = sorted(truth(u)), sorted(stated(u)), sorted(clf(u))
        fp, fn = sorted(clf(u) - truth(u)), sorted(truth(u) - stated(u))
        ev = "; ".join(f"{e['finding']} p={e['classifier_probability']:.2f} support {e['n_supporting']}/{e['n_retrieved']}" for e in c["evidence_rows"]) or "no classifier-positive finding"
        retrieved = "; ".join(f"[{r['rank']}] {','.join(r['mapped_findings']) or 'no mapped finding'}" for r in c["retrieved"]) or "none (empty query)"
        analysis = (f"Reference (mapped) findings: {t or 'none mapped'}. Classifier positives: {k or 'none'} (false positives: {fp or 'none'}). G1 stated: {s_ or 'none'}. "
                    f"Omitted reference findings: {fn or 'none'}. Retrieved mapped findings by rank: {retrieved}.")
        md += [f"## {name} — {c['anon_id']}", "", f"- **Classifier:** {ev}; No Finding {'positive' if c['no_finding_positive'] else 'negative'}",
               f"- **Query:** `{c['query'] or '(empty)'}`; retrieved mapped findings: {retrieved}", f"- **B0 (rule-based):** {c['b0']['findings']} {c['b0']['impression']}",
               f"- **G1 generated report:** FINDINGS: {gp['findings']} IMPRESSION: {gp['impression']}", f"- **Reference report:** {c['reference']['combined']}", f"- **Factual error analysis:** {analysis}", ""]
        rows.append({"category": name, "anon_id": c["anon_id"], "classifier_positives": ";".join(k), "reference_findings": ";".join(t), "g1_stated": ";".join(s_), "classifier_fp": ";".join(fp),
                     "omitted": ";".join(fn), "n_retrieved_supporting_any_truth": sum(bool(truth(u) & FS(r["mapped_findings"])) for r in c["retrieved"]), "g1_report_state": g.loc[u, "report_state"],
                     "reference_state": g.loc[u, "reference_state"]})
    (G1 / "qualitative_examples.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(rows).to_csv(G1 / "g1_representative_cases.csv", index=False)

    # ---------------------------------------------------------------- output schema with one real (validation) example
    ex_uid = next(u for u in sorted(cases, key=int) if cases[u]["retrieved"] and cases[u]["classifier_positive_findings"] and u in raw)
    ex = build_record(cases[ex_uid], parse_report(raw[ex_uid]["text"]), "medgemma1.5:4b", prompt_hash(), 0.0)
    validate_record(ex)
    n_valid = 0
    for u, c in cases.items():
        if u in raw:
            validate_record(build_record(c, parse_report(raw[u]["text"]), "medgemma1.5:4b", prompt_hash(), 0.0))
            n_valid += 1
    schema = {"version": "g1-v1", "notes": ["One record per image; fields feed the future Streamlit application.", "calibrated_probabilities_all_classes are independent per-class probabilities and are never normalised across classes.",
                                            "retrieved_evidence carries the retrieval rank and score information (dense rank, BM25 rank, fusion score) of each of the Top-5 reports.",
                                            "The reference report and the IU truth never appear in the record.", "The report is a preliminary AI draft for research use, not a radiologist report."],
              "schema": output_schema(), "example": ex, "n_records_validated_against_schema": n_valid}
    (G1 / "REPORT_GENERATION_OUTPUT_SCHEMA.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected": {k: (cases[u]["anon_id"] if u else None) for k, u in sel.items()}, "records_validated": n_valid}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
