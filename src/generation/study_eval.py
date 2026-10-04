"""Per-study evaluation row for a generated report, with the SAME definitions as the G1 evaluation (scripts/g1_04_evaluate.py):
stated = affirmative mention (definite or hedged), reference = MeSH-mapped IU findings, hallucination / omission / propagation /
retention / provenance exactly as in G1. Used by the G1A ablation; the G1 results are re-derived through this function as a
reproduction check against the stored G1 per-study results."""

from __future__ import annotations

from src.classification.labels import NO_FINDING
from src.generation.extraction import ReportExtractor
from src.generation.metrics import classifier_propagation, copy_stats, provenance, repeated_sentence_rate, study_counts, text_metrics, tokens
from src.generation.parse import combined

FS = lambda x=(): frozenset(x)  # noqa: E731


def state_of_truth(truth: list[str]) -> str:
    return "normal" if set(truth) == {NO_FINDING} else "abnormal"


def classifier_state(c: dict) -> str:
    if c["classifier_positive_findings"]:
        return "abnormal"
    return "normal" if c["no_finding_positive"] else "indeterminate"


def study_row(c: dict, sysname: str, gen: dict, ext: ReportExtractor) -> dict:
    """c: a g1_cases.jsonl entry (classifier info, G1 retrieval, reference); gen: {'findings','impression','format_ok'}."""
    ref = c["reference"]
    truth = FS(ref["truth_findings"]) - {NO_FINDING} if ref["truth_findings"] is not None else None
    ref_text_set = ext.stated(ref["combined"])
    retr_sets = [FS(r["mapped_findings"]) for r in c["retrieved"]]
    retr_union = FS().union(*retr_sets) if retr_sets else FS()
    clf = FS(c["classifier_positive_findings"])
    text = combined(gen)
    stated, definite = ext.stated(text), ext.definite(text)
    row = {"uid": c["uid"], "anon_id": c["anon_id"], "system": sysname, "in_clinical": truth is not None, "words": len(tokens(text)), "format_ok": gen["format_ok"], "report_state": ext.report_state(text),
           "stated": sorted(stated), "definite": sorted(definite), "clf_pos": sorted(clf), "truth": sorted(truth) if truth is not None else None, "classifier_state": classifier_state(c),
           "reference_state": state_of_truth(ref["truth_findings"]) if truth is not None else None, "repeated_sentence_rate": repeated_sentence_rate(text), **text_metrics(text, ref["combined"])}
    cs = copy_stats(text, [r["findings"] + " " + r["impression"] for r in c["retrieved"]])
    row.update({f"copy_{k}": v for k, v in cs.items()})
    row.update({"sentences": cs["n_sentences"], "unmappable_sentences": len(ext.unmappable_statements(text))})
    if truth is not None:
        for tag, gs in (("", stated), ("def_", definite), ("reftext_", stated)):
            tr = truth if tag != "reftext_" else ref_text_set
            row.update({f"{tag}{k}": v for k, v in study_counts(gs, tr).items()})
            row[f"{tag}hall"], row[f"{tag}omit"] = int(row[f"{tag}fp"] > 0), int(row[f"{tag}fn"] > 0)
        for tag, gs in (("", stated), ("def_", definite)):
            row.update({f"{tag}{k}": v for k, v in classifier_propagation(clf, truth, gs).items()})
        row.update({f"prov_{k}": v for k, v in provenance(stated, clf, retr_union).items()})
    return row
