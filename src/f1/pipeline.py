"""F1 end-to-end pipeline pieces, parameterised by partition ("validation" = dry run that must reproduce the stored results; "test" = the one locked run).

Everything here re-uses the frozen code paths (R1 mapping and corpus, frozen classifier pipeline, R2 Engine hybrid retrieval with the frozen specification, G1 payload/prompt builders, G1F guard).
The locked test partition cannot be loaded unless `test/test_opened.json` exists (written once by f1_02_open_test after the protocol and the code freeze).
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from src.classification import inference_policy
from src.classification.checkpointing import load_checkpoint
from src.classification.final_test import apply_frozen_pipeline, load_frozen, sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.classification.model import predict_proba
from src.generation.prompts import assert_payload_clean, build_payload, render_user_message
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.retrieval.findings import eval_set, map_study
from src.retrieval.queries import ABNORMAL, classifier_query
from src.retrieval.r2_context import FS
from src.retrieval.text import build_retrieval_text, clean_section
from src.system import guard
from src.utils.config import PROJECT_ROOT, load_paths
from src.validation.image_validator import ImageValidator

R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
EXP = PROJECT_ROOT / "results/classification/experiments"
F1 = PROJECT_ROOT / "results/final_test/f1_locked_end_to_end"
TOPK = 5
SPEC = {"phrases": "names", "normal": "no acute abnormality", "top_n": 3, "weighting": None, "oracle": False}      # the frozen R2 query specification (Top-3 finding names, normal phrase)


def out_dir(partition: str):
    d = F1 / ("test" if partition == "test" else "validation_dry_run")
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_open(partition: str) -> None:
    if partition == "test" and not (F1 / "test/test_opened.json").exists():
        raise PermissionError("the locked test partition is closed: run f1_02_open_test first (it records the code freeze and opens the test once)")


def partition_ids(partition: str) -> list[str]:
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    return sorted(split["study_ids"]["locked_test" if partition == "test" else "validation"], key=int)


def anon(uid: str) -> str:
    return "case_" + hashlib.sha256(f"g1-anon-v1|{uid}".encode()).hexdigest()[:10]


# ------------------------------------------------------------------ population (frozen r1-v1 mapping; same rules as R1/G1)
def load_population(partition: str) -> pd.DataFrame:
    require_open(partition)
    paths = load_paths()
    rep = pd.read_csv(paths.iu_reports_csv)
    proj = pd.read_csv(paths.iu_projections_csv)
    rep["uid"], proj["uid"] = rep["uid"].astype(str), proj["uid"].astype(str)
    ids = partition_ids(partition)
    mapping = json.loads((R1 / "iu_finding_mapping.json").read_text(encoding="utf-8"))
    front = proj[proj["projection"] == "Frontal"].sort_values(["uid", "filename"]).groupby("uid")["filename"].first()
    work = rep.set_index("uid").loc[ids]
    rows = []
    for u in ids:
        r = work.loc[u]
        ms = map_study(r["MeSH"], mapping)
        es = eval_set(ms["findings"], ms["normal"])
        rows.append({"uid": u, "frontal_image": front.get(u, ""), "mesh_raw": r["MeSH"], "mapped_findings": ";".join(ms["findings"]), "indexed_normal": ms["normal"], "eval_set": ";".join(sorted(es, key=LABELS.index)) if es else "",
                     "has_eval_set": es is not None, "uncertain_terms": ";".join(ms["uncertain_terms"]), "unmapped_terms": ";".join(ms["unmapped_terms"]), "findings_clean": clean_section(r["findings"]),
                     "impression_clean": clean_section(r["impression"]), "retrieval_text": build_retrieval_text(r["findings"], r["impression"])[0]})
    return pd.DataFrame(rows).fillna({"eval_set": "", "uncertain_terms": "", "unmapped_terms": "", "mapped_findings": ""}).set_index("uid")


# ------------------------------------------------------------------ frozen classifier inference
def classify(pop: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict]:
    require_open(partition)
    manifest_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    verify_freeze(manifest_path, PROJECT_ROOT)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy, cal = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    vs = pop[pop.frontal_image.fillna("") != ""]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, meta = load_checkpoint(PROJECT_ROOT / man["checkpoint"], device)
    assert sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"] and meta["epoch"] == man["checkpoint_epoch"]
    tf = EvalTransform(PreprocessConfig.from_dict(meta["preprocessing"]))
    assert {k: meta["preprocessing"][k] for k in man["preprocessing"]} == man["preprocessing"]
    inference_policy.apply(inference_policy.CANONICAL)
    images_dir = load_paths().iu_images_dir
    validator = ImageValidator()
    xs, ok, notes = [], [], []
    t_load = time.time()
    for fn in vs.frontal_image:
        res = validator.validate(images_dir / fn)
        ok.append(bool(res.valid))
        notes.append(";".join(w["code"] for w in res.warnings) + (";ERROR:" + ";".join(e["code"] for e in res.errors) if not res.valid else ""))
        xs.append(tf(res.image) if res.valid else None)
    t_loaded = time.time()
    good = [i for i, x in enumerate(xs) if x is not None]
    scores = np.full((len(vs), 14), np.nan)
    with torch.no_grad():
        for s in range(0, len(good), 32):
            idx = good[s:s + 32]
            scores[idx] = predict_proba(model, torch.stack([xs[i] for i in idx]).to(device)).cpu().numpy().astype(np.float64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_done = time.time()
    keep = ~np.isnan(scores).any(1)
    o = apply_frozen_pipeline(scores[keep], policy, cal)
    rows = []
    for k, (u, r) in enumerate(vs[keep].iterrows()):
        pf = {l: bool(o["pred_final"][k, j]) for j, l in enumerate(LABELS)}
        qa = classifier_query(pf)
        row = {"uid": u, "frontal_image": r.frontal_image, "query_all_positive": qa["query"], "query_all_positive_status": qa["status"], "query_all_positive_findings": ";".join(qa["findings"]),
               "system_interpretation_state": guard.route([l for l in ABNORMAL if pf[l]], pf[NO_FINDING])}
        for j, l in enumerate(LABELS):
            row[f"{l}__score"], row[f"{l}__prob"] = float(o["scores"][k, j]), float(o["prob"][k, j])
            row[f"{l}__pred_raw"], row[f"{l}__pred_final"] = int(o["pred_raw"][k, j]), int(o["pred_final"][k, j])
        rows.append(row)
    df = pd.DataFrame(rows)
    failing = [{"uid": u, "image": fn, "notes": n} for u, fn, good_, n in zip(vs.index, vs.frontal_image, ok, notes) if not good_]
    info = {"partition": partition, "n_studies": len(pop), "n_with_frontal_image": int(len(vs)), "n_without_frontal_image": int(len(pop) - len(vs)), "n_failing_image_validator": len(failing), "failing_images": failing, "n_classified": int(len(df)), "device": str(device),
            "inference_policy": "fp32_strict", "checkpoint_sha256": man["checkpoint_sha256"], "seconds": {"model_load": round(t_load - t0, 2), "image_loading_and_preprocessing": round(t_loaded - t_load, 2), "model_inference": round(t_done - t_loaded, 2), "total": round(t_done - t0, 2)},
            "image_validator_warnings": pd.Series([n for n in notes if n]).value_counts().head(10).to_dict(), "classifier_tuned_on_iu": False}
    return df, info


# ------------------------------------------------------------------ retrieval engine bound to a partition
def make_engine(pop: pd.DataFrame, cls_df: pd.DataFrame, partition: str):
    from scripts.r2_02_experiments import Engine
    from src.retrieval.evaluation import bootstrap_draws
    from src.retrieval.metrics import jaccard
    from src.retrieval.r2_context import norm_text
    eng = Engine()
    c = eng.ctx
    c._cache_path = out_dir(partition) / "query_embedding_cache_f1.npz"            # frozen R2 artefacts are never written
    c.val = pop
    c.cls = cls_df.fillna("").set_index("uid")
    c.val_text = pop["retrieval_text"]
    c.eligible = [u for u in pop.index if bool(pop.loc[u, "has_eval_set"])]
    c.truth = {u: FS(pop.loc[u, "eval_set"]) for u in c.eligible}
    c.with_cls = [u for u in c.eligible if u in c.cls.index]
    c.primary = [u for u in c.with_cls if c.cls.loc[u, "query_all_positive_status"] != "empty"]
    c.all_gain = {u: np.array([jaccard(c.truth[u], c.doc_set[d]) if c.doc_set[d] else 0.0 for d in c.ids]) for u in c.eligible}
    c.exact_possible = {u: any(c.doc_set[d] == c.truth[u] for d in c.ids) for u in c.eligible}
    c.dupset = {u for u in c.primary if norm_text(pop.loc[u, "retrieval_text"]) in c.corpus_norms}
    eng.draws = bootstrap_draws(len(c.primary), 1000, 42)
    return eng


# ------------------------------------------------------------------ routing + frozen retrieval + G1 cases (same construction as G1)
def build_cases(eng, pop: pd.DataFrame, cls_df: pd.DataFrame) -> tuple[list[dict], dict]:
    c = eng.ctx
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("").set_index("uid")
    corpus_ids, test_ids = set(c.ids), c.test_ids
    has_ref = lambda u: bool(str(pop.loc[u, "retrieval_text"]).strip()) and str(pop.loc[u, "retrieval_text"]) != "nan"  # noqa: E731
    primary = [u for u in pop.index if u in c.cls.index and has_ref(u)]
    cases, timing = [], {"query_and_retrieval_seconds": []}
    for u in primary:
        positives, probs, nf = c.positives(u), c.probs(u), c.no_finding_positive(u)
        state = guard.route(positives, nf)
        t0 = time.time()
        q = eng.query(u, SPEC)
        retrieved = []
        if q is not None:
            ranked, info = eng.retrieve(q, "hybrid")
            for r, d in enumerate(ranked[:TOPK], 1):
                dr, br, rs = info["prov"][d]
                retrieved.append({"rank": r, "study_id": d, "dense_rank": dr if dr is not None else -1, "bm25_rank": br if br is not None else -1, "rrf_score": float(rs), "findings": corpus.loc[d, "findings_clean"],
                                  "impression": corpus.loc[d, "impression_clean"], "mapped_findings": sorted(c.doc_set[d])})
        timing["query_and_retrieval_seconds"].append(time.time() - t0)
        assert (state == guard.STATE_INDETERMINATE) == (q is None), "routing guard and retrieval query disagree"
        payload = build_payload([{"finding": l, "calibrated_probability": probs[l]} for l in positives], nf, retrieved)
        assert_payload_clean(payload, u, corpus_ids, test_ids)
        msg = render_user_message(payload)
        truth = sorted(c.truth[u]) if u in c.truth else None
        ref_c = str(pop.loc[u, "retrieval_text"])
        cases.append({"uid": u, "anon_id": anon(u), "image_id": pop.loc[u, "frontal_image"], "classifier_positive_findings": positives, "classifier_probabilities": {l: probs[l] for l in LABELS}, "no_finding_positive": bool(nf),
                      "system_interpretation_state": state, "routing_path": guard.PATH_OF_STATE[state], "query": q["text"] if q else "", "query_status": q["status"] if q else "empty", "retrieved": retrieved, "payload": payload,
                      "user_message": msg, "user_message_sha256": hashlib.sha256(msg.encode()).hexdigest(), "in_clinical_subset": truth is not None,
                      "unmapped_terms": pop.loc[u, "unmapped_terms"], "uncertain_terms": pop.loc[u, "uncertain_terms"],
                      "reference": {"findings": pop.loc[u, "findings_clean"], "impression": pop.loc[u, "impression_clean"], "combined": ref_c, "truth_findings": truth}})
    timing["n_queries"] = len(cases)
    return cases, timing
