"""IU X-Ray parsing, report grouping, split leakage and the retrieval-corpus invariant."""

import numpy as np
import pandas as pd
import pytest

from src.data.iu_xray import approx_token_count, clean_section, load_projections, load_reports, section_status
from src.data.retrieval_corpus import RetrievalLeakageError, assert_train_only, eligible_report_ids
from src.utils.audit import AuditLog
from src.utils.config import SPLITS_DIR


def test_nan_text_and_sections():
    assert clean_section(np.nan) == "" and clean_section(None) == ""
    assert clean_section("  a \n b ") == "a b"
    assert section_status("x", "y") == "both"
    assert section_status("", "y") == "impression_only"
    assert section_status("x", "") == "findings_only"
    assert section_status("", "") == "neither"
    assert approx_token_count("No pneumothorax.") == 3


def test_loaders_audit_bad_rows(tmp_path):
    rp = tmp_path / "r.csv"
    pd.DataFrame({"uid": ["1", "", "3", "3"], "findings": ["a", "b", None, "d"], "impression": ["x", "y", "z", "w"]}).to_csv(rp, index=False)
    audit = AuditLog("t")
    r = load_reports(rp, audit)
    issues = set(audit.to_frame()["issue"])
    assert {"missing_report_uid", "duplicate_report_uid"} <= issues
    assert r.loc[2, "findings"] == ""

    pp = tmp_path / "p.csv"
    pd.DataFrame({"uid": ["1", "1", "2"], "filename": ["a.png", "a.png", "b.tif"],
                  "projection": ["Frontal", "Lateral", None]}).to_csv(pp, index=False)
    audit = AuditLog("t")
    p = load_projections(pp, audit)
    issues = set(audit.to_frame()["issue"])
    assert {"duplicate_image_reference", "unknown_projection"} <= issues
    assert p.loc[2, "extension"] == "tif"


def test_retrieval_corpus_is_train_only():
    splits = pd.DataFrame({"uid": ["1", "2", "3", "4"], "split": ["train", "val", "test", "train"],
                           "has_any_text": [True, True, True, False]})
    assert eligible_report_ids(splits) == ["1"]
    assert_train_only(["1"], splits)
    with pytest.raises(RetrievalLeakageError):
        assert_train_only(["1", "2"], splits)
    with pytest.raises(RetrievalLeakageError):
        assert_train_only(["99"], splits)


IU = SPLITS_DIR / "iu_xray"
needs_iu = pytest.mark.skipif(not (IU / "iu_report_splits.csv").exists(), reason="IU split manifest not generated yet")


@needs_iu
def test_real_iu_reports_in_exactly_one_split_and_images_follow_report():
    rep = pd.read_csv(IU / "iu_report_splits.csv", dtype={"uid": str})
    img = pd.read_csv(IU / "iu_image_manifest.csv", dtype={"uid": str})
    assert rep["uid"].is_unique
    assert set(rep["split"]) == {"train", "val", "test"}
    split_of = rep.set_index("uid")["split"]
    assigned = img[img["uid"].isin(split_of.index)]
    assert (assigned["split"] == assigned["uid"].map(split_of)).all()
    assert img.groupby("uid")["split"].nunique().max() == 1
    assert img["filename"].is_unique


@needs_iu
def test_real_retrieval_corpus_contains_no_val_or_test_report():
    rep = pd.read_csv(IU / "iu_report_splits.csv", dtype={"uid": str})
    corpus = pd.read_csv(IU / "retrieval_corpus_uids.csv", dtype={"uid": str})["uid"]
    assert_train_only(corpus, rep)
    assert set(corpus) == set(rep.loc[rep["retrieval_corpus_eligible"], "uid"])
    assert not set(corpus) & set(rep.loc[rep["split"] != "train", "uid"])
