"""CheXpert target detection, label-state preservation and identifier extraction."""

import numpy as np
import pandas as pd
import pytest

from src.data.chexpert import (EXPECTED_OBSERVATIONS, detect_target_columns, label_state_frame, load_chexpert_csv,
                               parse_path)
from src.utils.audit import AuditLog

HEADER = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *EXPECTED_OBSERVATIONS]


def _row(path, labels, sex="Male", age="60", view="Frontal", appa="AP"):
    return [path, sex, age, view, appa, *labels]


@pytest.fixture
def csv_file(tmp_path):
    n = len(EXPECTED_OBSERVATIONS)
    rows = [
        _row("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg", ["1.0"] + [""] * (n - 1)),
        _row("CheXpert-v1.0-small/train/patient00002/study2/view1_frontal.jpg", ["", "-1.0", "0.0"] + [""] * (n - 3)),
        _row("CheXpert-v1.0-small/train/patient00002/study2/view2_lateral.jpg", ["", "-1.0", "0.0"] + [""] * (n - 3),
             view="Lateral", appa=""),
    ]
    p = tmp_path / "train.csv"
    pd.DataFrame(rows, columns=HEADER).to_csv(p, index=False)
    return p


def test_detects_exactly_the_14_observations_in_csv_order(csv_file):
    df, targets = load_chexpert_csv(csv_file, "train")
    assert targets == list(EXPECTED_OBSERVATIONS)


def test_non_label_columns_are_not_targets():
    df = pd.DataFrame({"Path": ["a"], "Age": ["50"], "Edema": ["1.0"], "Notes": ["hello"], "Score": ["2.0"]})
    audit = AuditLog("t")
    assert detect_target_columns(df, audit) == ["Edema"]
    issues = set(audit.to_frame()["record_id"])
    assert {"Notes", "Score"} <= issues


def test_four_label_states_are_preserved_not_converted(csv_file):
    df, targets = load_chexpert_csv(csv_file, "train")
    ec = df["Enlarged Cardiomediastinum"].tolist()
    assert ec[0] != ec[0]  # NaN stays NaN (not 0)
    assert ec[1:] == [-1.0, -1.0]  # uncertain stays -1 (not 0 or 1)
    assert df["Cardiomegaly"].tolist()[1:] == [0.0, 0.0]
    st = label_state_frame(df, targets).set_index("observation")
    assert st.loc["Enlarged Cardiomediastinum", ["positive_count", "negative_count", "uncertain_count",
                                                 "missing_count"]].tolist() == [0, 0, 2, 1]
    assert (st["other_count"] == 0).all()


def test_invalid_label_value_is_audited_not_silently_coerced(tmp_path):
    n = len(EXPECTED_OBSERVATIONS)
    rows = [_row("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg", ["1.0"] + [""] * (n - 1)),
            _row("CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg", ["2.0"] + [""] * (n - 1))]
    p = tmp_path / "bad.csv"
    pd.DataFrame(rows, columns=HEADER).to_csv(p, index=False)
    audit = AuditLog("t")
    _, targets = load_chexpert_csv(p, "train", audit)
    # A column with an out-of-vocabulary value is reported, never treated as a clean target.
    assert "No Finding" not in targets
    assert "expected_target_missing" in set(audit.to_frame()["issue"])


def test_patient_and_study_ids_from_path():
    r = parse_path("CheXpert-v1.0-small/train/patient00042/study3/view2_lateral.jpg")
    assert r["patient_id"] == "patient00042"
    assert r["study_id"] == "patient00042/study3"
    assert r["view_index"] == 2 and r["view_from_path"] == "lateral" and r["extension"] == "jpg"


def test_study_ids_are_qualified_by_patient():
    a = parse_path("x/train/patient00001/study1/view1_frontal.jpg")["study_id"]
    b = parse_path("x/train/patient00002/study1/view1_frontal.jpg")["study_id"]
    assert a != b  # study numbering restarts per patient


def test_windows_separators_and_unparseable_paths():
    assert parse_path(r"CheXpert\train\patient00007\study1\view1_frontal.jpg")["patient_id"] == "patient00007"
    bad = parse_path("some/other/file.png")
    assert bad["patient_id"] is None and bad["study_id"] is None and bad["extension"] == "png"


def test_unparseable_path_is_audited(tmp_path):
    n = len(EXPECTED_OBSERVATIONS)
    p = tmp_path / "t.csv"
    pd.DataFrame([_row("weird/path.jpg", [""] * n)], columns=HEADER).to_csv(p, index=False)
    audit = AuditLog("t")
    df, _ = load_chexpert_csv(p, "train", audit)
    assert df["patient_id"].isna().all()
    assert "unparseable_path" in set(audit.to_frame()["issue"])


def test_missing_and_empty_csv_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chexpert_csv(tmp_path / "nope.csv", "train")
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=HEADER).to_csv(empty, index=False)
    with pytest.raises(ValueError):
        load_chexpert_csv(empty, "train")


def test_duplicate_path_reference_is_audited(tmp_path):
    n = len(EXPECTED_OBSERVATIONS)
    path = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    p = tmp_path / "dup.csv"
    pd.DataFrame([_row(path, ["1.0"] + [""] * (n - 1))] * 2, columns=HEADER).to_csv(p, index=False)
    audit = AuditLog("t")
    load_chexpert_csv(p, "train", audit)
    assert "duplicate_path_reference" in set(audit.to_frame()["issue"])
    assert np.isfinite(1.0)
