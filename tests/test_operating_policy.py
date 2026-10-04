"""C4: No Finding consistency rule and the final operating-policy file."""

import numpy as np
import pytest

from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS
from src.classification.operating_policy import (RULE_NAME, apply_final_policy, apply_no_finding_rule,
                                                 load_final_policy, save_final_policy)

NF = LABELS.index(NO_FINDING)
SD = LABELS.index("Support Devices")
EDEMA, FRACTURE = LABELS.index("Edema"), LABELS.index("Fracture")


def _pred(*positive):
    p = np.zeros((1, 14), dtype=bool)
    p[0, list(positive)] = True
    return p


def test_rule_suppresses_no_finding_when_an_abnormal_is_positive():
    out = apply_no_finding_rule(_pred(NF, EDEMA))
    assert not out[0, NF] and out[0, EDEMA]


def test_no_finding_stays_when_no_abnormal_is_positive():
    assert apply_no_finding_rule(_pred(NF))[0, NF]
    assert apply_no_finding_rule(_pred(NF, SD))[0, NF]       # a device is not a pathology
    assert not apply_no_finding_rule(_pred(EDEMA))[0, NF]    # the rule never turns No Finding ON


def test_abnormal_predictions_and_other_rows_are_unchanged_and_input_not_mutated():
    rng = np.random.default_rng(0)
    pred = rng.random((500, 14)) > 0.6
    before = pred.copy()
    out = apply_no_finding_rule(pred)
    assert np.array_equal(pred, before)                                  # input untouched
    keep = [i for i in range(14) if i != NF]
    assert np.array_equal(out[:, keep], pred[:, keep])                   # every non-NF column identical
    changed = np.flatnonzero(out[:, NF] != pred[:, NF])
    assert all(pred[i, NF] and pred[i, [LABELS.index(l) for l in PATHOLOGY_LABELS]].any() for i in changed)
    contradiction = out[:, NF] & out[:, [LABELS.index(l) for l in PATHOLOGY_LABELS]].any(axis=1)
    assert not contradiction.any()                                       # zero contradictions afterwards


def test_class_order_is_preserved_in_policy_file(tmp_path):
    t = np.linspace(0.05, 0.6, 14)
    path = tmp_path / "p.json"
    save_final_policy(path, t, {"dataset_split_for_selection": "validation", "threshold_method": "f1_optimal"})
    d = load_final_policy(path)
    assert list(d["classes"]) == list(LABELS) and d["label_order"] == list(LABELS)
    assert np.array_equal(d["thresholds"], t)                            # bit-exact round trip
    assert d["no_finding_rule"] == RULE_NAME


def test_policy_loader_rejects_bad_files(tmp_path):
    import json
    path = tmp_path / "p.json"
    save_final_policy(path, np.full(14, 0.3), {"dataset_split_for_selection": "validation"})
    d = json.loads(path.read_text())
    for mutate in (lambda x: x.update(no_finding_rule="make_everything_positive"),
                   lambda x: x.update(dataset_split_for_selection="test"),
                   lambda x: x["label_order"].reverse(),
                   lambda x: x["classes"].pop("Edema")):
        bad = json.loads(json.dumps(d))
        mutate(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(Exception):
            load_final_policy(path)
    with pytest.raises(ValueError):
        save_final_policy(path, np.array([0.5] * 13 + [np.nan]), {})


def test_final_policy_reproduces_expected_toy_outputs(tmp_path):
    t = np.full(14, 0.4)
    t[NF] = 0.2
    path = tmp_path / "p.json"
    save_final_policy(path, t, {"dataset_split_for_selection": "validation"})
    pol = load_final_policy(path)
    scores = np.full((3, 14), 0.1)
    scores[0, NF] = 0.9                      # image 0: No Finding only          -> stays positive
    scores[1, NF], scores[1, EDEMA] = 0.9, 0.5   # image 1: NF and Edema          -> NF suppressed
    scores[2, FRACTURE] = 0.39               # image 2: just below threshold, NF 0.1 < 0.2 -> nothing positive
    before = scores.copy()
    raw, final = apply_final_policy(scores, pol)
    assert raw[:, NF].tolist() == [True, True, False]
    assert final[:, NF].tolist() == [True, False, False]
    assert final[1, EDEMA] and not final[2, FRACTURE]
    assert np.array_equal(scores, before)                                # raw scores are never modified
    assert not np.isclose(scores.sum(axis=1), 1.0).all()                 # and never normalised across classes
