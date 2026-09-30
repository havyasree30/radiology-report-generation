"""Multi-label association measures, sampler side effects and assertion-aware concept matching."""

import numpy as np
import pytest

from src.analysis import multilabel as ml
from src.analysis.text_concepts import ConceptExtractor
from src.imbalance.sampling import expected_prevalence, inverse_prevalence_weights, kish_effective_sample_size
from src.utils.config import load_yaml

# 4 samples, labels A, B, C
Y = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)


def test_cardinality_density_and_counts():
    assert ml.cardinality(Y) == pytest.approx(5 / 4)  # row sums 2, 1, 2, 0
    assert ml.density(Y) == pytest.approx(5 / 12)
    assert ml.count_distribution(Y).to_dict() == {0: 1, 1: 1, 2: 2}


def test_jaccard_conditional_phi():
    J, C = ml.jaccard(Y), ml.conditional_probability(Y)
    assert J[0, 1] == pytest.approx(1 / 3)  # both=1, union=3
    assert C[0, 1] == pytest.approx(0.5)    # P(B|A)
    assert C[2, 1] == pytest.approx(1.0)    # P(B|C)
    assert ml.phi_matrix(Y)[0, 1] == pytest.approx(0.0)
    const = np.array([[1, 0], [1, 1]], dtype=bool)
    assert np.isnan(ml.phi_matrix(const)[0, 1])  # constant label -> undefined, not a crash


def test_rare_label_sampling_also_inflates_cooccurring_label():
    # Rare label R always co-occurs with common label C.
    Yc = np.zeros((1000, 2), dtype=bool)
    Yc[:500, 1] = True   # C prevalence 0.5
    Yc[:10, 0] = True    # R prevalence 0.01, all R images also carry C
    w = inverse_prevalence_weights(Yc, "max")
    ep = expected_prevalence(Yc, w)
    assert ep[0] > 0.1                       # R strongly boosted
    assert kish_effective_sample_size(w) < 1000
    np.testing.assert_allclose(expected_prevalence(Yc, np.ones(1000)), Yc.mean(axis=0))


@pytest.fixture(scope="module")
def extractor():
    return ConceptExtractor.from_config(load_yaml("iu_concept_lexicon.yaml"))


@pytest.mark.parametrize("text, concept, status", [
    ("No pneumothorax.", "Pneumothorax", "negated"),
    ("There is no focal consolidation, pleural effusion, or pneumothorax.", "Pneumothorax", "negated"),
    ("Small left pleural effusion.", "Pleural Effusion", "positive"),
    ("Pneumonia cannot be excluded.", "Pneumonia", "uncertain"),
    ("Possible right lower lobe atelectasis.", "Atelectasis", "uncertain"),
    ("No interval change in cardiomegaly.", "Cardiomegaly", "positive"),
    ("Pneumothorax is not seen.", "Pneumothorax", "negated"),
    ("Mild pulmonary edema but no effusion.", "Edema", "positive"),
    ("Mild pulmonary edema but no effusion.", "Pleural Effusion", "negated"),
    ("No pericardial effusion.", "Pleural Effusion", "not_mentioned"),
    ("", "Edema", "not_mentioned"),
])
def test_assertion_aware_matching(extractor, text, concept, status):
    assert extractor.report_status(text)[concept] == status
