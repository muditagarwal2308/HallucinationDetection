import math

import numpy as np
import pytest

from halludetect import metrics


def test_auroc_worked_example_from_notes():
    # 3 wrong answers scored {0.9, 0.7, 0.4}, 3 correct ones {0.6, 0.3, 0.2}: 8 of 9 pairs ordered correctly.
    scores = [0.9, 0.7, 0.4, 0.6, 0.3, 0.2]
    labels = [1, 1, 1, 0, 0, 0]
    assert metrics.auroc(scores, labels) == pytest.approx(8 / 9)


def test_auroc_ties_count_half_and_degenerate_labels():
    assert metrics.auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert math.isnan(metrics.auroc([0.1, 0.2], [1, 1]))


def test_auprc_perfect_and_ece_calibrated():
    assert metrics.auprc([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)
    conf = np.array([0.25] * 4 + [0.75] * 4)
    correct = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert metrics.ece(conf, correct) == pytest.approx(0.0)


def test_risk_coverage_and_selective_accuracy():
    risk = [0.1, 0.2, 0.3, 0.9]
    correct = [1, 1, 0, 0]
    cov, r = metrics.risk_coverage_curve(risk, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert r.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert metrics.accuracy_at_coverage(risk, correct, 0.5) == 1.0
    # a perfect ranking reaches the oracle AURAC
    assert metrics.aurac(risk, correct) == pytest.approx(metrics.oracle_aurac(correct))


def test_precision_recall_f1():
    p, r, f = metrics.precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert (p, r, f) == (0.5, 0.5, 0.5)
