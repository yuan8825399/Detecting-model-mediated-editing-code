import numpy as np

from provenance import exp_core as ec


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert ec.ece(y, p) < 0.02


def test_ece_detects_miscalibration():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)
    assert ec.ece(y, p) > 0.8


def test_tpr_at_fpr_is_monotone_in_fpr():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = 1 / (1 + np.exp(-(1.2 * y + rng.normal(0, 1.0, 4000))))
    t0 = ec.tpr_at_fpr(y, p, 0.00)
    t1 = ec.tpr_at_fpr(y, p, 0.05)
    t2 = ec.tpr_at_fpr(y, p, 0.50)
    t3 = ec.tpr_at_fpr(y, p, 1.00)
    assert t0 <= t1 <= t2 <= t3
    assert t3 == 1.0
    assert t0 < 0.6          # overlapping classes: no perfect separation at zero FPR


def test_tpr_at_fpr_perfect_separation():
    """With separable scores the ROC reaches TPR 1 at FPR 0; the helper must report it."""
    y = np.array([1] * 50 + [0] * 50)
    p = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])
    assert ec.tpr_at_fpr(y, p, 0.0) == 1.0


def test_binary_report_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.6 * y + rng.normal(0, 0.3, 500), 0, 1)
    r = ec.binary_report(y, p)
    for k in ["auc", "pr_auc", "bacc", "f1", "ece", "brier"]:
        assert k in r
    assert 0.7 < r["auc"] < 1.0
def test_hierarchical_predictions_are_normalized_without_true_labels():
    from provenance.exp_core import hierarchical_level_probabilities
    probabilities = hierarchical_level_probabilities(
        np.array([0.95, 0.1]), np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]))
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities.argmax(axis=1).tolist() == [1, 0]

