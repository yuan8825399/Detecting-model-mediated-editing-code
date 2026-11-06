import numpy as np
import pytest

from provenance.conformal import (class_conditional_sets, quantile_nonconformity,
                                  verdicts, wilson_interval)


def test_quantile_finite_sample_correction():
    scores = np.arange(100, dtype=float)
    # ceil((100 + 1) * 0.9) = 91st smallest value, i.e. index 90
    assert quantile_nonconformity(scores, 0.10) == 90.0
    assert quantile_nonconformity(scores, 0.01) == 99.0


def test_empty_calibration_raises():
    with pytest.raises(ValueError):
        quantile_nonconformity([], 0.1)


def test_unattainable_rank_includes_human_class():
    p_cal = np.r_[np.linspace(0.2, 0.8, 83), np.linspace(0.4, 0.95, 1328)]
    y_cal = np.r_[np.zeros(83, dtype=int), np.ones(1328, dtype=int)]
    _, absent = class_conditional_sets(p_cal, y_cal, [0.0, 0.5, 1.0], alpha=0.01)
    assert absent.all()
    assert np.isinf(quantile_nonconformity(p_cal[:83], 0.01))


def test_verdict_mapping():
    inc_p = np.array([True, False, True, False])
    inc_a = np.array([False, True, True, False])
    assert verdicts(inc_p, inc_a).tolist() == [1, 0, -1, -1]


def _synthetic(n=20000, prevalence=0.94, seed=0):
    """Detector-like scores: both classes score high, humans slightly lower.

    This mirrors the corpus, where the detector assigns a high probability of model
    processing to most texts, including human originals.
    """
    rng = np.random.default_rng(seed)
    n_human = int(round(n * (1 - prevalence)))
    y = np.concatenate([np.ones(n - n_human, dtype=int), np.zeros(n_human, dtype=int)])
    p = np.where(y == 1, rng.uniform(0.90, 1.00, n), rng.uniform(0.70, 0.98, n))
    idx = rng.permutation(n)
    return p[idx], y[idx]


def test_class_conditional_coverage_is_met_per_class():
    p, y = _synthetic()
    cal, test = slice(0, 12000), slice(12000, None)
    inc_p, inc_a = class_conditional_sets(p[cal], y[cal], p[test], alpha=0.10)
    covered = np.where(y[test] == 1, inc_p, inc_a)
    assert covered.mean() >= 0.87
    for cls in (0, 1):
        m = y[test] == cls
        assert covered[m].mean() >= 0.84, (cls, covered[m].mean())


def test_marginal_calibration_can_miss_a_class():
    """Marginal coverage can be met while the human class is never covered.

    This reproduces the failure mode reported in Section 4.9 of the paper: a single
    quantile is fixed by the edited majority, so "absent" is never admitted for a text
    that looks processed even when it is human-written.
    """
    p, y = _synthetic(seed=1)
    s = np.where(y == 1, 1 - p, p)
    q = quantile_nonconformity(s[:12000], 0.10)

    p_te, y_te = p[12000:], y[12000:]
    inc_p = (1 - p_te) <= q
    inc_a = p_te <= q
    covered = np.where(y_te == 1, inc_p, inc_a)

    assert covered.mean() >= 0.88          # marginal guarantee is satisfied
    assert covered[y_te == 0].mean() <= 0.05  # ...while the human class is not

    # the class-conditional rule repairs the human class on the same data
    inc_p2, inc_a2 = class_conditional_sets(p[:12000], y[:12000], p_te, alpha=0.10)
    covered2 = np.where(y_te == 1, inc_p2, inc_a2)
    assert covered2[y_te == 0].mean() >= 0.84


def test_wilson_interval():
    lo, hi = wilson_interval(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert wilson_interval(0, 10)[0] == 0.0
