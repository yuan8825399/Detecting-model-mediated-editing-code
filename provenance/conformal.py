"""Split conformal prediction helpers used by the audit method (PAL-CP)."""
from __future__ import annotations

import numpy as np


def quantile_nonconformity(scores, alpha: float) -> float:
    """Finite-sample corrected empirical quantile.

    Returns the ceil((n + 1)(1 - alpha))-th smallest score, which is the value required
    for marginal coverage of at least 1 - alpha under exchangeability.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def class_conditional_sets(p_cal, y_cal, p_new, alpha: float = 0.10):
    """Mondrian (class-conditional) split-conformal sets for the presence question.

    Parameters
    ----------
    p_cal : array-like
        Estimated probability of model processing on the calibration partition.
    y_cal : array-like
        1 for edited, 0 for human originals, on the calibration partition.
    p_new : array-like
        Same estimate on the texts to be predicted.

    Returns
    -------
    inc_present, inc_absent : ndarray of bool
        Whether each verdict is admitted to the prediction set. A row with both True is
        an undecidable verdict; a row with both False is an empty set (not observed in
        the study).
    """
    p_cal = np.asarray(p_cal, float)
    y_cal = np.asarray(y_cal, int)
    p_new = np.asarray(p_new, float)

    s_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    q_pos = quantile_nonconformity(s_cal[y_cal == 1], alpha)
    q_neg = quantile_nonconformity(s_cal[y_cal == 0], alpha)
    return (1.0 - p_new) <= q_pos + 1e-12, p_new <= q_neg + 1e-12


def verdicts(inc_present, inc_absent) -> np.ndarray:
    """Map set membership to a three-way verdict: 1 present, 0 absent, -1 undecidable."""
    inc_present = np.asarray(inc_present, bool)
    inc_absent = np.asarray(inc_absent, bool)
    out = np.full(inc_present.shape, -1, dtype=int)
    out[inc_present & ~inc_absent] = 1
    out[~inc_present & inc_absent] = 0
    return out


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))
