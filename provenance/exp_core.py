"""Core experiment utilities: feature builders, detectors, metrics, bootstrap."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, accuracy_score, cohen_kappa_score,
                             roc_curve, brier_score_loss)
from . import feats

STYLO_NAMES = None


def prepare_stylo(df):
    """Compute the stylometric matrix once and cache feature names."""
    global STYLO_NAMES
    M = feats.stylo_matrix(df.abstract_clean.tolist())
    STYLO_NAMES = list(feats.FEAT_NAMES)
    return M


class Detector:
    """kind in {sty, ngr, hyb, bert}. drop = list of stylometric feature names to exclude."""

    def __init__(self, kind, C=1.0, balanced=False, drop=(), use_meta=False):
        self.kind, self.C, self.balanced = kind, C, balanced
        self.drop = tuple(drop)
        self.use_meta = use_meta

    def _sty(self, S):
        if not self.drop:
            return S
        keep = [i for i, n in enumerate(STYLO_NAMES) if n not in self.drop]
        return S[:, keep]

    def _design(self, texts, S, N, fit=False):
        if self.kind == "sty":
            Xs = self._sty(S)
            if fit:
                self.sc = StandardScaler().fit(Xs)
            return self.sc.transform(Xs)
        if self.kind == "ngr":
            return N
        if self.kind == "hyb":
            Xs = self._sty(S)
            if fit:
                self.sc = StandardScaler().fit(Xs)
            from scipy.sparse import csr_matrix, hstack
            Nm = N if hasattr(N, "toarray") else csr_matrix(N)
            return hstack([Nm, csr_matrix(self.sc.transform(Xs))]).tocsr()
        if self.kind == "bert":
            return S
        raise ValueError(self.kind)

    def fit(self, texts, y, S, N=None):
        if self.kind in ("ngr", "hyb"):
            self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                                       sublinear_tf=True, max_features=300000)
            Nt = self.vec.fit_transform([feats.clean_for_ngram(t) for t in texts])
        else:
            Nt = None
        X = self._design(texts, S, Nt, fit=True)
        cw = "balanced" if self.balanced else None
        solv = "liblinear" if len(np.unique(y)) == 2 else "lbfgs"
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight=cw,
                                      solver=solv, random_state=0)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts, S, N=None):
        if self.kind in ("ngr", "hyb"):
            Nt = self.vec.transform([feats.clean_for_ngram(t) for t in texts])
        else:
            Nt = None
        X = self._design(texts, S, Nt, fit=False)
        P = self.clf.predict_proba(X)
        if P.shape[1] == 2:
            return P[:, 1], P
        return None, P

    def predict(self, texts, S, N=None):
        _, P = self.predict_proba(texts, S, N)
        return self.clf.classes_[P.argmax(1)]


# ---------------- metrics ----------------
def hierarchical_level_probabilities(p_present, conditional_levels):
    """Combine two predicted heads for every row, without using ground-truth labels."""
    p_present = np.asarray(p_present, dtype=float)
    conditional_levels = np.asarray(conditional_levels, dtype=float)
    if conditional_levels.shape != (len(p_present), 4):
        raise ValueError("expected four conditional level probabilities for every input")
    return np.column_stack([1 - p_present, p_present[:, None] * conditional_levels])


def tpr_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def binary_report(y, p):
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    prec = float(tp / max(1, pred.sum()))
    return dict(auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                bacc=float(balanced_accuracy_score(y, pred)), f1=float(f1_score(y, pred, zero_division=0)),
                prec=prec, rec=float(tp / max(1, y.sum())),
                tpr_at_5pct=tpr_at_fpr(y, p, 0.05), tpr_at_1pct=tpr_at_fpr(y, p, 0.01),
                ece=ece(y, p), brier=float(brier_score_loss(y, p)),
                pos_rate=float(y.mean()), n=len(y))


def multi_report(y, pred, proba=None, classes=None):
    y = np.asarray(y)
    out = dict(acc=float(accuracy_score(y, pred)),
               mf1=float(f1_score(y, pred, average="macro")),
               wf1=float(f1_score(y, pred, average="weighted")))
    if proba is not None:
        cls = classes if classes is not None else np.unique(y)
        idx = np.searchsorted(np.asarray(cls), y)
        out["logloss"] = float(-np.log(np.clip(proba[np.arange(len(y)), idx], 1e-9, 1)).mean())
    return out


def ordinal_report(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return dict(qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
                mae=float(np.abs(y - pred).mean()),
                adj=float((np.abs(y - pred) <= 1).mean()),
                acc=float(accuracy_score(y, pred)))


def group_bootstrap(y, s, fn, groups, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        try:
            v = fn(y[idx], s[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
