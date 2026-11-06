import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import json, sys, re, collections, difflib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

P = DATA_FILE
df = pd.DataFrame([json.loads(l) for l in open(P, encoding="utf-8") if l.strip()])
df.to_pickle(PROCESSED_DIR / "df.pkl")
print(df.shape)

# ---- E1 identity by provider
e1 = df[df.operation == "grammar_correction"]
print("\nE1 identical to input, by provider:")
print(e1.assign(idn=(e1.output_sha256 == e1.input_sha256)).groupby("provider").idn.agg(["mean", "sum", "count"]))

print("\nE1 n_char_changed (from edit distance*len):")
e1 = e1.assign(chg=(e1.normalized_edit_distance * e1.char_count).round())
print(e1.groupby("provider").chg.describe()[["mean", "50%", "max"]])

# ---- punctuation profile
FULL = "，。；：？！“”‘’（）【】"
HALF = ",.;:?!\"'()[]"
def punct_profile(t):
    return (sum(t.count(c) for c in "，"), sum(t.count(c) for c in ","))

rows = []
for _, r in df.iterrows():
    fw = r.abstract_clean.count("，"); hw = r.abstract_clean.count(",")
    rows.append((fw, hw))
df["fw_comma"] = [a for a, b in rows]
df["hw_comma"] = [b for a, b in rows]
print("\ncount of abstracts containing halfwidth comma, by operation:")
print(df.assign(has_hw=df.hw_comma > 0).groupby("operation").has_hw.mean().round(3))

# ---- sentence / char stats
df["n_sent"] = df.abstract_clean.str.count(r"[。；！？]")
df["n_punct"] = df.abstract_clean.str.count(r"[，,。；：、（）()“”\"]")
print("\nmean sentence count by operation:")
print(df.groupby("operation")[["n_sent", "n_punct", "char_count"]].mean().round(2))
print("\nmean sentence length by operation:")
print((df.char_count / df.n_sent.clip(lower=1)).groupby(df.operation).mean().round(2))

# ---- character-level edit ops on a subsample (paired with original)
print("\n--- char-level diff composition (subsample 60 groups) ---")
sub_groups = sorted(df.group_id.unique())[:60]
sub = df[df.group_id.isin(sub_groups)]
orig = sub[sub.operation == "original"].set_index("source_id").abstract_clean
comp = []
for _, r in sub[sub.operation != "original"].iterrows():
    a, b = orig.get(r.source_id), r.abstract_clean
    if a is None: continue
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = rep = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert": ins += j2 - j1
        elif tag == "delete": dele += i2 - i1
        elif tag == "replace": rep += max(i2 - i1, j2 - j1)
    comp.append((r.operation, r.provider, ins, dele, rep, len(a)))
cd = pd.DataFrame(comp, columns=["operation", "provider", "ins", "del", "rep", "L"])
cd["tot"] = cd[["ins", "del", "rep"]].sum(1)
print(cd.groupby("operation")[["ins", "del", "rep"]].mean().round(1))
print("\nprovider effect on total edits (all ops):")
print(cd.groupby("provider").tot.mean().round(1))

# ---- quick separability probe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

tr = df[df.group_split == "train"]; te = df[df.group_split == "test"]
print("\ntrain", tr.shape, "test", te.shape)

def probe(ycol, mask_tr, mask_te, name, **kw):
    Xtr_txt = tr[mask_tr].abstract_clean; Xte_txt = te[mask_te].abstract_clean
    ytr = tr[mask_tr][ycol]; yte = te[mask_te][ycol]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_txt); Xte = vec.transform(Xte_txt)
    clf = LogisticRegression(max_iter=2000, C=4.0, **kw).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out = dict(name=name, n_tr=len(ytr), n_te=len(yte), acc=accuracy_score(yte, pred),
               f1=f1_score(yte, pred, average="macro"))
    if len(np.unique(ytr)) == 2:
        out["auc"] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(out)
    return clf, vec

# binary presence
m_tr = np.ones(len(tr), bool); m_te = np.ones(len(te), bool)
probe("intervention_present", m_tr, m_te, "binary presence")

# binary excluding exact copies
m_tr = (tr.operation != "original") | True
e1copy = (tr.operation == "grammar_correction") & (tr.output_sha256 == tr.input_sha256)
m_tr = ~e1copy
e1copy_te = (te.operation == "grammar_correction") & (te.output_sha256 == te.input_sha256)
m_te = ~e1copy_te
probe("intervention_present", m_tr, m_te, "binary presence (drop identical E1)")

# 4-class operation
probe("operation", np.ones(len(tr), bool), np.ones(len(te), bool), "4-class operation")
