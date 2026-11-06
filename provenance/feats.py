"""Feature extraction: stylometric descriptors + char n-gram vectors."""
import re

import numpy as np


SENT_RE = re.compile(r"[^。；！？]+[。；！？]?")

FW = ["，", "。", "；", "：", "、", "（", "）", "“", "”", "！", "？", "《", "》", "—", "·"]
HW = [",", ".", ";", ":", "(", ")", '"', "'", "<", ">", "-"]

FUNC = list("的了是在和与及其之而也为以对被所者化性度")
MODAL = ["应当", "必须", "可以", "需要", "应该", "不得", "有权", "予以"]
CONN = ["因此", "然而", "但是", "此外", "同时", "从而", "进而", "基于此", "在此基础",
        "值得注意", "具体而言", "换言之", "由此可见", "总之", "综上", "一方面", "另一方面",
        "不仅", "而且", "由于", "尽管", "虽然", "如果", "因而", "故而", "亦", "并", "且"]
BOOST = ["显著", "核心", "关键", "重要", "至关重要", "亦是", "呈现出", "凸显", "彰显",
         "深刻", "全面", "系统", "有效", "充分", "高度", "进一步", "亟需", "亟待", "尤为"]
HEDGE = ["可能", "或许", "似乎", "一定程度上", "相对", "较为", "初步", "大体", "往往", "通常"]
LEGAL = ["法律", "制度", "规范", "权利", "义务", "责任", "立法", "司法", "法理", "法治",
         "刑法", "民法", "宪法", "行政", "程序", "主体", "行为", "关系", "价值", "原则"]


def sentences(t):
    return [s.strip() for s in SENT_RE.findall(t) if s.strip()]


def _rate(t, n):
    return n / max(1, len(t)) * 100.0


def stylo(text):
    """Handcrafted stylometric vector; returns (names, values)."""
    n = len(text)
    s = sentences(text)
    slen = np.array([len(x) for x in s], dtype=float) if s else np.array([0.0])
    chars = [c for c in text]
    uniq = len(set(chars)) / max(1, n)
    from collections import Counter
    cnt = Counter(chars)
    hapax = sum(1 for v in cnt.values() if v == 1) / max(1, len(cnt))
    p = np.array(list(cnt.values()), dtype=float) / max(1, n)
    ent = float(-(p * np.log2(p + 1e-12)).sum())
    fw_n = sum(text.count(c) for c in FW)
    hw_n = sum(text.count(c) for c in HW)
    vals, names = [], []

    def add(name, v):
        names.append(name)
        vals.append(float(v))

    add("log_chars", np.log(max(1, n)))
    add("n_sent", len(s))
    add("mean_slen", slen.mean())
    add("sd_slen", slen.std())
    add("cv_slen", slen.std() / max(1e-6, slen.mean()))
    add("max_slen", slen.max())
    for c in FW:
        add(f"fw_{ord(c)}", _rate(text, text.count(c)))
    for c in HW:
        add(f"hw_{ord(c)}", _rate(text, text.count(c)))
    add("fw_share", fw_n / max(1, fw_n + hw_n))
    add("punct_total", _rate(text, sum(1 for c in text if c in "，。；：、（）“”！？,.;:()\"'-—")))
    add("digits", _rate(text, sum(c.isdigit() for c in text)))
    add("latin", _rate(text, sum(("a" <= c.lower() <= "z") for c in text)))
    add("pct_sign", _rate(text, text.count("%")))
    add("ttr", uniq)
    add("hapax", hapax)
    add("entropy", ent)
    add("max_char_repeat", max((m.group(0).count(m.group(0)[0]) for m in re.finditer(r"(.)\1+", text)), default=0))
    for c in FUNC:
        add(f"f_{ord(c)}", _rate(text, text.count(c)))
    for w in MODAL:
        add(f"modal_{w}", _rate(text, text.count(w)))
    add("conn_total", _rate(text, sum(text.count(w) for w in CONN)))
    add("boost_total", _rate(text, sum(text.count(w) for w in BOOST)))
    add("hedge_total", _rate(text, sum(text.count(w) for w in HEDGE)))
    add("legal_total", _rate(text, sum(text.count(w) for w in LEGAL)))
    for w in CONN[:12]:
        add(f"conn_{w}", _rate(text, text.count(w)))
    for w in BOOST[:10]:
        add(f"boost_{w}", _rate(text, text.count(w)))
    # sentence-initial connective rate
    add("sent_init_conn", np.mean([1.0 if any(x.startswith(w) for w in CONN) else 0.0 for x in s]) if s else 0.0)
    # comma density per sentence
    add("commas_per_sent", (text.count("，") + text.count(",")) / max(1, len(s)))
    add("len_per_clause", n / max(1, text.count("，") + text.count(",") + text.count("；") + len(s)))
    return names, np.array(vals, dtype=np.float32)


FEAT_NAMES = None


def stylo_matrix(texts):
    global FEAT_NAMES
    out = []
    for t in texts:
        names, v = stylo(t)
        FEAT_NAMES = names
        out.append(v)
    return np.vstack(out)


def clean_for_ngram(t):
    t = re.sub(r"[\s\u3000]+", "", t)
    return t
