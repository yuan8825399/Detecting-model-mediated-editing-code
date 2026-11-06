"""Corpus construction and annotation for the LLM-provenance study."""
import json
import re
import sys

import numpy as np
import pandas as pd

from .config import DATA_FILE, PROCESSED_DIR

P = DATA_FILE
OUT = PROCESSED_DIR / "corpus.pkl"

OPS = ["original", "grammar_correction", "academic_polish",
       "sentence_restructure", "abstract_rewrite"]
OP_LABEL = {"original": "H0 human-written", "grammar_correction": "E1 grammar correction",
            "academic_polish": "E2 academic polish", "sentence_restructure": "E3 sentence restructuring",
            "abstract_rewrite": "E4 abstract rewriting"}
LEVEL = {op: i for i, op in enumerate(OPS)}

# ---- subfield rules (title, then abstract fallback), ordered by specificity
RULES = [
    ("criminal", r"刑法|犯罪|罪名|量刑|刑事|刑罚|共犯|未遂|既遂|诈骗|贪污|盗窃|受贿|渎职|走私|毒品|死刑|累犯|自首|缓刑|数罪|罪刑|刑诉|刑事诉讼|检察|侦查|证据|定罪|刑事责任|保安处分|少年司法"),
    ("constitutional", r"宪法|违宪|合宪|基本权利|宪政|全国人大|人民代表大会|立法法|宪法学|依宪|国家机构|民族区域|特别行政区|选举"),
    ("civil_commercial", r"民法|民法典|合同|侵权|物权|债权|婚姻|继承|担保|公司法|商法|商事|破产|保险|票据|知识产权|著作权|专利|商标|证券|信托|合伙|公司|股权|人格权|亲属|遗嘱|不当得利|无因管理|代理|时效|典权|优先权|融资|金融|银行|票据|反垄断|竞争法|消费者|产品责任|电子商务|数据|个人信息|隐私|虚拟财产|AIGC|版权"),
    ("administrative", r"行政|政府|行政处罚|行政许可|行政法|法治政府|监管|规制|治理|公共服务|信息公开|征收|拆迁|应急|环境法|生态|食品安全|公共卫生|教育法|劳动法|社会保障|税收|财政|预算|审计|公务员|编制|执法|城管|信访|社会组织"),
    ("legal_theory_history", r"法理|法学|法律史|法制史|思想|传统|古代|近代|清末|民国|法系|法律移植|法律方法|解释|法律解释|司法|法院|法官|审判|检察|律师|诉讼|程序|正义|权利|法治|合法性|规范|体系|概念|理论|价值|功能|范式|本土|全球化|国际法|WTO|人权|主权|联合国|比较法|实证"),
]
FALLBACK = "legal_theory_history"

def assign_subfield(title, abstract):
    for name, pat in RULES:
        if re.search(pat, title):
            return name
    for name, pat in RULES:
        if re.search(pat, abstract):
            return name
    return "other"

def load():
    with open(P, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    df["level"] = df.operation.map(LEVEL)
    originals = df[df.operation == "original"].set_index("source_id")
    if not originals.index.is_unique:
        raise ValueError("each source must have exactly one original")
    source_fields = pd.Series(
        [assign_subfield(t, a) for t, a in zip(originals.source_title_clean,
                                             originals.abstract_clean)],
        index=originals.index,
    )
    df["subfield"] = df.source_id.map(source_fields)
    if df.subfield.isna().any():
        raise ValueError("an edited record has no corresponding source original")
    yr = df.source_year.astype(int)
    df["era"] = pd.cut(yr, [1999, 2004, 2009, 2014, 2021],
                       labels=["2000-2004", "2005-2009", "2010-2014", "2015-2020"]).astype(str)
    # effective intervention: was any textual change produced at all?
    df["identical"] = (df.output_sha256 == df.input_sha256).astype(int)
    df["effective"] = ((df.intervention_present == 1) & (df.identical == 0)).astype(int)
    # edit magnitude in characters
    source_lengths = df.source_id.map(originals.char_count)
    df["edit_chars"] = np.rint(df.normalized_edit_distance * source_lengths).astype(int)
    df["log_len"] = np.log(df.char_count)
    return df


def effective_presence_mask(df):
    """Retain originals and only edited records that differ from their source."""
    return ~((df.intervention_present == 1) & (df.identical == 1)).to_numpy()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = load()
    df.to_pickle(OUT)

    # integrity checks
    orig = df[df.operation == "original"].set_index("source_id")
    ed = df[df.operation != "original"]
    link = ed.input_sha256 == ed.source_id.map(orig.output_sha256)
    print("edited input == original output:", link.mean().round(4), "n_bad:", (~link).sum())
    print("parent_record_id non-empty:", (df.parent_record_id.fillna("") != "").sum())
    print("records per source:", df.groupby("source_id").size().unique())

    print("\n--- subfield distribution (per source) ---")
    s = orig.subfield.value_counts()
    print(s, "\n", (s / len(orig) * 100).round(1))

    print("\n--- era x subfield (per source) ---")
    print(pd.crosstab(orig.subfield, orig.era))

    print("\n--- effective intervention rate by operation x provider ---")
    piv = ed.pivot_table(index="operation", columns="provider", values="effective", aggfunc="mean")
    print((piv * 100).round(1))

    print("\n--- median edit magnitude (chars) by operation x provider ---")
    print(ed.pivot_table(index="operation", columns="provider", values="edit_chars", aggfunc="median").round(1))

    print("\n--- mean normalized edit distance by operation x provider ---")
    print(ed.pivot_table(index="operation", columns="provider", values="normalized_edit_distance", aggfunc="mean").round(3))

    print("\n--- char_count by operation x provider (mean) ---")
    print(ed.pivot_table(index="operation", columns="provider", values="char_count", aggfunc="mean").round(1))

    print("\n--- split x subfield x model (record counts) ---")
    print(pd.crosstab([df.group_split, df.subfield], df.provider))

    print("\n--- level counts per split ---")
    print(pd.crosstab(df.group_split, df.level))
