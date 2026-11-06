"""Single-annotator validation of the rule-based subfield classifier.
The rule labels are compared against the first author's manual labels for a
stratified sample of 66 titles (12 per subfield, all 6 for 'other')."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
o = df[df.operation == "original"].reset_index(drop=True)

parts = [g.sample(min(len(g), 12), random_state=7) for _, g in o.groupby("subfield")]
samp = pd.concat(parts).sample(frac=1, random_state=11).reset_index(drop=True)

# Manual labels, assigned by reading each title (order matches the printed sample).
manual = [
    "criminal",            # 00 监狱行刑的功能与目的
    "legal_theory_history",  # 01 人与文化和法——中西法律文化交流
    "criminal",            # 02 论假想防卫过当
    "constitutional",      # 03 备案审查的跨程序拘束力研究
    "civil_commercial",    # 04 完善民事强制执行立法
    "criminal",            # 05 刑事审判监督程序
    "constitutional",      # 06 宪法的全球化
    "legal_theory_history",  # 07 简牍所见秦县少吏研究
    "criminal",            # 08 刑事诉讼证明标准
    "civil_commercial",    # 09 公司法两权分离
    "legal_theory_history",  # 10 法理即法律原理之解说
    "constitutional",      # 11 宪法中公共利益
    "constitutional",      # 12 人格尊严的宪法意义
    "administrative",      # 13 公私法汇合与行政法演进
    "constitutional",      # 14 司法解释权能与宪法实施
    "criminal",            # 15 刑事政策对刑法理论的影响
    "administrative",      # 16 行政诉讼和行政复议
    "other",               # 17 与投资有关的贸易措施
    "administrative",      # 18 行政罚款设定方式
    "constitutional",      # 19 法律的法规创造力
    "civil_commercial",    # 20 董事会角色、职能
    "administrative",      # 21 社保筹资分配机制
    "civil_commercial",    # 22 公司道德的法律化
    "civil_commercial",    # 23 论商法
    "legal_theory_history",  # 24 法治体系形成指标
    "criminal",            # 25 知识产权刑事司法解释
    "administrative",      # 26 社会组织及其法制化
    "other",               # 27 高新科技园区立法促进
    "criminal",            # 28 检察机关刑事审判监督
    "criminal",            # 29 刑法立法感言
    "civil_commercial",    # 30 金融法治重构
    "other",               # 31 外国国家豁免的仲裁例外
    "civil_commercial",    # 32 物权法的绿色思考
    "administrative",      # 33 政府规制中的信用工具
    "administrative",      # 34 行政主体研究
    "administrative",      # 35 公众参与与行政法
    "criminal",            # 36 刑事陪审
    "civil_commercial",    # 37 居住权
    "civil_commercial",    # 38 会计师第三人责任
    "legal_theory_history",  # 39 立法社会效果考量
    "civil_commercial",    # 40 市场竞争法基础
    "criminal",            # 41 刑事诉讼当事人辅助制度
    "criminal",            # 42 刑法立法阻却事由
    "administrative",      # 43 行政争议实质性解决
    "administrative",      # 44 行政诉讼合目的性审查
    "legal_theory_history",  # 45 司法改革中的法律职业
    "constitutional",      # 46 香港司法权的宪政维度
    "administrative",      # 47 英国行政法上的听证
    "criminal",            # 48 测谎结论与证据规则
    "legal_theory_history",  # 49 繁荣法学研究
    "administrative",      # 50 个人所得税纳税义务
    "other",               # 51 国家保护责任
    "constitutional",      # 52 西方宪政危机
    "criminal",            # 53 检警关系
    "constitutional",      # 54 宪法秩序
    "civil_commercial",    # 55 脱离人体的器官
    "administrative",      # 56 算法个性化定价规制
    "constitutional",      # 57 五四宪法
    "civil_commercial",    # 58 政府投资企业的物权
    "civil_commercial",    # 59 商事信用
    "legal_theory_history",  # 60 中国律师业发展
    "civil_commercial",    # 61 侵权责任法体系
    "constitutional",      # 62 作为根本法的宪法
    "civil_commercial",    # 63 归责原则与侵权责任方式
    "criminal",            # 64 共同故意中的认识因素
    "constitutional",      # 65 有限的宪法典
]
assert len(manual) == len(samp), (len(manual), len(samp))
samp["manual"] = manual
samp["agree"] = (samp.manual == samp.subfield).astype(int)

acc = float(samp.agree.mean())
kappa = float(cohen_kappa_score(samp.subfield, samp.manual))
print(f"n = {len(samp)}; agreement = {acc:.3f}; Cohen's kappa = {kappa:.3f}")
print("\nper-stratum agreement:")
print(samp.groupby("subfield").agree.agg(["mean", "sum", "count"]).round(3).to_string())

cm = pd.crosstab(samp.subfield, samp.manual)
cm.to_csv(os.path.join(OUT, "subfield_validation.csv"), encoding="utf-8-sig")
print("\nrule (rows) vs manual (cols):")
print(cm.to_string())

# how many misclassifications cross a major boundary (e.g. criminal vs civil)
print("\nmisclassified titles:")
for _, r in samp[samp.agree == 0].iterrows():
    print(f"  rule={r.subfield:22s} manual={r.manual:22s} | {r.source_title_clean}")

R = dict(n=len(samp), agreement=acc, kappa=kappa,
         per_stratum=samp.groupby("subfield").agree.mean().round(3).to_dict(),
         n_errors=int((samp.agree == 0).sum()))
json.dump(R, open(os.path.join(OUT, "subfield_validation.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved subfield_validation.json")
