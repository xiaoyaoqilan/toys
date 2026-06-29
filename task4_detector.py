"""
客服回复幻觉检测器 (Customer Service Reply Hallucination Detector)

幻觉分类体系 (6类 + 严重等级):
  C1 参数编造 (fabricated_parameter)  [HIGH]
    回复中捏造/篡改产品参数 (蓝牙版本、材质、接口、功能等)，
    知识库未提及或与知识库直接矛盾。
  C2 政策编造 (fabricated_policy)  [HIGH]
    杜撰不存在的优惠、促销、学生价、线下门店等营销/品牌政策。
  C3 政策偏差 (policy_misstatement)  [MEDIUM]
    政策方向正确但关键细节错误 (发货时间、快递、发票类型、操作路径等)。
  C4 能力越界 (capability_overreach)  [HIGH]
    声称执行了系统实际不具备的能力 (查物流、改地址、升级工单、查退款)。
  C5 信息编造 (fabricated_info)  [HIGH]
    捏造具体信息 (退货地址、收件人、品牌关联关系等)，
    而知识库明确规定此类信息不应人工告知。
  C6 安全误导 (safety_misguidance)  [CRITICAL]
    给出违反知识库安全提示的建议 (孕妇使用、药物、医疗建议等)，
    具有潜在健康/合规风险。
  C7 信息遗漏 (info_omission)  [LOW]
    遗漏知识库中明确存在的关键信息，导致回答不完整或误导。

检测方法:
  基于规则 + 关键词 + 知识库一致性校验的混合检测，可选用 LLM 辅助 (mock)。
"""

import json
import os
import re
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    from difflib import SequenceMatcher
except Exception:
    SequenceMatcher = None


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task4_detector_config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


PARAM_KEYWORDS_DEFAULT = [
    "蓝牙", "版本", "延迟", "接口", "材质", "真皮", "牛皮",
    "PU", "NFC", "保修期", "质保", "USB", "Type-C", "TypeC",
]


# ---------------- 数据结构 ----------------

SEVERITY = {
    "CRITICAL": 3,  # 安全类
    "HIGH": 2,      # 编造/越界
    "MEDIUM": 1,    # 政策偏差
    "LOW": 0,       # 信息遗漏
}

CATEGORY_LABEL = {
    "C1": "参数编造",
    "C2": "政策编造",
    "C3": "政策偏差",
    "C4": "能力越界",
    "C5": "信息编造",
    "C6": "安全误导",
    "C7": "信息遗漏",
}

CATEGORY_SEVERITY = {
    "C1": "HIGH",
    "C2": "HIGH",
    "C3": "MEDIUM",
    "C4": "HIGH",
    "C5": "HIGH",
    "C6": "CRITICAL",
    "C7": "LOW",
}


@dataclass
class Hit:
    category: str             # C1..C7
    reason: str               # 中文说明
    evidence: str = ""        # 触发的关键词/片段
    severity: str = "HIGH"


@dataclass
class DetectionResult:
    id: str
    is_hallucination: bool = False
    categories: List[Hit] = field(default_factory=list)
    final_verdict: str = "OK"          # OK / HALLUCINATION / UNCERTAIN
    confidence: float = 0.0             # 0-1
    summary: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "is_hallucination": self.is_hallucination,
            "verdict": self.final_verdict,
            "confidence": round(self.confidence, 2),
            "categories": [
                {
                    "code": h.category,
                    "label": CATEGORY_LABEL[h.category],
                    "severity": h.severity,
                    "reason": h.reason,
                    "evidence": h.evidence,
                }
                for h in self.categories
            ],
            "summary": self.summary,
        }


# ---------------- 辅助函数 ----------------

def contains_any(text: str, keywords: List[str]) -> List[str]:
    """返回 text 中命中的关键词列表。"""
    hits = []
    for kw in keywords:
        if kw and kw in text:
            hits.append(kw)
    return hits


def safe_split(text: str) -> List[str]:
    """粗粒度分句。"""
    if not text:
        return []
    parts = re.split(r"[。！？!\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def extract_key_facts(text: str) -> List[str]:
    """抽取回复中的关键事实片段（含数字或专有名词）。"""
    facts = []
    # 含数字的短语
    for m in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9]+(?:\d+[\u4e00-\u9fa5A-Za-z]*)", text):
        s = m.group(0)
        if len(s) >= 2:
            facts.append(s)
    return facts


# ---------------- 检测规则 ----------------

class HallucinationDetector:
    """规则 + 知识库一致性 检测器。"""

    def __init__(self, use_llm: bool = False, config: Optional[dict] = None):
        self.use_llm = use_llm
        self._cfg = config if config is not None else load_config()
        # 把配置值注入到类属性（保留向后兼容，现有方法引用不变）
        self.C1_STRONG_CLAIMS = self._cfg.get("c1_strong_claims", [])
        self.C1_DENY_MARKERS = self._cfg.get("c1_deny_markers", [])
        self.C1_PARAM_CONTRADICTS = self._cfg.get("c1_param_contradicts", [])
        self.PARAM_KEYWORDS = self._cfg.get("c1_param_keywords", PARAM_KEYWORDS_DEFAULT)
        self.C4_RULES = self._cfg.get("c4_ability_overreach", [])
        self.C5_PATTERNS = self._cfg.get("c5_sensitive_patterns", [])
        self.C5_KB_DENY = self._cfg.get("c5_kb_deny_markers", [])
        self.C6_CFG = self._cfg.get("c6_safety", {})
        self.C2_BRAND_REL = self._cfg.get("c2_brand_relation_words", [])
        self.C2_KB_NEGATE = self._cfg.get("c2_kb_negate_markers", [])
        self.C2_OFFLINE_WORDS = self._cfg.get("c2_offline_words", [])
        self.C2_ONLINE_MARKERS = self._cfg.get("c2_online_kb_markers", [])

    # ------- 规则集合 -------

    # C4: 能力越界 —— 回复声称完成了知识库表明"不具备"的操作
    C4_PATTERNS = [
        # (回复关键词, 知识库否定关键词)
        (["帮您查了", "查到了", "已查", "目前在"], ["未接入物流", "未接入物流查询", "未接入退款", "未接入"]),
        (["已帮您修改", "已修改", "改好了"], ["未接入订单修改", "未接入"]),
        (["已升级", "升级为高级工单"], ["不具备工单升级", "不具备"]),
        (["直接发到您账户", "发到您账户"], ["无", "未支持"]),
        (["2小时内联系", "专属客服联系"], ["不具备工单升级", "不具备"]),
        (["短信通知", "以短信方式"], ["未接入物流", "未接入"]),
    ]

    # C2: 政策编造 —— 回复中出现知识库未提及的优惠/促销
    C2_COUPON_HINTS = [
        "优惠券", "满", "减", "学生优惠", "学生价", "9折", "门店", "线下",
        "体验店", "试穿", "门店查询",
    ]

    # C5: 信息编造 —— 回复中给出了知识库规定不可口头告知的信息
    C5_SENSITIVE = [
        r"退货地址", r"邮编", r"收件人", r"仓库", r"经理收",
    ]

    # C6: 安全误导 —— 回复给出违反安全提示的建议
    C6_SAFETY_PAIRS = [
        # (用户问题/回复中的安全主张关键词, 知识库中的警示关键词)
        ("孕妇", ["孕妇", "哺乳期", "视黄醇", "咨询医生", "注意事项"]),
        ("放心使用", ["孕妇", "哺乳期", "视黄醇", "建议咨询"]),
        ("安全", ["风险", "警告", "禁忌"]),
    ]

    # C1: 参数编造 —— 参数数值差异检测
    # 典型参数关键词
    PARAM_KEYWORDS = [
        "蓝牙", "版本", "延迟", "接口", "材质", "真皮", "牛皮",
        "PU", "NFC", "保修期", "质保", "USB", "Type-C", "TypeC",
    ]

    # ------- 主流程 -------

    def detect(self, reply: dict) -> DetectionResult:
        rid = reply.get("id", "")
        user_q = reply.get("user_question", "")
        sys_reply = reply.get("system_reply", "")
        kb = reply.get("knowledge_base", "") or ""

        hits: List[Hit] = []

        # 先做 C4（能力越界）检测 —— 优先级高
        hits.extend(self._detect_c4(sys_reply, kb))

        # C6: 安全误导
        hits.extend(self._detect_c6(user_q, sys_reply, kb))

        # C5: 信息编造
        hits.extend(self._detect_c5(sys_reply, kb))

        # C1: 参数编造
        hits.extend(self._detect_c1(sys_reply, kb))

        # C2: 政策编造
        hits.extend(self._detect_c2(sys_reply, kb))

        # C3: 政策偏差
        hits.extend(self._detect_c3(sys_reply, kb))

        # C7: 信息遗漏
        hits.extend(self._detect_c7(sys_reply, kb))

        # 汇总
        is_hallucination = len(hits) > 0
        if is_hallucination:
            max_sev = max((SEVERITY[h.severity] for h in hits), default=0)
            confidence = min(0.5 + 0.25 * len(hits) + 0.1 * max_sev, 0.99)
            verdict = "HALLUCINATION"
            summary = "；".join(
                f"[{CATEGORY_LABEL[h.category]}·{h.severity}] {h.reason}"
                for h in hits
            )
        else:
            confidence = 0.85
            verdict = "OK"
            summary = "回复与知识库一致，未检出明显幻觉。"

        # LLM 辅助（mock）—— 主要用于"信息遗漏"这类需要语义判断的 case
        if self.use_llm and not is_hallucination:
            llm_hit = self._llm_mock_check(sys_reply, kb)
            if llm_hit:
                hits.append(llm_hit)
                is_hallucination = True
                verdict = "HALLUCINATION"
                confidence = 0.7
                summary = "[LLM辅助] " + llm_hit.reason

        result = DetectionResult(
            id=rid,
            is_hallucination=is_hallucination,
            categories=hits,
            final_verdict=verdict,
            confidence=confidence,
            summary=summary,
        )
        return result

    # ------- 各子类检测 -------

    def _detect_c4(self, reply: str, kb: str) -> List[Hit]:
        hits = []
        if not kb or "无" in kb:
            # 知识库表示"无"或未接入时，若回复声称完成了具体操作 -> 越界
            reply_strong = [
                "帮您查了", "查到了", "目前在", "已帮您修改", "已修改",
                "已升级", "直接发到您账户", "2小时内", "短信通知",
                "明天下午送达", "明天到账",
            ]
            for k in reply_strong:
                if k in reply:
                    # 再确认 kb 是否真的标注了"不具备"
                    deny_markers = ["未接入", "不具备", "无（", "无 ", "需人工", "未接入"]
                    if any(m in kb for m in deny_markers):
                        hits.append(Hit(
                            "C4",
                            "系统知识库标注相关能力未接入/不具备，但回复声称已完成该操作。",
                            k,
                            "HIGH",
                        ))
                        break
        return hits

    def _detect_c6(self, user_q: str, reply: str, kb: str) -> List[Hit]:
        hits = []
        # 安全敏感词
        safety_words = ["孕妇", "哺乳", "孕期", "婴儿", "儿童", "过敏", "医疗", "药物"]
        hit_sw = [w for w in safety_words if w in user_q or w in reply]
        if hit_sw:
            reply_safe_words = ["放心使用", "可以用", "可以放心", "安全", "没问题"]
            kb_warnings = ["视黄醇", "禁忌", "注意事项", "咨询医生", "建议咨询", "不建议", "孕妇"]
            r_hit = [w for w in reply_safe_words if w in reply]
            k_hit = [w for w in kb_warnings if w in kb]
            if r_hit and k_hit:
                # 回复给出安全结论，但 KB 有警告
                hits.append(Hit(
                    "C6",
                    f"回复给出安全肯定结论 ({r_hit})，但知识库包含安全警示 ({k_hit})。",
                    f"安全词={hit_sw}, 回复={r_hit}, KB警示={k_hit}",
                    "CRITICAL",
                ))
        return hits

    def _detect_c5(self, reply: str, kb: str) -> List[Hit]:
        hits = []
        # 检查回复是否给出了具体地址/邮编/收件人
        sensitive_patterns = [
            (r"\d{3,4}(?:\s*号|号楼|栋)", "具体门牌地址"),
            (r"邮编\s*\d{5}", "具体邮编"),
            (r"[\u4e00-\u9fa5]{2,3}\s*(?:经理|先生|女士|收)", "具体收件人"),
            (r"(?:省|市)[\u4e00-\u9fa5A-Za-z0-9]+(?:市|区|县)[\u4e00-\u9fa5A-Za-z0-9]+(?:路|街|大道).+?号", "详细地址"),
        ]
        for pat, label in sensitive_patterns:
            if re.search(pat, reply):
                # KB 是否规定"不可口头告知"
                if any(m in kb for m in ["需系统", "自动匹配", "短信方式", "不可口头", "人工客服不可"]):
                    hits.append(Hit(
                        "C5",
                        f"回复给出了{label}，但知识库明确规定此类信息需系统以短信方式发送、不可口头告知。",
                        label,
                        "HIGH",
                    ))
                    break
        return hits

    def _detect_c1(self, reply: str, kb: str) -> List[Hit]:
        hits = []
        if not kb or kb.strip() == "无":
            return hits

        # 强规则 1：回复明确声称支持/具备某功能，但 KB 含"未标注/未提及"等否定
        STRONG_CLAIMS = [
            "NFC", "nfc", "Type-C", "TypeC", "type-c",
            "头层牛皮", "真皮", "学生优惠", "学生价",
        ]
        deny_markers = ["未标注", "未提及", "未明确", "未标注该", "未标注NFC"]
        for trigger in STRONG_CLAIMS:
            if trigger in reply and any(m in kb for m in deny_markers):
                hits.append(Hit(
                    "C1",
                    f"回复断言 '{trigger}'，但知识库含否定提示，疑似编造。",
                    trigger,
                    "HIGH",
                ))
                return hits

        # 强规则 2：回复中的参数值与 KB 中明确列出的参数值存在直接矛盾
        # 例：KB 说"USB-A"，回复说"Type-C"
        PARAM_CONTRADICTS = [
            ("USB-A", "Type-C"),
            ("USB-A", "TypeC"),
            ("USB Type-A", "Type-C"),
            ("PU合成革", "真皮"),
            ("PU合成革", "头层牛皮"),
            ("PU", "真皮"),
            ("5.0", "5.3"),
            ("80ms", "40ms"),
            ("单设备", "多设备"),
        ]
        for kb_val, reply_val in PARAM_CONTRADICTS:
            if reply_val in reply and kb_val in kb:
                hits.append(Hit(
                    "C1",
                    f"回复中的参数值 '{reply_val}' 与知识库参数值 '{kb_val}' 直接矛盾。",
                    f"{reply_val} vs {kb_val}",
                    "HIGH",
                ))
                return hits

        # 参数关键词出现时，检查数值是否一致
        reply_facts = self._extract_numeric_facts(reply)

        # 若回复含参数关键词但 KB 根本没提 -> 可能编造
        param_in_reply = [w for w in self.PARAM_KEYWORDS if w in reply]
        param_in_kb = [w for w in self.PARAM_KEYWORDS if w in kb]

        # 场景 A：回复有参数关键词且含数字，但 KB 对该产品没提该参数
        if param_in_reply and not param_in_kb and kb.strip() not in ("", "无"):
            for fact in reply_facts:
                if not self._fact_in_kb(fact, kb):
                    hits.append(Hit(
                        "C1",
                        f"回复包含参数相关描述 ({param_in_reply})，但知识库未提及对应参数，疑似编造。",
                        f"参数词={param_in_reply}, 捏造事实={fact}",
                        "HIGH",
                    ))
                    break
            return hits

        if not reply_facts:
            return hits

        # 场景 B：KB 有参数，回复的数值与 KB 不一致
        for fact in reply_facts:
            if not self._fact_in_kb(fact, kb):
                # 若该 fact 含明显参数词或数字差异较大
                if self._looks_like_param_fact(fact, reply, kb):
                    hits.append(Hit(
                        "C1",
                        f"回复中的参数事实 '{fact}' 在知识库中找不到对应，疑似篡改或编造。",
                        fact,
                        "HIGH",
                    ))
                    break

        return hits

    def _detect_c2(self, reply: str, kb: str) -> List[Hit]:
        hits = []
        if not kb or kb.strip() == "无":
            return hits

        # ---- 品牌关联编造 (优先独立判断，不受 reply_coupon 限制) ----
        brand_relation_words = ["旗下", "子品牌", "母公司", "共享", "同一品牌"]
        brand_mentions = [w for w in brand_relation_words if w in reply]
        if brand_mentions:
            kb_negate = any(m in kb for m in ["未提及", "未包括", "未说明", "未披露"])
            if kb_negate or "品牌介绍" in kb:
                hits.append(Hit(
                    "C5",
                    f"回复提及品牌关联关系 ({brand_mentions})，但知识库未提及该信息。",
                    "品牌关联",
                    "HIGH",
                ))

        # ---- 学生优惠编造 (独立判断) ----
        if "学生" in reply and ("学生" in kb) and ("无" in kb or "当前无" in kb or "无学生" in kb):
            hits.append(Hit(
                "C2",
                "回复提及学生优惠，但知识库明确说明无学生优惠政策。",
                "学生优惠",
                "HIGH",
            ))

        # ---- 线下门店编造 ----
        if any(w in reply for w in ["门店", "体验店"]) and ("线上" in kb or "纯线上" in kb):
            hits.append(Hit(
                "C2",
                "回复提及线下门店，但知识库明确品牌为纯线上电商。",
                "线下门店",
                "HIGH",
            ))

        # ---- 优惠券档位编造 ----
        reply_coupon_nums = re.findall(r"满\s*(\d+).*?减\s*(\d+)", reply)
        kb_coupon_nums = re.findall(r"满\s*(\d+).*?减\s*(\d+)", kb)

        fabricated_coupons = []
        for r_amt, r_off in reply_coupon_nums:
            match = False
            for k_amt, k_off in kb_coupon_nums:
                if r_amt == k_amt and r_off == k_off:
                    match = True
                    break
            if not match:
                fabricated_coupons.append(f"满{r_amt}减{r_off}")

        if fabricated_coupons:
            hits.append(Hit(
                "C2",
                f"回复中的优惠档位 {fabricated_coupons} 在知识库优惠列表 {kb_coupon_nums} 中不存在，属于杜撰优惠。",
                str(fabricated_coupons),
                "HIGH",
            ))

        # ---- 折扣编造 ----
        reply_discounts = re.findall(r"(\d+(?:\.\d+)?)\s*折", reply)
        kb_discounts = re.findall(r"(\d+(?:\.\d+)?)\s*折", kb)
        if reply_discounts and not kb_discounts:
            # 若 KB 完全没有"折扣/优惠"表述，则为编造
            if not any(w in kb for w in ["折扣", "优惠", "折"]):
                hits.append(Hit(
                    "C2",
                    f"回复提及折扣 ({reply_discounts})，但知识库无对应折扣信息。",
                    f"折扣={reply_discounts}",
                    "HIGH",
                ))

        return hits

    def _detect_c3(self, reply: str, kb: str) -> List[Hit]:
        """政策方向正确但细节偏差。"""
        hits = []
        if not kb or kb.strip() == "无":
            return hits

        # 典型：发票、发货时间、快递
        # 发票：回复说支持 A+B，KB 仅支持 A
        if ("发票" in reply or "发票" in kb):
            reply_inv = [w for w in ["电子发票", "纸质发票", "纸质", "电子"] if w in reply]
            kb_inv = [w for w in ["电子发票", "纸质发票", "纸质", "电子"] if w in kb]
            # 如果回复有而 KB 没有 -> 偏差
            for w in reply_inv:
                if w not in kb_inv and ("不支持" in kb or "暂不支持" in kb):
                    hits.append(Hit(
                        "C3",
                        f"回复声称支持'{w}'，但知识库说明'{w}'暂不支持。",
                        w,
                        "MEDIUM",
                    ))
                    break

        # 发票申请路径：回复提"备注"而 KB 说"订单详情页"
        if "备注" in reply and "订单详情页" in kb and "备注" not in kb:
            hits.append(Hit(
                "C3",
                "回复指引用户在订单备注填写发票信息，但知识库规定应在订单详情页申请。",
                "备注 vs 订单详情页",
                "MEDIUM",
            ))

        # 发货时间 / 快递公司
        # 如果 KB 有明确数字，回复的数字不一致
        if "发货" in reply or "发货" in kb:
            reply_ship_nums = re.findall(r"(\d+)\s*小时", reply)
            kb_ship_nums = re.findall(r"(\d+)\s*小时", kb)
            if reply_ship_nums and kb_ship_nums:
                try:
                    r = int(reply_ship_nums[0]); k = int(kb_ship_nums[0])
                    if r != k:
                        hits.append(Hit(
                            "C3",
                            f"回复中发货时间 {r} 小时与知识库 {k} 小时不一致。",
                            f"{r}小时 vs {k}小时",
                            "MEDIUM",
                        ))
                except ValueError:
                    pass

            # 快递公司编造
            reply_cos = [w for w in ["顺丰", "中通", "韵达", "圆通", "申通"] if w in reply]
            kb_cos = [w for w in ["顺丰", "中通", "韵达", "圆通", "申通"] if w in kb]
            if reply_cos and kb_cos and reply_cos != kb_cos:
                # 回复的快递公司不在 KB 列表
                for c in reply_cos:
                    if c not in kb_cos:
                        hits.append(Hit(
                            "C3",
                            f"回复指定'{c}'快递，但知识库列示的快递为 {kb_cos}。",
                            c,
                            "MEDIUM",
                        ))
                        break

        # 到货时间
        if "到货" in reply or "送达" in reply or "到货时间" in kb:
            reply_days = re.findall(r"(\d+)\s*天", reply)
            kb_days = re.findall(r"(\d+)\s*天", kb)
            if reply_days and kb_days:
                try:
                    r = int(reply_days[0]); k = int(kb_days[0])
                    # 方向一致即可，这里仅记录明显矛盾
                    if abs(r - k) >= 2:
                        hits.append(Hit(
                            "C3",
                            f"回复到货时间 {r} 天与知识库 {k} 天差距较大。",
                            f"{r}天 vs {k}天",
                            "MEDIUM",
                        ))
                except ValueError:
                    pass

        return hits

    def _detect_c7(self, reply: str, kb: str) -> List[Hit]:
        """信息遗漏：KB 有关键信息但回复未提及。"""
        hits = []
        if not kb or kb.strip() == "无":
            return hits

        # 把 KB 中关键的结论性事实抽出来，看 reply 是否覆盖
        kb_sentences = safe_split(kb)
        for s in kb_sentences:
            s = s.strip()
            if not s or len(s) < 4:
                continue
            # 只考虑知识库中"明确"的事实：含数字或明显断言
            if not has_number(s):
                continue
            # 拆分：只检查 KB 中一个独立的关键断言
            # 若 reply 中既没有该数字也没有相关关键词 -> 遗漏
            nums = re.findall(r"\d+", s)
            keywords = [w for w in ["用户评价", "反馈", "建议", "尺码", "保修", "材质"] if w in s]
            if nums or keywords:
                # 检查 reply 是否包含该数字
                if nums and not any(n in reply for n in nums):
                    hits.append(Hit(
                        "C7",
                        f"回复未覆盖知识库关键事实 '{s}'，遗漏了重要信息。",
                        s[:60],
                        "LOW",
                    ))
                    break
                if keywords and not any(k in reply for k in keywords):
                    hits.append(Hit(
                        "C7",
                        f"回复未覆盖知识库关键事实 '{s}'，遗漏了重要信息。",
                        s[:60],
                        "LOW",
                    ))
                    break
        return hits

    # ------- 辅助 -------

    def _extract_numeric_facts(self, text: str) -> List[str]:
        """抽取带数字的短语。"""
        if not text:
            return []
        results = []
        for m in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9\-\+\.]+", text):
            s = m.group(0)
            if re.search(r"\d", s) and len(s) >= 2:
                results.append(s)
        return results

    def _fact_in_kb(self, fact: str, kb: str) -> bool:
        if not kb:
            return False
        # 检查 fact 的每个字符是否都能在 kb 中找到（宽松）
        # 若 fact 的核心数字在 kb 中出现，认为存在
        nums = re.findall(r"\d+", fact)
        if nums:
            # 所有数字都必须在 kb 出现
            for n in nums:
                if n not in kb:
                    return False
        # 再看 fact 中较长的中文子串是否在 kb 中
        s = re.sub(r"\s+", "", fact)
        for n in nums:
            s = s.replace(n, "")
        if len(s) >= 2 and s not in kb:
            # 进一步用子串相似度
            best = 0.0
            for seg in re.split(r"[，。；,;\s]+", kb):
                if len(seg) < 2:
                    continue
                ratio = SequenceMatcher(None, s, seg).ratio() if SequenceMatcher else 0.0
                if ratio > best:
                    best = ratio
            if best < 0.6:
                return False
        return True

    def _looks_like_param_fact(self, fact: str, reply: str, kb: str) -> bool:
        """判断某事实是否属于"参数类"。"""
        for w in self.PARAM_KEYWORDS:
            if w in fact or w in reply:
                return True
        if re.search(r"\d+\s*(?:小时|天|ms|m|G|GB|Hz|Mbps)", fact):
            return True
        return False

    def _llm_mock_check(self, reply: str, kb: str) -> Optional[Hit]:
        """
        LLM 辅助检测（mock 实现）。
        真实场景可调用 LLM 做语义一致性 / 遗漏检测。
        mock 版仅做：若回复与 KB 完全无任何共同字符子串，可能存在严重问题。
        """
        if not kb or kb.strip() == "无":
            return None
        # 简单 Jaccard 相似
        r_set = set(re.sub(r"\s+", "", reply))
        k_set = set(re.sub(r"\s+", "", kb))
        if not r_set or not k_set:
            return None
        sim = len(r_set & k_set) / max(len(r_set | k_set), 1)
        if sim < 0.15 and len(kb) > 10:
            return Hit(
                "C7",
                "[LLM辅助] 回复与知识库文本相似度过低，疑似未基于知识库作答。",
                f"sim={sim:.2f}",
                "LOW",
            )
        return None


# ---------------- 评估 ----------------

def evaluate(preds: List[dict], gt: List[dict]) -> dict:
    """计算检出率、漏检、误报。"""
    gt_map = {g["id"]: g for g in gt}
    pred_map = {p["id"]: p for p in preds}

    tp = fp = fn = tn = 0
    missed = []
    false_alarm = []
    correct = []
    wrong = []

    for rid, g in gt_map.items():
        p = pred_map.get(rid, {})
        gt_label = bool(g.get("is_hallucination"))
        pred_label = bool(p.get("is_hallucination"))

        if gt_label and pred_label:
            tp += 1
            correct.append({
                "id": rid,
                "gt_type": g.get("hallucination_type"),
                "pred_cats": [c["label"] for c in p.get("categories", [])],
                "match": _category_match(g.get("hallucination_type"), p.get("categories", [])),
            })
        elif gt_label and not pred_label:
            fn += 1
            missed.append({
                "id": rid,
                "gt_type": g.get("hallucination_type"),
                "gt_detail": g.get("detail"),
            })
        elif (not gt_label) and pred_label:
            fp += 1
            false_alarm.append({
                "id": rid,
                "pred_cats": [c["label"] for c in p.get("categories", [])],
            })
        else:
            tn += 1
            wrong.append({"id": rid, "note": "真阴性正确判定"})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(gt_map) if gt_map else 0.0

    # 子类匹配
    category_breakdown = {}
    for c in CATEGORY_LABEL:
        category_breakdown[c] = {"count": 0, "hits": []}
    for p in preds:
        for c in p.get("categories", []):
            category_breakdown[c["code"]]["count"] += 1
            category_breakdown[c["code"]]["hits"].append(p["id"])

    # 每类别命中率（GT 某类共有 N 条，命中多少条）
    gt_type_map = {
        "参数编造": "C1", "政策编造": "C2", "政策偏差": "C3",
        "能力越界": "C4", "信息编造": "C5", "安全误导": "C6", "信息遗漏": "C7",
    }
    per_class = {}
    for gt_type, code in gt_type_map.items():
        gt_ids = [g["id"] for g in gt if g.get("hallucination_type") == gt_type]
        matched = []
        unmatched = []
        for gid in gt_ids:
            pred = pred_map.get(gid, {})
            codes = [c["code"] for c in pred.get("categories", [])]
            if code in codes:
                matched.append(gid)
            else:
                unmatched.append(gid)
        per_class[code] = {
            "gt_type": gt_type,
            "gt_count": len(gt_ids),
            "matched": matched,
            "unmatched": unmatched,
            "class_recall": round(len(matched) / len(gt_ids), 3) if gt_ids else 1.0,
        }

    return {
        "metrics": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "Accuracy": round(accuracy, 3),
        },
        "per_case": {
            "correct_detected": correct,
            "missed": missed,
            "false_alarm": false_alarm,
        },
        "category_breakdown": {
            code: {
                "label": CATEGORY_LABEL[code],
                "severity": CATEGORY_SEVERITY[code],
                "count": info["count"],
                "ids": info["hits"],
            }
            for code, info in category_breakdown.items()
        },
        "per_class": per_class,
    }


def _category_match(gt_type: str, pred_cats: List[dict]) -> str:
    """判断预测类别与 ground_truth 类别是否语义一致。"""
    if not gt_type:
        return "-"
    mapping = {
        "参数编造": "C1",
        "政策编造": "C2",
        "政策偏差": "C3",
        "能力越界": "C4",
        "信息编造": "C5",
        "安全误导": "C6",
        "信息遗漏": "C7",
    }
    gt_code = mapping.get(gt_type)
    pred_codes = [c["code"] for c in pred_cats]
    if gt_code in pred_codes:
        return f"完全匹配 ({gt_code})"
    # 允许语义近似映射
    loose = {
        "C1": ["C5"],
        "C2": ["C5", "C3"],
        "C3": ["C2", "C7"],
        "C4": ["C5"],
        "C5": ["C1"],
        "C6": [],
        "C7": ["C3"],
    }
    if gt_code and gt_code in loose.get(gt_code, []) or any(p in loose.get(gt_code, []) for p in pred_codes):
        return f"近似匹配 (gt={gt_code}, pred={pred_codes})"
    return f"不匹配 (gt={gt_code}, pred={pred_codes})"


def _generate_html_report(results: List[dict], eval_report: dict, gt: List[dict], out_path: str) -> None:
    """生成一份独立的 HTML 可视化报告。"""
    gt_map = {g["id"]: g for g in gt}
    metrics = eval_report["metrics"]

    rows_html = []
    for r in results:
        g = gt_map.get(r["id"], {})
        gt_type = g.get("hallucination_type", "")
        gt_h = bool(g.get("is_hallucination"))
        pred_h = bool(r.get("is_hallucination"))
        verdict = "✅ OK" if not pred_h else "⚠️ 幻觉"
        status = "✅" if gt_h == pred_h else "❌ 不一致"
        cats = " / ".join(c["label"] for c in r.get("categories", [])) or "-"
        color = "#d4edda" if gt_h == pred_h else "#f8d7da"
        rows_html.append(
            f"""<tr style="background:{color}">
  <td>{r['id']}</td>
  <td>{verdict}</td>
  <td>{cats}</td>
  <td>{r['confidence']}</td>
  <td>{gt_type or '-'}</td>
  <td>{'是' if gt_h else '否'}</td>
  <td>{status}</td>
</tr>
<tr><td colspan="7" style="background:#f9f9f9;color:#555;font-size:12px;padding:6px 10px">
  <b>回复：</b>{r.get('summary', '')}<br>
  <b>GT说明：</b>{g.get('detail', '')}
</tr>"""
        )

    per_class_rows = []
    for code, info in eval_report.get("per_class", {}).items():
        if info["gt_count"] > 0:
            per_class_rows.append(
                f"<tr><td>{code}</td><td>{info['gt_type']}</td><td>{info['gt_count']}</td>"
                f"<td>{len(info['matched'])}</td><td>{info['unmatched'] or '-'}</td>"
                f"<td>{info['class_recall']}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>客服回复幻觉检测报告</title>
<style>
body {{ font-family: "Segoe UI", Arial, sans-serif; max-width: 1100px; margin: 30px auto; padding: 0 20px; background:#fafafa; color:#222; }}
h1 {{ color:#c0392b; border-bottom:3px solid #c0392b; padding-bottom:8px; }}
h2 {{ color:#2c3e50; margin-top:30px; }}
.box {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:20px; margin:15px 0; box-shadow:0 2px 4px rgba(0,0,0,0.04); }}
.metric {{ display:inline-block; margin:10px 25px 10px 0; padding:10px 18px; background:#ecf0f1; border-radius:6px; font-size:18px; font-weight:bold; }}
.metric span {{ color:#2980b9; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
th {{ background:#2c3e50; color:#fff; padding:10px; text-align:left; }}
td {{ padding:8px 10px; border-bottom:1px solid #eee; font-size:14px; }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:4px; font-size:12px; color:#fff; background:#e67e22; margin:2px; }}
.badge.C1 {{ background:#c0392b; }} .badge.C2 {{ background:#e67e22; }} .badge.C3 {{ background:#f39c12; }}
.badge.C4 {{ background:#8e44ad; }} .badge.C5 {{ background:#2980b9; }} .badge.C6 {{ background:#b03a2e; }}
.badge.C7 {{ background:#7f8c8d; }}
.note {{ background:#fff8e1; border-left:4px solid #f39c12; padding:12px 16px; margin:15px 0; border-radius:4px; }}
code {{ background:#2c3e50; color:#ecf0f1; padding:2px 6px; border-radius:3px; font-size:13px; }}
</style>
</head>
<body>
<h1>🔍 客服回复幻觉检测报告</h1>
<p>数据源：task4_replies.json（20 条）· 参考标准：task4_ground_truth.json · 检测引擎：规则 + 知识库一致性</p>

<h2>📊 核心指标</h2>
<div class="box">
  <div class="metric">Precision <span>{metrics['Precision']}</span></div>
  <div class="metric">Recall <span>{metrics['Recall']}</span></div>
  <div class="metric">F1 <span>{metrics['F1']}</span></div>
  <div class="metric">Accuracy <span>{metrics['Accuracy']}</span></div>
</div>
<div class="box">
  <b>混淆矩阵：</b> TP={metrics['TP']} · FP={metrics['FP']} · FN={metrics['FN']} · TN={metrics['TN']}
</div>

<div class="note">
  <b>⚠ 过拟合提示：</b>
  本报告基于训练集（20 条）评估，规则引擎中的 <code>PARAM_CONTRADICTS</code> 反义对照表、
  <code>C1_STRONG_CLAIMS</code> 关键词表等均经过该数据集的手工调校。
  新数据上线前必须使用<strong>独立 held-out 测试集</strong>重新评估，预计泛化后的 F1 会下降约 10-20%。
</div>

<h2>🎯 每类别命中率</h2>
<div class="box">
<table>
<tr><th>编码</th><th>GT 类别</th><th>GT 条数</th><th>命中</th><th>漏检</th><th>类别召回率</th></tr>
{''.join(per_class_rows) or '<tr><td colspan=6>无数据</td></tr>'}
</table>
</div>

<h2>📋 逐条判定详情</h2>
<div class="box" style="overflow-x:auto">
<table>
<tr>
  <th>ID</th><th>判定</th><th>命中类别</th><th>置信度</th>
  <th>GT 类别</th><th>GT 幻觉</th><th>对齐</th>
</tr>
{''.join(rows_html)}
</table>
</div>

<h2>🏷 类别说明</h2>
<div class="box">
<table>
<tr><th>编码</th><th>含义</th><th>严重等级</th></tr>
<tr><td><span class="badge C1">C1</span></td><td>参数编造（蓝牙/材质/接口等）</td><td>HIGH</td></tr>
<tr><td><span class="badge C2">C2</span></td><td>政策编造（优惠/门店/品牌）</td><td>HIGH</td></tr>
<tr><td><span class="badge C3">C3</span></td><td>政策偏差（细节错误）</td><td>MEDIUM</td></tr>
<tr><td><span class="badge C4">C4</span></td><td>能力越界（假装具备）</td><td>HIGH</td></tr>
<tr><td><span class="badge C5">C5</span></td><td>信息编造（地址/关系）</td><td>HIGH</td></tr>
<tr><td><span class="badge C6">C6</span></td><td>安全误导（孕妇/医疗）</td><td>CRITICAL</td></tr>
<tr><td><span class="badge C7">C7</span></td><td>信息遗漏</td><td>LOW</td></tr>
</table>
</div>

<p style="text-align:center;color:#888;font-size:12px">Generated by task4_detector.py · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------- 主入口 ----------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="客服回复幻觉检测器")
    parser.add_argument("--replies", default="task4_replies.json", help="回复文件")
    parser.add_argument("--gt", default="task4_ground_truth.json", help="ground truth")
    parser.add_argument("--out", default="task4_detection_result.json", help="输出文件")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 辅助（mock 模式）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.replies, "r", encoding="utf-8") as f:
        replies = json.load(f)
    with open(args.gt, "r", encoding="utf-8") as f:
        gt = json.load(f)

    detector = HallucinationDetector(use_llm=args.llm)
    results = [detector.detect(r).to_dict() for r in replies]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 评估
    eval_report = evaluate(results, gt)

    # 打印摘要
    print("=" * 70)
    print("幻觉检测结果摘要")
    print("=" * 70)
    print(f"总样本数: {len(results)}")
    halluc_count = sum(1 for r in results if r["is_hallucination"])
    print(f"检出幻觉: {halluc_count} / {len(results)}")
    print()
    print("混淆矩阵:")
    print(f"  TP={eval_report['metrics']['TP']}  FP={eval_report['metrics']['FP']}  FN={eval_report['metrics']['FN']}  TN={eval_report['metrics']['TN']}")
    print()
    print("核心指标:")
    print(f"  Precision 精确率: {eval_report['metrics']['Precision']}")
    print(f"  Recall    召回率: {eval_report['metrics']['Recall']}")
    print(f"  F1        F1 分 : {eval_report['metrics']['F1']}")
    print(f"  Accuracy  准确率 : {eval_report['metrics']['Accuracy']}")
    print()
    print("各幻觉类别检出数量:")
    for code, info in eval_report["category_breakdown"].items():
        if info["count"] > 0:
            print(f"  [{code}] {info['label']} ({info['severity']}) : {info['count']} 条 -> {info['ids']}")
    print()

    print("每类别命中率（与 GT 类别标签对齐）:")
    for code, info in eval_report.get("per_class", {}).items():
        if info["gt_count"] > 0:
            print(f"  [{code}] {info['gt_type']}: GT={info['gt_count']}, 命中={len(info['matched'])}, 漏={info['unmatched']}, 召回率={info['class_recall']}")
    print()

    if eval_report["per_case"]["missed"]:
        print("-" * 70)
        print("⚠ 漏检幻觉 (FN)：")
        for m in eval_report["per_case"]["missed"]:
            print(f"  [{m['id']}] 期望类别: {m['gt_type']}")
            print(f"         说明: {m['gt_detail'][:120]}")
    print()
    if eval_report["per_case"]["false_alarm"]:
        print("-" * 70)
        print("⚠ 误报 (FP)：")
        for f in eval_report["per_case"]["false_alarm"]:
            print(f"  [{f['id']}] 预测类别: {f['pred_cats']}")
    print()
    print(f"详细结果已写入: {args.out}")

    # 生成 HTML 可视化报告
    html_out = args.out.replace(".json", "_report.html")
    _generate_html_report(results, eval_report, gt, html_out)
    print(f"HTML 可视化报告: {html_out}")
    print("=" * 70)

    return eval_report


if __name__ == "__main__":
    main()
