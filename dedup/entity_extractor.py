"""金融实体抽取：公司/股票代码/政策/经济指标。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\.(?:US|CN|HK|L|SE)\b")
GENERIC_TICKER_RE = re.compile(r"\b(?:NASDAQ|NYSE|NYSEMKT|AMEX|SGX|HKEX|SSE|SZSE)\s*[:：]?\s*([A-Z]{1,6})\b")
COMPANY_SUFFIX_RE = re.compile(
    r"\b(Inc|Corp|Corporation|LLC|Ltd|Limited|Group|Holdings|Technologies|Technology|Systems|Industries|Industry|Partners|International|Plc|PLC)\b",
    re.IGNORECASE,
)

COMPANY_HINTS = re.compile(
    r"\b("
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"
    r"|"
    r"[A-Z][a-z]+\s+[A-Z][a-z]+"
    r")"
)

POLICY_KEYWORDS = [
    "rate hike", "rate cut", "interest rate", "federal funds rate",
    "inflation", "cpi", "ppi", "gdp", "unemployment", "nonfarm payroll",
    "tariff", "sanction", "embargo", "stimulus", "quantitative easing", "taper",
    "debt ceiling", "fiscal stimulus", "monetary policy",
    "base rate", "policy rate", "fed funds",
]

MACRO_KEYWORDS = [
    "recession", "soft landing", "hard landing", "bull market", "bear market",
    "bond yield", "treasury yield", "credit spread", "usd", "euro", "yen",
    "oil price", "gold price", "copper", "lithium", "semiconductor",
    "supply chain", "global trade", "supply chain disruption",
    "war", "conflict", "pandemic", "election",
]

COMPANY_BLACKLIST = {
    "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Netflix",
    "Nvidia", "AMD", "Intel", "Samsung", "Sony", "TSMC", "Taiwan",
    "JPMorgan", "Goldman", "Morgan Stanley", "Bank of America", "Citi",
    "Wells Fargo", "Visa", "Mastercard", "PayPal",
    "Boeing", "Airbus", "GM", "Ford", "Toyota", "Volkswagen",
    "Walmart", "Target", "Costco", "Home Depot",
}


@dataclass
class EntityExtractionResult:
    companies: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)
    macro: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "companies": self.companies,
            "tickers": self.tickers,
            "policies": self.policies,
            "macro": self.macro,
            "tags": self.tags,
        }


class EntityExtractor:
    def __init__(self):
        self.policy_lower = [p.lower() for p in POLICY_KEYWORDS]
        self.macro_lower = [m.lower() for m in MACRO_KEYWORDS]

    def extract(self, text: str) -> EntityExtractionResult:
        result = EntityExtractionResult()
        if not text:
            return result

        tickers = set(TICKER_RE.findall(text.upper()))
        for m in GENERIC_TICKER_RE.findall(text.upper()):
            tickers.add(m)
        result.tickers = sorted(tickers)

        text_lower = text.lower()
        for kw in self.policy_lower:
            if kw in text_lower:
                result.policies.append(kw)
        for kw in self.macro_lower:
            if kw in text_lower:
                result.macro.append(kw)

        companies = self._extract_companies(text)
        result.companies = sorted(set(companies))

        tags: List[str] = []
        if any(c in text_lower for c in ("earnings", "revenue", "guidance", "eps", "q1", "q2", "q3", "q4")):
            tags.append("earnings")
        if any(c in text_lower for c in ("m&a", "acquisition", "merger", "buyout")):
            tags.append("m&a")
        if any(c in text_lower for c in ("ipo", "shares", "stock", "equity")):
            tags.append("equity")
        if any(c in text_lower for c in ("bond", "debt", "credit", "yield")):
            tags.append("bond")
        if any(c in text_lower for c in ("fed", "ecb", "boj", "pboc", "central bank")):
            tags.append("central_bank")
        if any(c in text_lower for c in ("crypto", "bitcoin", "ethereum", "btc", "eth")):
            tags.append("crypto")
        result.tags = sorted(set(tags))

        return result

    def _extract_companies(self, text: str) -> List[str]:
        found: set = set()

        for word in re.findall(r"[A-Z][A-Za-z\-]{1,}", text):
            if word in COMPANY_BLACKLIST:
                found.add(word)

        for m in COMPANY_SUFFIX_RE.finditer(text):
            start = max(0, m.start() - 40)
            prefix = text[start : m.start()].strip().split()
            if prefix:
                candidate = " ".join(prefix[-2:] + [m.group(0)])
                if len(candidate.split()) <= 4:
                    found.add(candidate)

        for m in COMPANY_HINTS.finditer(text):
            candidate = m.group(0)
            if 2 <= len(candidate) <= 40 and not candidate.startswith(("The", "This", "That", "These", "Those", "After", "Before")):
                if candidate[0].isupper() and any(c.islower() for c in candidate):
                    found.add(candidate)

        return list(found)
