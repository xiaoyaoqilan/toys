"""finurls 财经聚合爬虫。"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from .models import RawArticle
from config.settings import AppConf, SOURCE_REGISTRY


TIME_RE = re.compile(
    r"(\d+)\s*(h|hr|hour|hours?|d|day|days?|w|week|weeks?|m|min|mins?|minute|minutes?|mo|months?|y|year|years?)",
    re.IGNORECASE,
)

FULL_NAMES_RE = re.compile(r"data-site-full-names\s*=\s*(\{.*?\})", re.DOTALL)


def _parse_relative_time(text: str) -> Optional[datetime]:
    if not text:
        return None
    text = text.strip().lower()
    m = TIME_RE.search(text)
    if not m:
        return None
    amount = int(m.group(1))
    unit = m.group(2)[0]
    now = datetime.utcnow()
    if unit in ("s",):
        return now - timedelta(seconds=amount)
    if unit in ("m",):
        return now - timedelta(minutes=amount)
    if unit in ("h",):
        return now - timedelta(hours=amount)
    if unit in ("d",):
        return now - timedelta(days=amount)
    if unit in ("w",):
        return now - timedelta(weeks=amount)
    if unit in ("mo",):
        return now - timedelta(days=amount * 30)
    if unit in ("y",):
        return now - timedelta(days=amount * 365)
    return None


def _load_full_names(html: str) -> dict:
    m = FULL_NAMES_RE.search(html)
    if not m:
        return {}
    try:
        raw = m.group(1)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _normalize_source_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "")


SOURCE_ALIAS = {
    "wiredbusiness": "wired",
    "theeconomist": "economist",
    "wallstreetjournal": "wsj",
    "marketwatch": "marketwatch",
    "forbesmoney": "forbes",
    "reddit/r/finance": "reddit_finance",
    "yahoofinance": "yfinance",
    "yahoofinance": "yfinance",
    "bloomberg": "bloomberg",
    "cnnbusiness": "cnn",
    "reuters": "reuters",
    "businessinsider": "businessinsider",
    "nytimes": "nytimes",
    "theguardian": "guardian",
    "theverge": "theverge",
    "mediumbusiness": "medium",
}


class FinurlsCrawler:
    def __init__(self, conf: AppConf):
        self.conf = conf
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": conf.crawler.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._source_cache = {s.name: s.display for s in SOURCE_REGISTRY}

    def _resolve_source(self, raw_name: str) -> tuple[str, str]:
        key = _normalize_source_slug(raw_name)
        mapped = SOURCE_ALIAS.get(key)
        if mapped:
            return mapped, self._source_cache.get(mapped, raw_name)
        for s in SOURCE_REGISTRY:
            if s.name == key or _normalize_source_slug(s.display) == key:
                return s.name, s.display
        return key, raw_name

    def _fetch_page(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(
                url,
                timeout=(self.conf.crawler.connect_timeout, self.conf.crawler.request_timeout),
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"fetch {url} failed: {e}")
            return None

    def _parse_page(self, html: str, base_url: str) -> List[RawArticle]:
        soup = BeautifulSoup(html, "lxml")
        articles: List[RawArticle] = []
        full_names = _load_full_names(html)

        publisher_blocks = soup.select(".publisher-block[data-publisher]")
        if not publisher_blocks:
            publisher_blocks = soup.select("[data-publisher]")

        for block in publisher_blocks:
            slug = block.get("data-publisher", "")
            display_name = full_names.get(slug, slug.replace("-", " ").title())

            link_items = block.select(".publisher-link") or block.select("[data-id]")
            for item in link_items:
                link = item.select_one("a.article-link") or item.select_one("a[href]")
                if not link:
                    continue
                href = link.get("href", "").strip()
                if not href or href.startswith("#"):
                    continue
                if not href.startswith("http"):
                    continue

                title = link.get_text(strip=True)
                if not title:
                    continue

                time_text = ""
                aside = item.select_one(".aside .text") or item.select_one(".text")
                if aside:
                    time_text = aside.get("title", "") or aside.get_text(strip=True)

                pub_at = _parse_relative_time(time_text) if time_text else None

                source_name, source_display = self._resolve_source(
                    display_name or slug
                )

                articles.append(
                    RawArticle(
                        url=href,
                        title=title,
                        source=source_name,
                        source_display=source_display,
                        published_at=pub_at,
                        raw_html=str(item),
                        metadata={
                            "time_text": time_text,
                            "publisher_slug": slug,
                        },
                    )
                )

        logger.info(f"parsed {len(articles)} articles from {base_url}")
        return articles

    def crawl(self) -> List[RawArticle]:
        all_articles: List[RawArticle] = []
        base = self.conf.crawler.finurls_seed.rstrip("/")
        pages = [base]
        for i in range(2, self.conf.crawler.max_pages + 1):
            pages.append(f"{base}/?paged={i}")

        seen_urls = set()
        for url in pages:
            html = self._fetch_page(url)
            if not html:
                continue
            for art in self._parse_page(html, url):
                if art.url in seen_urls:
                    continue
                seen_urls.add(art.url)
                all_articles.append(art)
            time.sleep(self.conf.crawler.page_interval_sec)

        return all_articles


def crawl_and_fetch_content(articles: List[RawArticle], conf: AppConf) -> List[RawArticle]:
    """可选：对文章正文进行轻量抓取。"""
    session = requests.Session()
    session.headers.update({"User-Agent": conf.crawler.user_agent})
    enriched: List[RawArticle] = []
    for art in articles:
        try:
            resp = session.get(
                art.url,
                timeout=(conf.crawler.connect_timeout, conf.crawler.request_timeout),
            )
            if resp.status_code != 200:
                art.metadata["fetch_status"] = resp.status_code
                enriched.append(art)
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup.select("script,style,nav,header,footer,aside,iframe,noscript"):
                tag.decompose()
            main = (
                soup.select_one("article")
                or soup.select_one('[itemprop="articleBody"]')
                or soup.select_one("main")
                or soup.body
            )
            text = main.get_text("\n", strip=True) if main else ""
            text = re.sub(r"\n{2,}", "\n", text).strip()
            art.content = text[:8000]
            if not art.title and soup.title:
                art.title = soup.title.get_text(strip=True)
            art.metadata["fetched_content"] = True
        except requests.RequestException as e:
            art.metadata["fetch_error"] = str(e)
        enriched.append(art)
    return enriched
