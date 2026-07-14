"""
Web crawler for Vietnamese sports news sources.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from models.schemas import ArticleSchema

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

def cutoff_for_lookback(lookback_days: int) -> datetime:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    return datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)


def _get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.error("HTTP request failed for %s: %s", url, exc)
        return None


def _parse_date_meta(soup: BeautifulSoup) -> Optional[datetime]:
    """Extract published_at from article:published_time meta tag or date span."""
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        try:
            return datetime.fromisoformat(str(meta["content"]).replace("Z", "+00:00"))
        except ValueError:
            pass

    # VnExpress: <span class="date">Thứ ba, 21/4/2026, 00:00 (GMT+7)</span>
    date_span = soup.find("span", class_="date")
    if date_span:
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", date_span.get_text())
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%d/%m/%Y")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    # Tuoi Tre: "09/07/2026 16:13 GMT+7"
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})\s+GMT\+7", text)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d/%m/%Y %H:%M")
            return dt.replace(tzinfo=timezone(timedelta(hours=7))).astimezone(timezone.utc)
        except ValueError:
            pass

    # fallback: <time datetime="...">
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            return datetime.fromisoformat(str(time_tag["datetime"]).replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def _extract_content(soup: BeautifulSoup) -> str:
    for selector in [
        {"class_": re.compile(r"(fck_detail|article-body|detail-content|content-detail|article__body)", re.I)},
        {"itemprop": "articleBody"},
    ]:
        container = soup.find(["div", "article"], **selector)
        if container:
            for tag in container.find_all(["script", "style", "figure"]):
                tag.decompose()
            return container.get_text(separator=" ", strip=True)
    article_tag = soup.find("article")
    if article_tag:
        return " ".join(p.get_text(strip=True) for p in article_tag.find_all("p") if p.get_text(strip=True))
    return ""


def _crawl_source(
    section_url: str,
    article_url_pattern: str,
    source_name: str,
    base_url: str = "",
    lookback_days: int = 7,
) -> list[ArticleSchema]:
    """Generic crawler for a Vietnamese news sports section."""
    articles: list[ArticleSchema] = []
    cutoff = cutoff_for_lookback(lookback_days)

    try:
        resp = _get(section_url)
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        link_tags = soup.find_all("a", href=re.compile(article_url_pattern))
        logger.info("%s: found %d candidate links", source_name, len(link_tags))

        seen_urls: set[str] = set()
        candidate_urls: list[str] = []
        for tag in link_tags:
            href = tag.get("href", "")
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            # skip comment anchors
            if "#" in href:
                href = href.split("#")[0]
            if href and href not in seen_urls:
                seen_urls.add(href)
                candidate_urls.append(href)

        for url in candidate_urls[:20]:
            try:
                article_resp = _get(url)
                if article_resp is None:
                    continue

                article_soup = BeautifulSoup(article_resp.text, "html.parser")

                title_tag = article_soup.find("h1")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                if not title:
                    continue

                published_at = _parse_date_meta(article_soup)
                if published_at is None:
                    continue
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if published_at < cutoff:
                    continue

                content = _extract_content(article_soup)
                if not content:
                    content = title  # fallback

                articles.append(ArticleSchema(
                    title=title,
                    content=content,
                    source=source_name,
                    url=url,
                    published_at=published_at,
                    category="sports",
                ))
            except requests.RequestException as exc:
                logger.error("%s article fetch failed for %s: %s", source_name, url, exc)

    except requests.RequestException as exc:
        logger.error("%s section fetch failed: %s", source_name, exc)
        return []

    logger.info("%s: collected %d articles", source_name, len(articles))
    return articles


def crawl_vnexpress(lookback_days: int = 7) -> list[ArticleSchema]:
    # VnExpress article URLs: https://vnexpress.net/slug-1234567.html
    return _crawl_source(
        section_url="https://vnexpress.net/the-thao",
        article_url_pattern=r"https://vnexpress\.net/[a-z0-9][a-z0-9\-]+-\d{7}\.html$",
        source_name="VnExpress",
        lookback_days=lookback_days,
    )


def crawl_thanhnien(lookback_days: int = 7) -> list[ArticleSchema]:
    # Thanh Nien: article links are relative paths like /slug-185XXXXXXXXX.htm
    return _crawl_source(
        section_url="https://thanhnien.vn/the-thao/",
        article_url_pattern=r"-185\d{15}\.htm$",
        source_name="Thanh Nien",
        base_url="https://thanhnien.vn",
        lookback_days=lookback_days,
    )


def crawl_tuoitre(lookback_days: int = 7) -> list[ArticleSchema]:
    # Tuoi Tre article IDs can use different numeric prefixes, for example:
    # /slug-100260709160247808.htm
    return _crawl_source(
        section_url="https://tuoitre.vn/the-thao.htm",
        article_url_pattern=r"-\d{15,21}\.htm$",
        source_name="Tuoi Tre",
        base_url="https://tuoitre.vn",
        lookback_days=lookback_days,
    )


def crawl_all_sources(lookback_days: int = 7) -> list[ArticleSchema]:
    results: list[ArticleSchema] = []
    for crawler_fn in (crawl_vnexpress, crawl_thanhnien, crawl_tuoitre):
        try:
            results.extend(crawler_fn(lookback_days=lookback_days))
        except Exception as exc:
            logger.error("Unexpected error in %s: %s", crawler_fn.__name__, exc)
    return results
