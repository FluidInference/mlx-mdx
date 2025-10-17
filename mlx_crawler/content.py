"""HTML extraction and metadata parsing utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from readability import Document

from .config import CrawlConfig
from .models import ImageCandidate, PageMetadata

_MIN_PLAINTEXT_CHARS = 200


def _clean_content(soup: BeautifulSoup, strip_chrome: bool = False) -> BeautifulSoup:
    """Remove noisy tags while keeping relevant article markup."""
    for tag in soup(["script", "style", "noscript", "form"]):
        tag.decompose()
    if strip_chrome:
        for tag in soup(["header", "footer", "nav", "aside"]):
            tag.decompose()
    return soup


def _join_plain_text(soup: BeautifulSoup) -> str:
    """Join text nodes for summary generation."""
    return "\n".join(s for s in soup.stripped_strings)


def _iter_primary_candidates(soup_full: BeautifulSoup) -> Iterable[BeautifulSoup]:
    """Yield progressively broader content scopes to fall back on."""
    for selector in ("main", "article"):
        candidate = soup_full.select_one(selector)
        if candidate:
            yield BeautifulSoup(str(candidate), "html.parser")
    if soup_full.body:
        yield BeautifulSoup(str(soup_full.body), "html.parser")


def extract_content(html: str, final_url: str, config: CrawlConfig):
    """Extract article content, metadata, and candidate images from HTML."""
    document = Document(html)
    soup_full = BeautifulSoup(html, "html.parser")

    summary_html = document.summary(html_partial=True)
    summary = BeautifulSoup(summary_html, "html.parser")
    summary = _clean_content(summary)

    plain_text = _join_plain_text(summary)
    summary_has_images = bool(summary.find("img"))
    full_has_images = bool(soup_full.find("img"))

    if (
        len(plain_text) < _MIN_PLAINTEXT_CHARS
        or (full_has_images and not summary_has_images)
    ):
        for candidate in _iter_primary_candidates(soup_full):
            candidate = _clean_content(candidate, strip_chrome=True)
            candidate_plain = _join_plain_text(candidate)
            candidate_has_images = bool(candidate.find("img"))
            if (
                len(candidate_plain) >= _MIN_PLAINTEXT_CHARS
                or (full_has_images and candidate_has_images)
            ):
                summary = candidate
                plain_text = candidate_plain
                summary_has_images = candidate_has_images
                break

    if len(plain_text) > config.max_text_chars:
        plain_text = plain_text[: config.max_text_chars] + "\n[truncated]"

    content_html = summary.decode()
    if len(content_html) > config.max_html_chars:
        content_html = content_html[: config.max_html_chars] + "\n<!-- truncated -->"

    title = document.short_title()
    if not title and soup_full.title and soup_full.title.string:
        title = soup_full.title.string.strip()

    description: Optional[str] = None
    description_tag = soup_full.find("meta", attrs={"name": "description"})
    if description_tag and description_tag.get("content"):
        description = description_tag["content"].strip()

    byline: Optional[str] = None
    author_tag = soup_full.find("meta", attrs={"name": "author"})
    if author_tag and author_tag.get("content"):
        byline = author_tag["content"].strip()

    image_candidates: List[ImageCandidate] = []
    for img in summary.find_all("img"):
        src = img.get("src")
        if not src or src.startswith("data:"):
            continue
        abs_url = urljoin(final_url, src)
        alt_text = img.get("alt", "").strip()
        image_candidates.append(ImageCandidate(src, abs_url, alt_text))

    metadata = PageMetadata(
        source_url=final_url,
        title=title,
        description=description,
        byline=byline,
    )
    return metadata, content_html, plain_text, image_candidates
