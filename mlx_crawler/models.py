"""Data models used throughout the crawler pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PageMetadata:
    """Metadata describing the crawled page."""

    source_url: str
    title: Optional[str]
    description: Optional[str]
    byline: Optional[str]


@dataclass
class ImageCandidate:
    """Raw image reference discovered while parsing article content."""

    original_src: str
    absolute_url: str
    alt_text: str


@dataclass
class ImageAsset:
    """Downloaded and validated image asset stored on disk."""

    original_src: str
    absolute_url: str
    alt_text: str
    filename: str
    relative_path: str
