"""Configuration objects and constants for the crawler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "mlx-community/jinaai-ReaderLM-v2"


@dataclass
class CrawlConfig:
    """Top-level settings that control crawling and generation behaviour."""

    output_root: Path
    wait_after_load: float = 1.0
    navigation_timeout: float = 30.0
    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = 2048
    max_html_chars: int = 50_000
    max_text_chars: int = 20_000
