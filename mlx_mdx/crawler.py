"""High-level orchestration for rendering pages and producing Markdown."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import (
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from .config import CrawlConfig
from .content import extract_content
from .images import download_images
from .markdown import (
    ReaderLMMarkdownGenerator,
    compose_markdown,
    replace_image_links,
    strip_code_fence,
)
from .models import PageMetadata
from .utils import slugify

logger = logging.getLogger("mlx_mdx")


@dataclass
class ProcessMetrics:
    """Timing details for a processed URL."""

    url: str
    output_path: Path
    total_seconds: float
    inference_seconds: float
    model_load_seconds: float


def build_output_dir(config: CrawlConfig, metadata: PageMetadata) -> Path:
    """Create an output directory based on the page metadata."""
    parsed = urlparse(metadata.source_url)
    domain = slugify(parsed.netloc or "site", fallback="site")
    title_slug = slugify(metadata.title or parsed.path or "page")
    if not title_slug:
        title_slug = slugify(parsed.path or "page")
    output_dir = config.output_root / domain / title_slug[:80]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def render_page(
    playwright: Playwright,
    url: str,
    config: CrawlConfig,
) -> Tuple[str, str]:
    """Navigate to a URL using Playwright and return the HTML and final URL."""
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    page.set_default_navigation_timeout(config.navigation_timeout * 1000)
    try:
        logger.info("Loading %s", url)
        await page.goto(url, wait_until="networkidle")
        if config.wait_after_load:
            await page.wait_for_timeout(int(config.wait_after_load * 1000))
        html = await page.content()
        final_url = page.url
    finally:
        await browser.close()
    return html, final_url


async def process_url(
    playwright: Playwright,
    url: str,
    config: CrawlConfig,
    generator: ReaderLMMarkdownGenerator,
) -> Optional[ProcessMetrics]:
    """Process a single URL and persist the generated Markdown."""
    overall_start = time.perf_counter()
    try:
        html, final_url = await render_page(playwright, url, config)
    except PlaywrightTimeoutError as exc:
        logger.error("Timeout while loading %s: %s", url, exc)
        return None
    except Exception:  # pylint: disable=broad-except
        logger.exception("Unexpected error loading %s", url)
        return None

    metadata, content_html, plain_text, image_candidates = extract_content(
        html, final_url, config
    )

    output_dir = build_output_dir(config, metadata)
    assets = download_images(image_candidates, output_dir)
    inference_start = time.perf_counter()
    body_markdown = generator.generate_body(
        metadata,
        content_html,
        plain_text,
        assets,
    )
    inference_elapsed = time.perf_counter() - inference_start
    load_elapsed = generator.consume_model_load_seconds()
    body_markdown = strip_code_fence(body_markdown)
    body_markdown = replace_image_links(body_markdown, assets)
    markdown = compose_markdown(metadata, body_markdown, assets)

    output_path = output_dir / "index.md"
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Saved Markdown to %s", output_path)
    total_elapsed = time.perf_counter() - overall_start
    return ProcessMetrics(
        url=url,
        output_path=output_path,
        total_seconds=total_elapsed,
        inference_seconds=inference_elapsed,
        model_load_seconds=load_elapsed,
    )


async def run_crawler(
    urls: List[str],
    config: CrawlConfig,
    generator: ReaderLMMarkdownGenerator,
) -> List[ProcessMetrics]:
    """Render each URL sequentially and feed results to the Markdown generator."""
    metrics: List[ProcessMetrics] = []
    async with async_playwright() as playwright:
        for url in urls:
            result = await process_url(playwright, url, config, generator)
            if result:
                metrics.append(result)
    return metrics
