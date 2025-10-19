"""MCP server exposing mlx-mdx crawl/document tools."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import CrawlConfig, DEFAULT_MODEL_ID
from .crawler import run_crawler
from .documents import (
    DocumentConfig,
    DocumentOCRMarkdownGenerator,
    process_document,
)

from .markdown import ReaderLMMarkdownGenerator

logger = logging.getLogger("mlx_mdx.mcp")
logger.setLevel(logging.ERROR)

mcp = FastMCP(name="mlx-mdx")

async def _run_crawl_once(
    url: str,
    config: CrawlConfig,
    generator: ReaderLMMarkdownGenerator,
) -> str:
    metrics = await run_crawler([url], config, generator)
    if not metrics:
        raise RuntimeError(f"Failed to crawl {url}")
    return metrics[0].markdown

@mcp.tool()
async def crawl(
    url: str,
) -> str:
    """Render a web page with Playwright and return MLX-generated Markdown."""

    with tempfile.TemporaryDirectory(prefix="mlx-mdx-crawl-") as tmp_dir:
        output_root = Path(tmp_dir)
        config = CrawlConfig(
            output_root=output_root,
            wait_after_load=1.0,
            navigation_timeout=30.0,
            model_id=DEFAULT_MODEL_ID,
            max_tokens=2048,
            max_html_chars=50_000,
            max_text_chars=20_000,
        )
        generator = ReaderLMMarkdownGenerator(config.model_id, config.max_tokens)
        markdown = await _run_crawl_once(url, config, generator)
    return markdown

@mcp.tool()
async def document(
    path: str,
) -> str:
    """Transcribe a PDF, image, or directory of page images to Markdown."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Document path does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="mlx-mdx-doc-") as tmp_dir:
        output_root = Path(tmp_dir)
        config = DocumentConfig(output_root=output_root)
        generator = DocumentOCRMarkdownGenerator(
            config.model_id,
            config.max_tokens,
            temperature=config.temperature,
            system_prompt=config.system_prompt,
            max_image_side=config.max_image_side,
            figure_system_prompt=config.figure_system_prompt,
            figure_summary_max_tokens=config.figure_summary_max_tokens,
        )
        result = process_document(source, config, generator)
        if result is None:
            raise RuntimeError(f"Failed to transcribe document: {source}")
        markdown = result.markdown
    return markdown

def main() -> None:
    """Entry point for running the MCP server."""
    logging.basicConfig(level=logging.ERROR)
    mcp.run()

if __name__ == "__main__":
    main()
