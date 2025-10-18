"""Command-line entry point for the MLX crawler."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from .config import CrawlConfig, DEFAULT_MODEL_ID
from .crawler import run_crawler
from .documents import (
    DEFAULT_OCR_SYSTEM_PROMPT,
    DocumentConfig,
    DocumentOCRMarkdownGenerator,
    process_document,
)
from .markdown import ReaderLMMarkdownGenerator

logger = logging.getLogger("mlx_mdx.cli")


def _ensure_command_prefix(argv: Sequence[str], commands: Iterable[str]) -> Sequence[str]:
    if not argv:
        return argv
    first = argv[0]
    if first in commands or first.startswith("-"):
        return argv
    return ("crawl", *argv)


def _add_crawl_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("urls", nargs="+", help="One or more URLs to capture")
    parser.add_argument(
        "--output",
        default="output",
        type=Path,
        help="Directory where Markdown and assets should be written",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="Seconds to wait after network idle before reading HTML",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Navigation timeout in seconds",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="MLX model identifier to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate for the Markdown body",
    )
    parser.add_argument(
        "--max-html-chars",
        type=int,
        default=50_000,
        help="Trim extracted HTML to this many characters before sending to the model",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=20_000,
        help="Trim extracted plain text to this many characters before sending to the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Emit crawler output to STDOUT as Markdown (suitable for MCP)",
    )


def _add_document_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Document PDFs, standalone images, or directories of page images to process",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=Path,
        help="Directory where Markdown and assets should be written",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Nanonets-OCR2-3B-4bit",
        help="MLX VLM identifier to use for OCR",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate per page",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the OCR model",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="Render PDFs at this DPI before OCR (default: 200).",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=2048,
        help="Resize images so the longest edge is at most this many pixels before OCR",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the default OCR system prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render websites via Playwright or transcribe document images to Markdown using MLX models."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser(
        "crawl", help="Render web pages and convert them to Markdown"
    )
    _add_crawl_arguments(crawl_parser)

    document_parser = subparsers.add_parser(
        "document", help="Transcribe documents with the Nanonets OCR model"
    )
    _add_document_arguments(document_parser)

    argv = list(sys.argv[1:] if argv is None else argv)
    argv = list(_ensure_command_prefix(argv, subparsers.choices.keys()))
    return parser.parse_args(argv)


def _run_crawl(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.verbose else logging.INFO
    if args.mcp and not args.verbose:
        level = logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    config = CrawlConfig(
        output_root=Path(args.output).resolve(),
        wait_after_load=args.wait,
        navigation_timeout=args.timeout,
        model_id=args.model,
        max_tokens=args.max_tokens,
        max_html_chars=args.max_html_chars,
        max_text_chars=args.max_text_chars,
    )

    generator = ReaderLMMarkdownGenerator(config.model_id, config.max_tokens)
    overall_start = time.perf_counter()
    metrics = asyncio.run(run_crawler(args.urls, config, generator))
    total_elapsed = time.perf_counter() - overall_start
    
    successes = len(metrics)
    total_urls = len(args.urls)
    failures = total_urls - successes
    logger.info(
        "Finished in %.2fs (%d/%d succeeded, %d failed)",
        total_elapsed,
        successes,
        total_urls,
        failures,
    )
    
    if args.verbose:
        for metric in metrics:
            logger.debug(
                "Timing for %s -> total: %.2fs | inference: %.2fs | model_load: %.2fs",
                metric.url,
                metric.total_seconds,
                metric.inference_seconds,
                metric.model_load_seconds,
            )

    if args.mcp:
        for idx, metric in enumerate(metrics):
            markdown = metric.markdown
            if idx:
                if not markdown.startswith("\n"):
                    sys.stdout.write("\n")
            sys.stdout.write(markdown if markdown.endswith("\n") else markdown + "\n")
        sys.stdout.flush()
        return


def _run_documents(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    config = DocumentConfig(
        output_root=Path(args.output).resolve(),
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt or DEFAULT_OCR_SYSTEM_PROMPT,
        pdf_dpi=args.pdf_dpi,
        max_image_side=args.max_image_side,
    )

    generator = DocumentOCRMarkdownGenerator(
        config.model_id,
        config.max_tokens,
        temperature=config.temperature,
        system_prompt=config.system_prompt,
        max_image_side=config.max_image_side,
        figure_system_prompt=config.figure_system_prompt,
        figure_summary_max_tokens=config.figure_summary_max_tokens,
    )

    overall_start = time.perf_counter()
    results = []
    for path in args.paths:
        result = process_document(path, config, generator)
        if result:
            results.append(result)
    total_elapsed = time.perf_counter() - overall_start

    if args.verbose:
        successes = len(results)
        total_inputs = len(args.paths)
        failures = total_inputs - successes
        logger.debug(
            "Document OCR finished in %.2fs (%d/%d succeeded, %d failed)",
            total_elapsed,
            successes,
            total_inputs,
            failures,
        )
        for res in results:
            logger.debug(
                "Processed %s -> %s (pages=%d, elapsed=%.2fs)",
                res.source_path,
                res.output_path,
                res.page_count,
                res.total_seconds,
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "crawl":
        _run_crawl(args)
    else:
        _run_documents(args)


if __name__ == "__main__":
    main()
