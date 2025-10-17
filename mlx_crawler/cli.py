"""Command-line entry point for the MLX crawler."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from .config import CrawlConfig, DEFAULT_MODEL_ID
from .crawler import run_crawler
from .markdown import ReaderLMMarkdownGenerator

logger = logging.getLogger("mlx_crawler.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render websites via Playwright and convert them to Markdown using the MLX ReaderLM model.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
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

    if args.verbose:
        successes = len(metrics)
        total_urls = len(args.urls)
        failures = total_urls - successes
        logger.debug(
            "Crawler finished in %.2fs (%d/%d succeeded, %d failed)",
            total_elapsed,
            successes,
            total_urls,
            failures,
        )
        for metric in metrics:
            logger.debug(
                "Timing for %s -> total: %.2fs | inference: %.2fs",
                metric.url,
                metric.total_seconds,
                metric.inference_seconds,
            )


if __name__ == "__main__":
    main()
