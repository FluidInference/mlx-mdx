# ML Crawler

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)

A command-line tool that renders web pages with Playwright, extracts the main article content, and converts it to Markdown using the `mlx-community/jinaai-ReaderLM-v2` model. Images referenced in the article are downloaded locally with size and type validation, so the Markdown can be read fully offline.

It's a lightweight tool for our use cases, and you can extend it with VLMs, OCR models, or FluidAudio ASR to process images, PDFs, and speech-to-text. If you need more functionality, please open an issue—or even better, submit a PR.

## Quick Start

### Run without installing

Execute the CLI directly via `uvx`:

```bash
uvx --from git+https://github.com/FluidInference/mlx-crawler.git@v0.0.1 mlx-crawler --help
```

This clones the repository into uv's cache, builds it, runs the `mlx-crawler` entry point, and keeps the build artifacts for future runs.

### Install as a tool

Keep the CLI on your `PATH` with a persistent install:

```bash
uv tool install --from git+https://github.com/FluidInference/mlx-crawler.git@v0.0.1 mlx-crawler
```

Afterward, you can invoke `mlx-crawler` directly.

### Work in a local checkout

For hacking on the codebase or running local checks:

- `uv pip install --python .venv/bin/python .` prepares a virtual environment with runtime dependencies.
- `uv tool run ty check --python .venv/bin/python` mirrors the static type check used in CI.

Before your first crawl, download Playwright's Chromium binary:

```bash
uvx --from playwright python -m playwright install chromium
```

If you are working in a local virtual environment, run `python -m playwright install chromium` from within that environment instead.

## Examples & Knowledge Base

See how we publish MLX-produced Markdown, prompts, and knowledge bases in [möbius](https://github.com/FluidInference/mobius). The repo includes curated examples you can adapt to your own crawls.

## Features

```text
URL input
  │
  ▼
Playwright renders the page in Chromium (handles client-side rendering)
  │
  ▼
Readability extracts the main article HTML
  │
  ▼
ReaderLM (MLX) rewrites the content as clean Markdown (no YAML in generation)
  │
  ▼
Image pipeline downloads, validates, and relinks assets
  │
  ▼
Outputs saved to <output>/<domain>/<slug>/index.md (with optional images/)
```

CLI flags let you tune the model choice, timeouts, token limits, logging, and more.

## Prerequisites

- macOS with Apple Silicon (required for MLX) and Python 3.12 or 3.13 (3.13 recommended)
- A working C compiler toolchain (needed by some dependencies)
- Playwright browser binaries (install once with `python3 -m playwright install chromium`)

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m playwright install chromium
```

## Usage

```bash
uv run mlx-crawler https://ml-explore.github.io/mlx/build/html/index.html --verbose
```

Key options:

- `--output`: Destination directory for Markdown and images (default: `./output`).
- `--wait`: Seconds to wait after Playwright considers the page idle (default: `1.0`).
- `--timeout`: Navigation timeout in seconds (default: `30`).
- `--model`: MLX model identifier (default: `mlx-community/jinaai-ReaderLM-v2`).
- `--max-tokens`: Maximum tokens to generate for the Markdown body (default: `2048`).
- `--max-html-chars` / `--max-text-chars`: Trim limits for the extracted article before sending it to the model.
- `--verbose`: Emit detailed logging, including readability extraction decisions.

## Output Layout

Each processed URL produces:

```bash
<output>/<domain>/<slug>/
  ├─ index.md          # Markdown with YAML front matter
  └─ images/           # Downloaded images (if any)
```

The Markdown front matter records metadata such as the original URL and retrieval time. Any images included in the article reference the downloaded local files.

## Notes

- The first run of a new model may take additional time while weights download from Hugging Face; subsequent runs reuse the cache.
- Only common web image formats under 10 MB are saved; files smaller than 512 bytes are skipped to avoid placeholders.
- Remove the output directory manually (`rm -rf output`) to clear prior crawls.
- The crawler only processes URLs you provide; it does not follow links or perform multi-level site traversal.

## Disclaimer

Please follow each site's terms of service. You are responsible for respecting rate limits and avoiding bans.
