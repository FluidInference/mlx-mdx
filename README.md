# MLX Crawler

A command-line tool that renders web pages with Playwright, extracts the main article content, and converts it to Markdown using the `mlx-community/jinaai-ReaderLM-v2` model. Images referenced in the article are downloaded locally with size and type validation so the Markdown can be read fully offline.

## Features

- Captures web pages in a real Chromium browser (via Playwright) to account for client-side rendering.
- Uses ReaderLM (ported to MLX) to rewrite extracted content as clean Markdown, omitting YAML from the generation step.
- Downloads linked images, validates file type/size, and rewrites image references to local paths.
- Stores each crawl in a structured folder (`<output>/<domain>/<slug>/index.md` with optional `images/`).
- Provides CLI flags for model selection, timeouts, token limits, and verbosity.

## Prerequisites

- macOS with Apple Silicon (required for MLX) and Python 3.9+
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
uv run mlx-crawler https://example.com https://ml-explore.github.io/mlx/build/html/index.html --verbose
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

- First run of a new model may take additional time while weights download from Hugging Face (cached afterwards).
- Only common web image formats under 10 MB are saved; smaller than 512 bytes are skipped to avoid placeholders.
- Removing the output directory between runs must be done manually (`rm -rf output`) if desired.
- Crawls only the URLs you provide; it does not follow links or perform multi-level site traversal.

## Development

**Option 1: Temporary run (no permanent install)**

If you just want to execute the CLI without installing it locally:

```bash
uvx --from git+https://github.com/FluidInference/mlx-crawler.git mlx-crawler --help
```

This clones the repo into uv’s cache, builds it, runs the `mlx-crawler` entry point, and caches it for future use.

**Option 2: Persistent install**

Install the CLI globally through uv and keep it on your `PATH`:

```bash
uv tool install --from git+https://github.com/FluidInference/mlx-crawler.git mlx-crawler
```

After that you can run `mlx-crawler` directly.

For local hacking and type checking:

- `uv pip install --python .venv/bin/python .` provisions a virtual environment with all runtime dependencies.
- `uv tool run ty check --python .venv/bin/python` runs the static type checker (mirrors the CI workflow).

## Disclaimer

Please adhere to TOS for websites, you are responsibel for adherring to rate limits and not getting banned :)
