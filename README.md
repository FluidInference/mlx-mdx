<p align="left">
  <img src="banner.png" alt="mlx-markdown banner" width="360" height="240">
</p>

# mlx-markdown

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)

`mlx-markdown` modern take for something like `turndown` to easily convert documents, websites into Markdown that's loved by LLMs. `mlx-markdown` is powered by small MLX models that can run on any Apple Sillicon device and its flexible enough to swap out for other MLX models or even Pytorch models. Everything runs locally, so you can customize prompts, swap in different models, and adapt the flow for your publishing targets. Contributions and experiments are welcome—the CLI is intentionally small and hackable.

This will not give you cloud level performances but it gets you maybe 70-80% of the way there for quick use cases.

The resulting Markdown stays LLM-ready—headings, tables, and figure takeaways are explicit—so you can feed it straight into retrieval pipelines or even wire `mlx-mdx` up as an MCP tool to auto-parse documents on demand. This is particularly useful as not all codign agents can read or understand PDFs or deal with raw HTML.

**Before:**

<img src="./mermaid-example.png" alt="Power of AI slide" width="360">

**After:**
    <td width="40%" valign="top">
      <small><small><pre><code class="language-mermaid">graph LR
    A[NVIDIA GPU VRAM] --> B[Scenario 1: Both CUDA and Mac devices have enough memory]
    A --> C[Scenario 2: CUDA device has insufficient memory, but Mac device has enough memory]
    A --> D[Scenario 3: Mac device has enough memory but quite near the capacity while CUDA device has insufficient memory]
    B --> E[Apple Silicon Unified Memory]
    C --> E
    D --> E</code></pre></small></small>

The graph in a PDF becomes structured text that mermaid can render.


## Key Capabilities

- `crawl`: Render websites with Playwright, isolate the readable article, and rewrite it with `mlx-community/jinaai-ReaderLM-v2`.
- `document`: Transcribe PDFs, page images, and photo scans with `mlx-community/Nanonets-OCR2-3B-4bit`.
- Outputs stay portable: assets are downloaded, validated, and relinked alongside YAML-front-matter Markdown.

## Requirements

- macOS on Apple Silicon (MLX requirement).
- Python 3.12 (recommended).
- [uv](https://docs.astral.sh/uv/) 0.4+ for packaging and tooling.
- Playwright Chromium binaries for the `crawl` subcommand (install once with `uvx --from playwright python -m playwright install chromium`).

## Quick Start

Choose the path that fits how you want to run the CLI.

If you're not on [`uv`](https://docs.astral.sh/uv/getting-started/installation/) yet - you're missing out big.

`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Run without installing with `uvx`

```bash
uvx --from git+https://github.com/FluidInference/mlx-mdx.git@v0.0.3 mlx-mdx --help
```

`uvx` (an alias for `uv tool run`) clones the repository into uv's cache, builds it, and launches the `mlx-mdx` entry point—handy for trying the pipelines without installing anything permanently.

### Install as a uv tool

```bash
uv tool install --from git+https://github.com/FluidInference/mlx-mdx.git@v0.0.3 mlx-mdx

uv tool run mlx-mdx -- crawl https://ml-explore.github.io/mlx/build/html/index.html --output output/mlx-docs --verbose

uv tool run mlx-mdx -- document examples/2501.14925v2.pdf --output output/mlx-docs --verbose
```

`uv tool run` ensures the tool executes inside the managed environment even if your shell `PATH` is unaware of `~/.local/bin`. Swap `crawl` for `document` to run the OCR pipeline.

## Usage

The CLI exposes two focused subcommands. For backward compatibility, calling `mlx-mdx <url>` still routes to `crawl`.

### Crawl websites

```bash
uv tool run mlx-mdx -- crawl https://example.com --output output/example --verbose
```

Key options:

- `--output` — destination directory for Markdown and images (default: `./output`).
- `--wait` — seconds to wait after Playwright reports the page is idle (default: `1.0`).
- `--timeout` — navigation timeout in seconds (default: `30`).
- `--model` — MLX model identifier (default: `mlx-community/jinaai-ReaderLM-v2`).
- `--max-html-chars` / `--max-text-chars` — trim limits before passing content to the model.
- `--verbose` — emit detailed logging, including readability decisions.

### OCR documents or images

```bash
uv tool run mlx-mdx -- document examples/2501.14925v2.pdf --output output/docs --verbose
```

Accepts PDFs, standalone images, or directories of page images. Each input becomes `output/documents/<slug>/index.md`.

The VLM also looks at embedded charts and figures. Captions such as “Figure 4 …” are followed by a short `Figure insight:` summary so the Markdown captures the visual takeaway even when the image is absent.

Useful flags:

- `--model` — VLM identifier (default: `mlx-community/Nanonets-OCR2-3B-4bit`).
- `--max-tokens` — limit Markdown length per page (default: `2048`).
- `--temperature` — sampling temperature (default: `0.0`).
- `--pdf-dpi` — render PDFs at this DPI before OCR (default: `200`).
- `--max-image-side` — clamp the longest edge of page images (default: `2048`).
- `--system-prompt` — override the OCR system instructions.
- `--verbose` — emit per-page timings once transcription completes.

## Outputs

Website crawls produce:

```text
<output>/<domain>/<slug>/
  ├─ index.md          # Markdown with YAML front matter
  └─ images/           # Downloaded images (if any)
```

Document transcription produces:

```text
<output>/documents/<slug>/
  └─ index.md          # Markdown assembled from per-page OCR
```

## Examples

Browse `examples/` for sample outputs. For a larger knowledge base that uses these pipelines, see [möbius](https://github.com/FluidInference/mobius) and adapt the prompts or publishing recipes for your own content.

### Work from a local checkout

1. `uv sync` — creates `.venv/` with the runtime dependencies.
2. `uv run python -m playwright install chromium` — downloads the browser used by `crawl`.
3. `uv run mlx-mdx crawl https://ml-explore.github.io/mlx/build/html/index.html --output output/mlx-docs --verbose`
4. _(Optional)_ `uv tool run ty check --python .venv/bin/python` — mirror the CI static type check.

Use `uv run` for development tasks inside the synced virtual environment.

## Pipeline Overview

```text
crawl input (URL)
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

```text
document input (PDFs, images, or directories)
  │
  ▼
Optional PDF rendering with pypdfium2
  │
  ▼
Nanonets OCR (MLX VLM) transcribes each page to structured Markdown
  │
  ▼
Markdown composer stitches pages with metadata and downloads referenced assets
  │
  ▼
Outputs saved to <output>/documents/<slug>/index.md
```

## Operational Notes

- First runs of new models download weights from Hugging Face; subsequent runs reuse the cache.
- Only common web image formats under 10 MB are saved; files smaller than 512 bytes are skipped.
- Remove the output directory manually (`rm -rf output`) to clear prior runs.
- The crawler only processes URLs you provide—it does not follow links or recurse through sites.
- Document OCR relies on [`mlx-vlm`](https://pypi.org/project/mlx-vlm/). PDF rendering uses `pypdfium2`; if it is missing, reinstall with extras or provide page images directly.

## Disclaimer

Please follow each site's terms of service. You are responsible for respecting rate limits and avoiding bans.
