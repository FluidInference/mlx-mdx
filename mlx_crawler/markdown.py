"""Markdown generation helpers backed by the MLX ReaderLM model."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, List

from mlx_lm import generate as generate_text
from mlx_lm import load as load_model

from .models import ImageAsset, PageMetadata

logger = logging.getLogger("mlx_crawler")


class ReaderLMMarkdownGenerator:
    """Thin wrapper around the MLX ReaderLM model for Markdown generation."""

    def __init__(self, model_id: str, max_tokens: int) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_model(self) -> None:
        if self._model is None or self._tokenizer is None:
            logger.info("Loading model %s", self.model_id)
            self._model, self._tokenizer = load_model(self.model_id)

    def generate_body(
        self,
        metadata: PageMetadata,
        content_html: str,
        plain_text: str,
        images: List[ImageAsset],
    ) -> str:
        """Produce Markdown body text from raw article content."""
        self._ensure_model()

        system_prompt = (
            "You convert rendered HTML from web articles into precise Markdown suitable for offline reading. "
            "Return only the Markdown body without any YAML front matter. Preserve headings, lists, tables, "
            "inline emphasis, code blocks, and hyperlinks. Reference images only if they appear relevant. "
            "If you include images, use the provided local path values exactly. Avoid adding commentary about the task."
        )

        content_sections: List[str] = [
            f"Source URL: {metadata.source_url}",
        ]

        if metadata.title:
            content_sections.append(f"Page Title: {metadata.title}")
        if metadata.description:
            content_sections.append(f"Description: {metadata.description}")
        if metadata.byline:
            content_sections.append(f"Byline: {metadata.byline}")

        if content_html:
            content_sections.append("Main content HTML:\n" + content_html)
        if plain_text:
            content_sections.append("Extracted plain text:\n" + plain_text)

        if images:
            manifest_lines = [
                "Image manifest (use relative paths in Markdown if needed):"
            ]
            for idx, asset in enumerate(images, start=1):
                alt = asset.alt_text or "(no alt text provided)"
                manifest_lines.append(
                    f"{idx}. alt: {alt} | local_path: {asset.relative_path} | source: {asset.absolute_url}"
                )
            content_sections.append("\n".join(manifest_lines))
        else:
            content_sections.append("No downloadable images were captured.")

        content_sections.append(
            "Return only the Markdown body content for the article above."
        )

        user_content = "\n\n".join(content_sections)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.debug("Prompt length: %s characters", len(prompt))
        result = generate_text(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return result.strip()


def replace_image_links(markdown: str, assets: List[ImageAsset]) -> str:
    """Swap remote image URLs with downloaded asset paths."""
    if not assets:
        return markdown
    updated = markdown
    for asset in assets:
        updated = updated.replace(asset.absolute_url, asset.relative_path)
        if asset.original_src:
            updated = updated.replace(asset.original_src, asset.relative_path)
    return updated


def strip_code_fence(markdown: str) -> str:
    """Remove a surrounding Markdown code fence if the model adds one."""
    text = markdown.strip()
    if not text.startswith("```"):
        return markdown.strip()

    lines = text.splitlines()
    fence_info = lines[0].strip()
    if not fence_info.startswith("```"):
        return text

    closing_index = None
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].strip().startswith("```"):
            closing_index = idx
            break

    if closing_index is None:
        return text

    inner = "\n".join(lines[1:closing_index]).strip()
    return inner


def compose_markdown(metadata: PageMetadata, body: str, assets: List[ImageAsset]) -> str:
    """Generate final Markdown including front matter."""
    timestamp = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    front_matter_lines = ["---"]
    if metadata.title:
        front_matter_lines.append(f"title: {metadata.title}")
    front_matter_lines.append(f"source_url: {metadata.source_url}")
    front_matter_lines.append(f"retrieved_at: {timestamp}")
    if metadata.description:
        front_matter_lines.append(f"description: {metadata.description}")
    if metadata.byline:
        front_matter_lines.append(f"byline: {metadata.byline}")
    if assets:
        image_files = [asset.relative_path for asset in assets]
        files_str = "[" + ", ".join(image_files) + "]"
        front_matter_lines.append(f"images: {files_str}")
    front_matter_lines.append("---\n")

    return "\n".join(front_matter_lines) + body.strip() + "\n"
