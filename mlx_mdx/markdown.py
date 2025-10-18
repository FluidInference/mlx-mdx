"""Markdown generation helpers backed by the MLX ReaderLM model."""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from mlx_lm import batch_generate as batch_generate_text
from mlx_lm import load as load_model

from .models import ImageAsset, PageMetadata

logger = logging.getLogger("mlx_mdx")


@dataclass
class MarkdownRequest:
    """Payload used for batch Markdown generation."""

    metadata: PageMetadata
    content_html: str
    plain_text: str
    images: List[ImageAsset]


class ReaderLMMarkdownGenerator:
    """Thin wrapper around the MLX ReaderLM model for Markdown generation."""

    _ALLOW_PATTERNS = [
        "*.json",
        "model*.safetensors",
        "*.py",
        "tokenizer.model",
        "*.tiktoken",
        "tiktoken.model",
        "*.txt",
        "*.jsonl",
        "*.jinja",
    ]

    def __init__(self, model_id: str, max_tokens: int) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_load_seconds: float = 0.0
        self._model_load_reported = False
        self._resolved_load_target: Optional[str] = None

    def _resolve_env_override(self) -> Optional[Path]:
        override = os.getenv("MODEL_DIR")
        if not override:
            return None
        override_path = Path(override).expanduser()
        if override_path.exists():
            logger.debug("MODEL_DIR override detected at %s", override_path)
            return override_path
        logger.warning(
            "MODEL_DIR is set to %s but the path does not exist; falling back to %s",
            override_path,
            self.model_id,
        )
        return None

    def _resolve_snapshot_path(self, repo_id: str) -> Path:
        from huggingface_hub import snapshot_download

        try:
            local_path = Path(
                snapshot_download(
                    repo_id,
                    local_files_only=True,
                    allow_patterns=self._ALLOW_PATTERNS,
                )
            )
            logger.info(
                "Resolved cached model for %s at %s (local_files_only=True)",
                repo_id,
                local_path,
            )
            return local_path
        except Exception as err:  # noqa: BLE001 - propagate diagnostics
            logger.info(
                "Local cache for %s was not found (%s); attempting snapshot download.",
                repo_id,
                err,
            )
            local_path = Path(
                snapshot_download(repo_id, allow_patterns=self._ALLOW_PATTERNS)
            )
            logger.debug("Downloaded model %s to %s", repo_id, local_path)
            return local_path

    def _determine_load_target(self) -> str:
        if self._resolved_load_target:
            return self._resolved_load_target

        env_override = self._resolve_env_override()
        if env_override:
            self._resolved_load_target = str(env_override)
            return self._resolved_load_target

        candidate = Path(self.model_id).expanduser()
        if candidate.exists():
            logger.debug(
                "Model identifier %s resolves to local path %s",
                self.model_id,
                candidate,
            )
            self._resolved_load_target = str(candidate)
            return self._resolved_load_target

        snapshot_path = self._resolve_snapshot_path(self.model_id)
        self._resolved_load_target = str(snapshot_path)
        return self._resolved_load_target

    def _ensure_model(self) -> None:
        if self._model is None or self._tokenizer is None:
            load_target = self._determine_load_target()
            if load_target != self.model_id:
                logger.info(
                    "Loading model %s (resolved from %s)",
                    load_target,
                    self.model_id,
                )
            else:
                logger.info("Loading model %s", load_target)
            start = time.perf_counter()
            self._model, self._tokenizer = load_model(load_target)
            self._model_load_seconds = time.perf_counter() - start
            self._model_load_reported = False
            logger.debug(
                "Loaded model from %s in %.2fs",
                load_target,
                self._model_load_seconds,
            )

    def consume_model_load_seconds(self) -> float:
        """Return the model load time once per load event."""
        if not self._model_load_reported and self._model_load_seconds:
            self._model_load_reported = True
            return self._model_load_seconds
        return 0.0

    def _build_prompt(
        self,
        metadata: PageMetadata,
        content_html: str,
        plain_text: str,
        images: List[ImageAsset],
    ) -> List[int]:
        """Render a tokenized chat prompt for the ReaderLM model."""
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

        return self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

    def _generate_from_prompts(self, prompts: Sequence[List[int]]) -> List[str]:
        """Generate Markdown bodies from prepared token sequences."""
        logger.debug("Running batch generation for %d prompt(s)", len(prompts))
        batch_result = batch_generate_text(
            self._model,
            self._tokenizer,
            prompts=list(prompts),
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return [text.strip() for text in batch_result.texts]

    def generate_body(
        self,
        metadata: PageMetadata,
        content_html: str,
        plain_text: str,
        images: List[ImageAsset],
    ) -> str:
        """Produce Markdown body text from raw article content."""
        self._ensure_model()
        prompt = self._build_prompt(metadata, content_html, plain_text, images)
        logger.debug("Prompt token length: %d", len(prompt))
        return self._generate_from_prompts([prompt])[0]

    def generate_bodies(
        self,
        requests: Sequence[MarkdownRequest],
    ) -> List[str]:
        """Produce Markdown bodies for multiple requests in a single batch."""
        if not requests:
            return []

        self._ensure_model()

        # Limit batch size to 3 to prevent memory/processing issues
        max_batch_size = 3
        results: List[str] = []

        for i in range(0, len(requests), max_batch_size):
            batch = requests[i : i + max_batch_size]
            logger.debug(
                "Processing batch %d of %d (size: %d)",
                (i // max_batch_size) + 1,
                (len(requests) + max_batch_size - 1) // max_batch_size,
                len(batch),
            )

            prompts = [
                self._build_prompt(
                    req.metadata,
                    req.content_html,
                    req.plain_text,
                    req.images,
                )
                for req in batch
            ]
            batch_results = self._generate_from_prompts(prompts)
            results.extend(batch_results)

        return results


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
