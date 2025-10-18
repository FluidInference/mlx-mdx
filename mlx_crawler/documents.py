"""Document OCR pipeline powered by MLX vision-language models."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union, cast

try:  # Optional PDF rendering support
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - optional dependency
    pdfium = None  # type: ignore[assignment]

from PIL import Image

from mlx_vlm import generate as generate_text
from mlx_vlm import load as load_vlm_model
from mlx_vlm.prompt_utils import apply_chat_template

from .markdown import ReaderLMMarkdownGenerator, compose_markdown
from .models import PageMetadata
from .utils import slugify

logger = logging.getLogger("mlx_crawler.documents")


DEFAULT_OCR_MODEL_ID = "mlx-community/Nanonets-OCR2-3B-4bit"
DEFAULT_OCR_SYSTEM_PROMPT = (
    "You transcribe document pages into clean Markdown. Reproduce layout faithfully "
    "using headings, lists, and tables when relevant. Preserve math notation and spellings. "
    "Return only Markdown without extra commentary."
)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_PDF_DPI = 200
DEFAULT_MAX_IMAGE_SIDE = 2048


ImageInput = Union[str, Image.Image]


@dataclass
class DocumentConfig:
    """Settings controlling OCR-driven Markdown generation."""

    output_root: Path
    model_id: str = DEFAULT_OCR_MODEL_ID
    max_tokens: int = 2048
    temperature: float = 0.0
    system_prompt: str = DEFAULT_OCR_SYSTEM_PROMPT
    pdf_dpi: int = DEFAULT_PDF_DPI
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE


@dataclass
class DocumentResult:
    """Timing details for processed documents."""

    source_path: Path
    output_path: Path
    page_count: int
    total_seconds: float


@dataclass
class DocumentPage:
    """A single page from a document along with rendering metadata."""

    index: int
    label: str
    image_input: ImageInput


class DocumentOCRMarkdownGenerator:
    """Wrapper around Nanonets OCR model to emit Markdown per page."""

    def __init__(
        self,
        model_id: str,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_OCR_SYSTEM_PROMPT
        self.max_image_side = max_image_side
        self._model = None
        self._processor = None
        self._config = None
        self._resolved_model_path: Optional[str] = None

    def _resolve_env_override(self) -> Optional[Path]:
        for env_var in ("OCR_MODEL_DIR", "MODEL_DIR"):
            override = os.getenv(env_var)
            if not override:
                continue
            override_path = Path(override).expanduser()
            if override_path.exists():
                logger.debug("%s override detected at %s", env_var, override_path)
                return override_path
            logger.warning(
                "%s is set to %s but the path does not exist; falling back to %s",
                env_var,
                override_path,
                self.model_id,
            )
        return None

    def _resolve_snapshot_path(self, repo_id: str) -> Path:
        from huggingface_hub import snapshot_download

        allow_patterns = ReaderLMMarkdownGenerator._ALLOW_PATTERNS
        try:
            local_path = Path(
                snapshot_download(
                    repo_id,
                    local_files_only=True,
                    allow_patterns=allow_patterns,
                )
            )
            logger.info(
                "Resolved cached OCR model for %s at %s (local_files_only=True)",
                repo_id,
                local_path,
            )
            return local_path
        except Exception as err:  # noqa: BLE001
            logger.info(
                "Local cache for OCR model %s was not found (%s); attempting snapshot download.",
                repo_id,
                err,
            )
            local_path = Path(snapshot_download(repo_id, allow_patterns=allow_patterns))
            logger.debug("Downloaded OCR model %s to %s", repo_id, local_path)
            return local_path

    def _determine_model_path(self) -> str:
        if self._resolved_model_path:
            return self._resolved_model_path

        env_override = self._resolve_env_override()
        if env_override:
            self._resolved_model_path = str(env_override)
            return self._resolved_model_path

        candidate = Path(self.model_id).expanduser()
        if candidate.exists():
            logger.debug(
                "OCR model identifier %s resolves to local path %s",
                self.model_id,
                candidate,
            )
            self._resolved_model_path = str(candidate)
            return self._resolved_model_path

        snapshot_path = self._resolve_snapshot_path(self.model_id)
        self._resolved_model_path = str(snapshot_path)
        return self._resolved_model_path

    def _ensure_model(self) -> None:
        if self._model is None or self._processor is None or self._config is None:
            load_target = self._determine_model_path()
            if load_target != self.model_id:
                logger.info(
                    "Loading OCR model %s (resolved from %s)",
                    load_target,
                    self.model_id,
                )
            else:
                logger.info("Loading OCR model %s", load_target)
            model, processor = load_vlm_model(load_target, trust_remote_code=True)
            self._model = model
            self._processor = processor
            self._config = getattr(model, "config", None)
            if self._config is None:
                raise RuntimeError("Loaded OCR model does not expose configuration")

    def generate_page_markdown(
        self,
        page: DocumentPage,
        total_pages: int,
        document_name: str,
    ) -> str:
        """Run OCR on a single page and return Markdown text."""

        self._ensure_model()
        assert self._model is not None
        assert self._processor is not None
        assert self._config is not None

        page_prompt = (
            f"Document: {document_name}. Page {page.index} of {total_pages}. "
            "Transcribe the provided page to Markdown, preserving hierarchy, tables, and lists. "
            "Do not add commentary or restate the instructions."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": page_prompt},
        ]

        formatted_prompt = cast(
            str,
            apply_chat_template(
                self._processor,
                self._config,
                messages,
                num_images=1,
            ),
        )

        image_argument: List[ImageInput]
        if isinstance(page.image_input, Image.Image):
            image = page.image_input
            width, height = image.size
            longest_edge = max(width, height)
            if longest_edge > self.max_image_side:
                scale = self.max_image_side / float(longest_edge)
                new_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            image_argument = [image]
        else:
            image_path = Path(page.image_input)
            try:
                with Image.open(image_path) as raw_image:
                    image = raw_image.convert("RGB")
                    width, height = image.size
                    longest_edge = max(width, height)
                    if longest_edge > self.max_image_side:
                        scale = self.max_image_side / float(longest_edge)
                        new_size = (
                            max(1, int(width * scale)),
                            max(1, int(height * scale)),
                        )
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    image_argument = [image.copy()]
            except (OSError, FileNotFoundError):
                image_argument = [str(page.image_input)]

        result = generate_text(
            self._model,
            self._processor,
            formatted_prompt,
            image=image_argument,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return result.text.strip()


def _render_pdf(path: Path, dpi: int) -> List[Image.Image]:
    if pdfium is None:  # pragma: no cover - optional dependency handling
        raise RuntimeError(
            "PDF support requires the optional 'pypdfium2' package. Install it with 'pip install pypdfium2'."
        )
    dpi = max(int(dpi or DEFAULT_PDF_DPI), 72)
    scale = dpi / 72
    document = pdfium.PdfDocument(str(path))
    images: List[Image.Image] = []
    for page_index, page in enumerate(document, start=1):
        pil_image = page.render(scale=scale).to_pil()
        images.append(pil_image.convert("RGB"))
        logger.debug("Rendered %s page %d to image", path.name, page_index)
    return images


def _collect_pages_from_directory(path: Path) -> List[DocumentPage]:
    pages: List[DocumentPage] = []
    for index, image_path in enumerate(sorted(path.iterdir()), start=1):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        pages.append(DocumentPage(index=index, label=image_path.name, image_input=str(image_path)))
    return pages


def _collect_pages_from_file(path: Path, config: DocumentConfig) -> List[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        images = _render_pdf(path, config.pdf_dpi)
        return [
            DocumentPage(index=idx, label=f"{path.name}#page={idx}", image_input=image)
            for idx, image in enumerate(images, start=1)
        ]
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [DocumentPage(index=1, label=path.name, image_input=str(path))]
    raise ValueError(f"Unsupported document format: {path}")


def collect_document_pages(path: Path, config: DocumentConfig) -> List[DocumentPage]:
    if path.is_dir():
        pages = _collect_pages_from_directory(path)
        if not pages:
            raise ValueError(f"No supported image files found in directory: {path}")
        return pages
    if path.is_file():
        return _collect_pages_from_file(path, config)
    raise FileNotFoundError(f"Document path does not exist: {path}")


def build_document_output_dir(config: DocumentConfig, source: Path) -> Path:
    slug = slugify(source.stem or "document", fallback="document")
    output_dir = config.output_root / "documents" / slug[:80]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compose_document_markdown(
    metadata: PageMetadata,
    page_markdown: Sequence[str],
) -> str:
    cleaned_pages = [section.strip() for section in page_markdown if section.strip()]
    if not cleaned_pages:
        return compose_markdown(metadata, "", [])

    body_parts: List[str] = []
    for idx, section in enumerate(cleaned_pages, start=1):
        body_parts.append(section)
        if idx < len(cleaned_pages):
            body_parts.append(f"\n<!-- page {idx} end -->\n")
    body = "\n\n".join(body_parts).strip()
    return compose_markdown(metadata, body, [])


def process_document(
    path: Path,
    config: DocumentConfig,
    generator: DocumentOCRMarkdownGenerator,
) -> Optional[DocumentResult]:
    pages: List[DocumentPage]
    try:
        pages = collect_document_pages(path, config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Skipping %s: %s", path, exc)
        return None

    if not pages:
        logger.warning("No renderable pages found in %s", path)
        return None

    total_pages = len(pages)
    logger.info("Transcribing %s (%d page%s)", path, total_pages, "s" if total_pages != 1 else "")

    output_dir = build_document_output_dir(config, path)
    metadata = PageMetadata(
        source_url=path.resolve().as_uri(),
        title=path.stem,
        description=None,
        byline=None,
    )

    page_markdown: List[str] = []
    start = time.perf_counter()
    for page in pages:
        page_start = time.perf_counter()
        markdown = generator.generate_page_markdown(page, total_pages, metadata.title or path.name)
        elapsed = time.perf_counter() - page_start
        logger.debug("Page %d processed in %.2fs", page.index, elapsed)
        page_markdown.append(markdown)
    total_elapsed = time.perf_counter() - start

    full_markdown = compose_document_markdown(metadata, page_markdown)
    output_path = output_dir / "index.md"
    output_path.write_text(full_markdown, encoding="utf-8")
    logger.info("Saved Markdown to %s", output_path)

    return DocumentResult(
        source_path=path,
        output_path=output_path,
        page_count=total_pages,
        total_seconds=total_elapsed,
    )
