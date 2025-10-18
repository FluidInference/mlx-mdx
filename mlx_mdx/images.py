"""Image downloading and validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests
from filetype import guess

from .models import ImageAsset, ImageCandidate
from .utils import slugify

logger = logging.getLogger("mlx_mdx")

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_BYTES = 512
ALLOWED_IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}


def detect_image_format(data: bytes) -> Optional[str]:
    """Detect image type using filetype; returns lowercase extension."""
    kind = guess(data)
    if kind and kind.mime.startswith("image/"):
        ext = kind.extension.lower()
        if ext == "jpeg":
            return "jpg"
        return ext
    return None


def infer_image_extension(content_type: Optional[str], data: bytes) -> Optional[str]:
    """Guess an image file extension from HTTP metadata or file signature."""
    detected = detect_image_format(data)
    if detected:
        return detected
    if not content_type:
        return None
    parts = content_type.split(";")[0].split("/")
    if len(parts) == 2 and parts[0] == "image":
        ext = parts[1].strip().lower()
        if ext == "jpeg":
            ext = "jpg"
        return ext
    return None


def download_images(
    candidates: List[ImageCandidate],
    output_dir: Path,
) -> List[ImageAsset]:
    """Download images referenced by the article and persist them locally."""
    if not candidates:
        return []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    downloaded: Dict[str, ImageAsset] = {}
    assets: List[ImageAsset] = []

    for index, candidate in enumerate(candidates, start=1):
        if candidate.absolute_url in downloaded:
            assets.append(downloaded[candidate.absolute_url])
            continue
        try:
            resp = session.get(candidate.absolute_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch image %s: %s", candidate.absolute_url, exc)
            continue

        content_type = resp.headers.get("Content-Type", "")
        data = resp.content
        if len(data) < MIN_IMAGE_BYTES:
            logger.warning("Skipping %s: response too small", candidate.absolute_url)
            continue
        if len(data) > MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping %s: image larger than %s bytes",
                candidate.absolute_url,
                MAX_IMAGE_BYTES,
            )
            continue

        extension = infer_image_extension(content_type, data)
        if not extension or extension.lower() not in ALLOWED_IMAGE_TYPES:
            logger.warning(
                "Skipping %s: unsupported image type (Content-Type=%s)",
                candidate.absolute_url,
                content_type,
            )
            continue

        alt_slug = slugify(candidate.alt_text or "image", fallback="image")
        filename = f"image-{index:02d}-{alt_slug}"[:80] + f".{extension}"
        destination = image_dir / filename

        try:
            destination.write_bytes(data)
        except OSError as exc:
            logger.warning("Failed to write image %s: %s", destination, exc)
            continue

        asset = ImageAsset(
            original_src=candidate.original_src,
            absolute_url=candidate.absolute_url,
            alt_text=candidate.alt_text,
            filename=filename,
            relative_path=str(Path("images") / filename),
        )
        downloaded[candidate.absolute_url] = asset
        assets.append(asset)
    return assets
