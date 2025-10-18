"""Utility helpers for string normalization and path handling."""

from __future__ import annotations

import re

SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str, fallback: str = "page") -> str:
    """Generate a filesystem-friendly slug using ASCII characters only."""
    normalized = value.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = SLUG_PATTERN.sub("-", normalized).strip("-")
    return normalized or fallback
