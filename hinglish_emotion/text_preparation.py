"""Versioned text preparation shared by emotion training and inference."""

from __future__ import annotations

import re
import unicodedata

from .normalization import HinglishNormalizer

PREPARATION_VERSION = 1
TEXT_MODES = ("raw", "normalized")


def prepare_text(text: str, mode: str = "raw", version: int = PREPARATION_VERSION) -> str:
    if version != PREPARATION_VERSION:
        raise ValueError(f"Unsupported text preparation version: {version}")
    if mode not in TEXT_MODES:
        raise ValueError(f"Text mode must be one of {TEXT_MODES}")
    if mode == "normalized":
        return HinglishNormalizer().normalize(text).normalized_text
    # Do not lowercase, demojize, collapse repeated letters, or discard punctuation.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def duplicate_key(text: str) -> str:
    """Conservative equivalence key used only for leakage checks."""
    return HinglishNormalizer().normalize(text).normalized_text.casefold()
