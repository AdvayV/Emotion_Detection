from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import re
import unicodedata


TOKEN_PATTERN = re.compile(
    r"https?://\S+|@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|[A-Za-z]+|[0-9]+|[\u0900-\u097F]+|[^\w\s]",
    flags=re.UNICODE,
)
LONG_RUN_PATTERN = re.compile(r"(.)\1{2,}", flags=re.IGNORECASE)


DEFAULT_VARIANTS: dict[str, str] = {
    "acha": "accha",
    "achha": "accha",
    "accha": "accha",
    "acchha": "accha",
    "achhi": "acchi",
    "acchi": "acchi",
    "achi": "acchi",
    "bahut": "bahut",
    "bohot": "bahut",
    "bahoot": "bahut",
    "bhut": "bahut",
    "nahi": "nahi",
    "nahin": "nahi",
    "nhi": "nahi",
    "nai": "nahi",
    "bakwas": "bakwaas",
    "bakwaas": "bakwaas",
    "bekar": "bekaar",
    "bekaar": "bekaar",
    "mast": "mast",
    "wah": "wah",
    "yaar": "yaar",
    "theek": "theek",
    "thik": "theek",
    "good": "good",
    "bad": "bad",
    "happy": "happy",
    "sad": "sad",
    "love": "love",
    "hate": "hate",
}

NEGATIONS = {"nahi", "not", "never", "no", "mat", "without"}


@dataclass(frozen=True)
class NormalizedToken:
    raw: str
    normalized: str
    kind: str
    language: str
    elongated: bool = False
    removed_characters: int = 0
    is_negation: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedMessage:
    raw_text: str
    normalized_text: str
    tokens: tuple[NormalizedToken, ...]
    emojis: tuple[str, ...]
    has_negation: bool
    elongated_token_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "emojis": list(self.emojis),
            "has_negation": self.has_negation,
            "elongated_token_count": self.elongated_token_count,
        }


class HinglishNormalizer:
    """Conservative normalizer for Romanized Hindi and expressive spellings.

    Only known vocabulary forms are normalized. Unknown words are preserved so
    that preprocessing does not silently invent meanings.
    """

    def __init__(self, variants: dict[str, str] | None = None) -> None:
        self.variants = dict(DEFAULT_VARIANTS)
        if variants:
            self.variants.update({key.lower(): value.lower() for key, value in variants.items()})
        self.known_forms = set(self.variants)

    def normalize(self, text: str) -> NormalizedMessage:
        text = unicodedata.normalize("NFKC", text).strip()
        raw_tokens = TOKEN_PATTERN.findall(text)
        normalized_tokens = tuple(self._normalize_token(token) for token in raw_tokens)
        emojis = tuple(token.raw for token in normalized_tokens if token.kind == "symbol")
        normalized_text = self._join_tokens(token.normalized for token in normalized_tokens)
        return NormalizedMessage(
            raw_text=text,
            normalized_text=normalized_text,
            tokens=normalized_tokens,
            emojis=emojis,
            has_negation=any(token.is_negation for token in normalized_tokens),
            elongated_token_count=sum(token.elongated for token in normalized_tokens),
        )

    def _normalize_token(self, raw: str) -> NormalizedToken:
        if raw.startswith("http://") or raw.startswith("https://"):
            return NormalizedToken(raw, "<URL>", "url", "other")
        if raw.startswith("@"):
            return NormalizedToken(raw, "<USER>", "mention", "other")
        if raw.startswith("#"):
            content = raw[1:].lower()
            normalized, elongated, removed = self._normalize_latin(content)
            return NormalizedToken(
                raw, f"#{normalized}", "hashtag", self._language(normalized), elongated, removed,
                normalized in NEGATIONS,
            )
        if re.fullmatch(r"[A-Za-z]+", raw):
            normalized, elongated, removed = self._normalize_latin(raw.lower())
            return NormalizedToken(
                raw, normalized, "word", self._language(normalized), elongated, removed,
                normalized in NEGATIONS,
            )
        if raw.isdigit():
            return NormalizedToken(raw, raw, "number", "other")
        if re.fullmatch(r"[\u0900-\u097F]+", raw):
            return NormalizedToken(raw, raw, "word", "hi")
        return NormalizedToken(raw, raw, "symbol", "other")

    def _normalize_latin(self, token: str) -> tuple[str, bool, int]:
        direct = self.variants.get(token)
        if direct is not None:
            return direct, False, 0

        if not LONG_RUN_PATTERN.search(token):
            return token, False, 0

        candidates = self._collapse_candidates(token)
        known = [candidate for candidate in candidates if candidate[0] in self.known_forms]
        if known:
            # Prefer the candidate that removes the fewest characters, then the
            # shortest canonical form. This retains legitimate doubles in good.
            surface, removed = sorted(known, key=lambda item: (item[1], len(item[0])))[0]
            return self.variants[surface], True, removed

        # Unknown expressive spelling: collapse long runs to two characters.
        # The raw form is still available to the model and to error analysis.
        collapsed = LONG_RUN_PATTERN.sub(lambda match: match.group(1) * 2, token)
        removed = len(token) - len(collapsed)
        return collapsed, True, removed

    @staticmethod
    def _collapse_candidates(token: str) -> list[tuple[str, int]]:
        parts: list[str | tuple[str, int]] = []
        cursor = 0
        for match in LONG_RUN_PATTERN.finditer(token):
            parts.append(token[cursor : match.start()])
            parts.append((match.group(1), len(match.group(0))))
            cursor = match.end()
        parts.append(token[cursor:])

        run_choices = []
        for part in parts:
            if isinstance(part, tuple):
                character, original_length = part
                run_choices.append([(character, original_length - 1), (character * 2, original_length - 2)])
            else:
                run_choices.append([(part, 0)])

        candidates: list[tuple[str, int]] = []
        for selected in product(*run_choices):
            candidate = "".join(piece for piece, _ in selected)
            removed = sum(count for _, count in selected)
            candidates.append((candidate, removed))
        return candidates

    def _language(self, token: str) -> str:
        if token in {"accha", "acchi", "bahut", "nahi", "bakwaas", "bekaar", "mast", "wah", "yaar", "theek", "mat"}:
            return "hi"
        if token in {"good", "bad", "happy", "sad", "love", "hate", "not", "never", "no", "without"}:
            return "en"
        return "unknown"

    @staticmethod
    def _join_tokens(tokens: object) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text
