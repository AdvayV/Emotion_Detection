from __future__ import annotations

import json
from urllib import error, request

from .schemas import ClassifierResult, ReviewerResult


SYSTEM_PROMPT = """You review difficult English-Hinglish sentiment examples.
Return JSON only with exactly these fields:
sentiment: positive, neutral, or negative
sarcasm: true or false
text_emoji_relation: agreement, conflict, none, or uncertain
evidence: a short list containing only exact spans from the input
Do not add explanations outside the JSON object."""


class OllamaReviewer:
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def review(self, raw_text: str, normalized_text: str, primary: ClassifierResult) -> ReviewerResult:
        user_prompt = (
            f"Raw text: {raw_text}\n"
            f"Normalized text: {normalized_text}\n"
            f"Primary classifier: {json.dumps(primary.to_dict(), ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start it and download qwen3:4b before using --reviewer ollama."
            ) from exc
        content = response_payload.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned an unexpected response")
        return ReviewerResult.from_dict(json.loads(content))

