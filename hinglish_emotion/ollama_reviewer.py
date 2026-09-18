from __future__ import annotations

import json
from urllib import error, request

from .schemas import ClassifierResult, ReviewerResult


DEFAULT_OLLAMA_MODEL = "qwen3:0.6b"

SYSTEM_PROMPT = """You review difficult English-Hinglish sentiment examples.
Classify the speaker's intended attitude, not only the literal wording.
If praise appears together with a complaint, delay, failure, or a laughing emoji,
consider sarcasm. A sarcastic complaint has negative sentiment even if it starts
with words such as "wah", "great", or "kya service".

Example:
Input: wah kya service hai, 2 ghante late 😂
Output: {"sentiment":"negative","sarcasm":true,"text_emoji_relation":"conflict","evidence":["wah","2 ghante late","😂"]}

Return JSON only with exactly these fields:
sentiment: positive, neutral, or negative
sarcasm: true or false
text_emoji_relation: agreement, conflict, none, or uncertain
evidence: a short list containing only exact spans from the input
Do not add explanations outside the JSON object."""


class OllamaReviewer:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def availability(self) -> tuple[bool, str]:
        """Check that Ollama is running and that the configured model is installed."""
        try:
            with request.urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError):
            return False, "Ollama is not reachable at http://localhost:11434. Open Ollama, then try again."

        models = payload.get("models", [])
        installed = {
            item.get("name", "")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if self.model not in installed:
            return False, f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
        return True, f"Ollama is ready with {self.model}."

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
        except (error.URLError, error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach local Ollama. Start it and download {self.model} before using --reviewer ollama."
            ) from exc
        content = response_payload.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned an unexpected response")
        return ReviewerResult.from_dict(json.loads(content))
