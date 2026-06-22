"""LLM client for agent reasoning (plan §7.2).

Provides OpenAI-compatible API integration with fallback to deterministic
heuristics when no API key is available. Supports custom base_url for
proxy/alternative endpoints (e.g. https://api.ml-router.su/v1).
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"


class LLMClient:
    """LLM client with OpenAI-compatible provider and heuristic fallback.

    Supports custom ``base_url`` for proxy endpoints and alternative providers.
    When ``api_key`` is None or empty, all ``generate()`` calls return
    ``None`` so callers can fall back to their heuristic logic.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.available = bool(self.api_key)

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2000
    ) -> dict | None:
        """Call the LLM and return a parsed JSON response.

        Returns ``None`` if no API key is available or the call fails,
        so callers can fall back to heuristic logic.
        """
        if not self.available:
            return None
        try:
            if self.provider == "openai":
                return self._call_openai(system_prompt, user_prompt, temperature, max_tokens)
            logger.warning(f"Unknown LLM provider: {self.provider}")
            return None
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    def generate_text(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2000
    ) -> str | None:
        """Call the LLM and return raw text (not parsed as JSON)."""
        if not self.available:
            return None
        try:
            if self.provider == "openai":
                result = self._call_openai_raw(system_prompt, user_prompt, temperature, max_tokens)
                return result
            return None
        except Exception as e:
            logger.warning(f"LLM text call failed: {e}")
            return None

    def _call_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> dict | None:
        """Call OpenAI-compatible chat completions API and parse JSON response."""
        text = self._call_openai_raw(system_prompt, user_prompt, temperature, max_tokens)
        if text is None:
            return None
        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict | None:
        """Try to extract and parse JSON from potentially messy LLM output."""
        if not text:
            return None

        text = text.strip()

        # Strip markdown code fences
        if "```" in text:
            lines = text.split("\n")
            in_fence = False
            cleaned = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence or not line.strip().startswith("```"):
                    cleaned.append(line)
            text = "\n".join(cleaned).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find all {...} blocks and try each
        import re

        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = re.sub(r",\s*}", "}", candidate)
                fixed = re.sub(r",\s*]", "]", fixed)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # Last resort: find first { to last } and try
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                import re

                fixed = re.sub(r",\s*}", "}", candidate)
                fixed = re.sub(r",\s*]", "]", fixed)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Could not parse JSON from LLM output: {text[:300]}")
        return None

    def _call_openai_raw(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str | None:
        """Call OpenAI-compatible chat completions and return raw text."""
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        logger.info(f"LLM call: {self.model} @ {self.base_url}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"]
