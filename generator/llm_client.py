from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    api_calls: int = 1


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    latency_sec: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.api_calls += r.api_calls
        self.latency_sec += r.latency_sec


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResult:
        ...


class MockLLMClient(LLMClient):
    """Offline stub for CI and cheap dev loops."""

    def generate(self, prompt: str, **kwargs) -> LLMResult:
        t0 = time.perf_counter()
        snippet = prompt[-400:] if len(prompt) > 400 else prompt
        text = f"[mock] Based on the prompt tail: {snippet[:200]}..."
        return LLMResult(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(prompt) + len(text)) // 4,
            latency_sec=time.perf_counter() - t0,
            api_calls=1,
        )


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible HTTP API (OpenAI, vLLM, etc.)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: int = 120,
    ):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str, **kwargs) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.0),
        }
        t0 = time.perf_counter()
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        lat = time.perf_counter() - t0
        return LLMResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            latency_sec=lat,
            api_calls=1,
        )


def parse_evidence_ids_line(text: str) -> tuple[str, list[str]]:
    """Split model output into answer text and citation ids from EVIDENCE_IDS: [...] line."""
    m = re.search(r"EVIDENCE_IDS:\s*(\[[^\]]*\])", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return text.strip(), []
    try:
        ids = json.loads(m.group(1))
        if isinstance(ids, list):
            clean = [str(x) for x in ids]
        else:
            clean = []
    except json.JSONDecodeError:
        clean = []
    answer = text[: m.start()].strip()
    return answer, clean


def build_llm_from_config(cfg: dict) -> LLMClient:
    kind = (cfg.get("llm_backend") or "mock").lower()
    if kind == "openai":
        key = cfg.get("openai_api_key")
        if key is None or (isinstance(key, str) and not key.strip()):
            key = os.environ.get("OPENAI_API_KEY", "")
        base = cfg.get("openai_base_url")
        if isinstance(base, str) and not base.strip():
            base = None
        return OpenAICompatibleClient(
            model=cfg.get("llm_model"),
            api_key=key or None,
            base_url=base,
        )
    return MockLLMClient()
