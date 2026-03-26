from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemOutput:
    question: str
    answer: str
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    navigation_path: list[str] = field(default_factory=list)
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    raw_model_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_contexts": self.retrieved_contexts,
            "retrieved_ids": self.retrieved_ids,
            "citations": self.citations,
            "navigation_path": self.navigation_path,
            "latency": self.latency_sec,
            "token_usage": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "api_calls": self.api_calls,
            "raw_model_text": self.raw_model_text,
            "extra": self.extra,
        }


def empty_usage() -> dict[str, float | int]:
    return {
        "latency_sec": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }
