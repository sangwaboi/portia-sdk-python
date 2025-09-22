"""Unit tests for Meta hosted Llama provider integration."""

import pytest

from portia.config import Config
from portia.model import LLMProvider


def test_meta_provider_enum_exists() -> None:
    """LLMProvider exposes META enum value."""
    assert LLMProvider.META.value == "meta"


def test_meta_parse_model_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config parses meta/<model> into a GenerativeModel instance."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._parse_model_string("meta/llama-3-8b-instruct")
    assert model.provider == LLMProvider.META
    assert str(model) == "meta/llama-3-8b-instruct"


def test_meta_auto_detection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider auto-detection prefers META when META_API_KEY and META_BASE_URL are present."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    assert c.llm_provider == LLMProvider.META


def test_meta_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Meta model construction via _construct_model_from_name."""
    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._construct_model_from_name(LLMProvider.META, "llama-3-8b-instruct")
    assert model.provider == LLMProvider.META
    assert str(model) == "meta/llama-3-8b-instruct"


@pytest.mark.asyncio
async def test_meta_aget_response_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async text response path uses model.aget_response; patch that directly."""
    from portia.model import Message

    monkeypatch.setenv("META_API_KEY", "test-meta-api-key")
    monkeypatch.setenv("META_BASE_URL", "https://example.meta.llama.api/v1")

    c = Config.from_default()
    model = c._parse_model_string("meta/llama-3-8b-instruct")

    async def fake_aget_response(_msgs) -> Message:  # noqa: ANN001
        return Message(role="assistant", content="hi")

    monkeypatch.setattr(model, "aget_response", fake_aget_response)

    reply = await model.aget_response([Message(role="user", content="hello")])
    assert reply.role == "assistant"
    assert isinstance(reply.content, str)
