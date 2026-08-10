"""
Tests for the embedding service (embedding_service.py).

The transport is LangChain's OllamaEmbeddings, so what is tested here is the
behaviour we layer on top of it: nomic-embed-text task prefixes, context
truncation, request batching, and graceful degradation when a batch fails.
The parent client call is stubbed, so no Ollama server is needed.
"""
import pytest
from langchain_ollama import OllamaEmbeddings

from services.embedding_service import (
    MAX_EMBED_CHARS,
    EmbeddingService,
    NomicOllamaEmbeddings,
)


@pytest.fixture()
def calls(monkeypatch):
    """Record the texts handed to the underlying client; return dummy vectors."""
    recorded: list[list[str]] = []

    async def _fake(self, texts):
        recorded.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(OllamaEmbeddings, "aembed_documents", _fake)
    return recorded


@pytest.fixture()
def embedder():
    return NomicOllamaEmbeddings(model="nomic-embed-text", base_url="http://test")


# ═══════════════════════════════════════════════════════════════════════════════
# nomic task prefixes
# ═══════════════════════════════════════════════════════════════════════════════

async def test_documents_get_search_document_prefix(embedder, calls):
    await embedder.aembed_documents(["def foo(): pass", "class Bar: ..."])
    assert calls[0] == [
        "search_document: def foo(): pass",
        "search_document: class Bar: ...",
    ]


async def test_query_gets_search_query_prefix(embedder, calls):
    """Queries use the asymmetric query prefix — and must not be double-prefixed."""
    await embedder.aembed_query("how does auth work")
    assert calls[0] == ["search_query: how does auth work"]


async def test_query_returns_single_vector(embedder, calls):
    vector = await embedder.aembed_query("q")
    assert vector == [0.1, 0.2, 0.3]


# ═══════════════════════════════════════════════════════════════════════════════
# Truncation
# ═══════════════════════════════════════════════════════════════════════════════

async def test_long_document_truncated_to_context_limit(embedder, calls):
    await embedder.aembed_documents(["a" * (MAX_EMBED_CHARS * 2)])
    sent = calls[0][0]
    assert sent.startswith("search_document: ")
    assert len(sent) == len("search_document: ") + MAX_EMBED_CHARS


async def test_short_document_not_truncated(embedder, calls):
    await embedder.aembed_documents(["short"])
    assert calls[0] == ["search_document: short"]


# ═══════════════════════════════════════════════════════════════════════════════
# Batching
# ═══════════════════════════════════════════════════════════════════════════════

async def test_empty_input_makes_no_request(calls):
    svc = EmbeddingService(batch_size=4)
    assert await svc.embed_texts([]) == []
    assert calls == []


async def test_texts_split_into_batches(calls):
    """LangChain sends whole lists in one request, so batching stays our job."""
    svc = EmbeddingService(batch_size=2)
    results = await svc.embed_texts([f"t{i}" for i in range(5)])

    assert len(results) == 5
    assert [len(batch) for batch in calls] == [2, 2, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Failure handling
# ═══════════════════════════════════════════════════════════════════════════════

async def test_failed_batch_retries_individually(monkeypatch):
    """A batch that blows up is retried one text at a time before giving up."""
    attempts: list[int] = []

    async def _fake(self, texts):
        attempts.append(len(texts))
        if len(texts) > 1:
            raise RuntimeError("context length exceeded")
        return [[1.0, 2.0]]

    monkeypatch.setattr(OllamaEmbeddings, "aembed_documents", _fake)

    svc = EmbeddingService(batch_size=3)
    results = await svc.embed_texts(["a", "b", "c"])

    assert attempts == [3, 1, 1, 1]  # failed batch, then one-by-one
    assert results == [[1.0, 2.0]] * 3


async def test_single_failure_falls_back_to_zero_vector(monkeypatch):
    from config import settings

    async def _always_fails(self, texts):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(OllamaEmbeddings, "aembed_documents", _always_fails)

    svc = EmbeddingService(batch_size=2)
    results = await svc.embed_texts(["a", "b"])

    assert results == [[0.0] * settings.EMBEDDING_DIMENSIONS] * 2


async def test_embed_single_delegates_to_query_path(calls):
    svc = EmbeddingService()
    vector = await svc.embed_single("find the router")

    assert calls[0] == ["search_query: find the router"]
    assert vector == [0.1, 0.2, 0.3]
