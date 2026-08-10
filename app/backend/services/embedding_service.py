"""
Embedding service backed by LangChain's OllamaEmbeddings.
Generates vector embeddings for code chunks.
"""
import httpx
from langchain_ollama import OllamaEmbeddings

from config import settings
from logger import logger


# nomic-embed-text context limit is 8192 tokens; leave room for prefix overhead
MAX_EMBED_CHARS = 24000  # ~6000 tokens at 4 chars/token, safe margin


def _truncate(text: str) -> str:
    """Truncate text to fit within the embedding model context window."""
    if len(text) <= MAX_EMBED_CHARS:
        return text
    return text[:MAX_EMBED_CHARS]


class NomicOllamaEmbeddings(OllamaEmbeddings):
    """
    OllamaEmbeddings with nomic-embed-text task prefixes and context truncation.

    nomic-embed-text requires asymmetric 'search_document: ' / 'search_query: '
    prefixes to distinguish indexed documents from queries; LangChain's client
    does not inject them, and it does not truncate over-long inputs either.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(
            [f"search_document: {_truncate(t)}" for t in texts]
        )

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents(
            [f"search_document: {_truncate(t)}" for t in texts]
        )

    def embed_query(self, text: str) -> list[float]:
        # Bypass our own embed_documents so the query prefix isn't overwritten
        results = super().embed_documents([f"search_query: {_truncate(text)}"])
        return results[0] if results else []

    async def aembed_query(self, text: str) -> list[float]:
        results = await super().aembed_documents([f"search_query: {_truncate(text)}"])
        return results[0] if results else []


class EmbeddingService:
    """Generates embeddings via Ollama, batching and degrading gracefully."""

    MAX_EMBED_CHARS = MAX_EMBED_CHARS

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_EMBED_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.timeout = settings.OLLAMA_TIMEOUT
        self.embeddings = NomicOllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
            client_kwargs={"timeout": float(self.timeout)},
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of document texts, batching as needed.

        LangChain sends the whole list to /api/embed in one request, so batching
        stays here to keep individual requests bounded.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per text)
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails (e.g. context length), fall back to one-by-one
                logger.warning(f"Batch embed failed ({len(batch)} texts), retrying individually: {e}")
                for text in batch:
                    try:
                        single = await self.embeddings.aembed_documents([text])
                        all_embeddings.extend(single)
                    except Exception as e2:
                        logger.warning(f"Single embed failed, using zero vector: {e2}")
                        all_embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        return await self.embeddings.aembed_query(text)

    async def is_available(self) -> bool:
        """Check if Ollama and the embedding model are accessible."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Check Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check the embedding model is available
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match model name with or without :latest tag
                target = self.model
                target_with_tag = f"{target}:latest"
                return any(
                    m == target or m == target_with_tag or m.split(":")[0] == target
                    for m in models
                )
        except Exception:
            return False


# Global instance
embedding_service = EmbeddingService()
