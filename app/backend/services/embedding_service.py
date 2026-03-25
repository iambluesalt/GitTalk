"""
Embedding service using direct Ollama HTTP API.
Generates vector embeddings for code chunks.
"""
import httpx

from config import settings
from logger import logger


class EmbeddingService:
    """Generates embeddings via Ollama /api/embed endpoint."""

    # nomic-embed-text context limit is 8192 tokens; leave room for prefix overhead
    MAX_EMBED_CHARS = 24000  # ~6000 tokens at 4 chars/token, safe margin

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

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within embedding model context window."""
        if len(text) <= self.MAX_EMBED_CHARS:
            return text
        return text[: self.MAX_EMBED_CHARS]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of document texts, batching as needed.
        Automatically adds 'search_document: ' prefix for nomic-embed-text
        and truncates texts that exceed the model's context window.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per text)
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        prefixed = [f"search_document: {self._truncate(t)}" for t in texts]

        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            try:
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails (e.g. context length), fall back to one-by-one
                logger.warning(f"Batch embed failed ({len(batch)} texts), retrying individually: {e}")
                for text in batch:
                    try:
                        single = await self._embed_batch([text])
                        all_embeddings.extend(single)
                    except Exception as e2:
                        logger.warning(f"Single embed failed, using zero vector: {e2}")
                        all_embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single query text.
        Automatically adds 'search_query: ' prefix for nomic-embed-text."""
        results = await self._embed_batch([f"search_query: {self._truncate(text)}"])
        if results:
            return results[0]
        return []

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a batch to Ollama /api/embed."""
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    logger.warning(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )
                return embeddings
        except httpx.TimeoutException:
            logger.error(f"Embedding request timed out after {self.timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

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
