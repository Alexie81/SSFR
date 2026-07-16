"""Offline-capable embedding providers and cache helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Protocol

import numpy as np

from .metrics import normalize_rows, normalize_vector


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        ...

    @property
    def provider_id(self) -> str:
        ...

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        ...

    def encode_query(self, query: str) -> np.ndarray:
        ...


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


class DeterministicHashEmbeddingProvider:
    """Signed feature hashing over word and character n-grams."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_id(self) -> str:
        return f"deterministic-hash-v1:{self.dimension}"

    def _add_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "little") % self.dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * weight

    def _encode(self, text: str) -> np.ndarray:
        normalized = _normalized_text(text)
        tokens = _TOKEN_PATTERN.findall(normalized)
        vector = np.zeros(self.dimension, dtype=np.float64)
        for token in tokens:
            self._add_feature(vector, f"w:{token}", 2.0)
            padded = f"^{token}$"
            for width, weight in ((3, 0.35), (4, 0.25), (5, 0.15)):
                for start in range(max(0, len(padded) - width + 1)):
                    self._add_feature(vector, f"c{width}:{padded[start:start + width]}", weight)
        for first, second in zip(tokens, tokens[1:], strict=False):
            self._add_feature(vector, f"b:{first}_{second}", 0.75)
        if not np.any(vector):
            self._add_feature(vector, "empty", 1.0)
        return normalize_vector(vector, name="text embedding")

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        return np.vstack([self._encode(text) for text in texts]).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query text cannot be empty")
        return self._encode(query).astype(np.float32)


class SentenceTransformersEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is not installed; install ssfr[embeddings] "
                "or use the deterministic hash provider"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_id(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query text cannot be empty")
        return self.encode_texts([query], batch_size=1)[0]


class OpenAIEmbeddingProvider:
    """Optional provider enabled only through explicit configuration."""

    def __init__(self, model_name: str = "text-embedding-3-small", dimension: int = 1536) -> None:
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("OpenAIEmbeddingProvider requires the optional 'openai' package")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        from openai import OpenAI

        self.model_name = model_name
        self._dimension = int(dimension)
        self._client = OpenAI()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_id(self) -> str:
        return f"openai:{self.model_name}:{self.dimension}"

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts[start : start + batch_size],
                dimensions=self.dimension,
            )
            batches.append(np.asarray([item.embedding for item in response.data]))
        return normalize_rows(np.vstack(batches)).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_texts([query], batch_size=1)[0]


def create_embedding_provider(
    provider: str = "hash",
    *,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    dimension: int = 384,
) -> EmbeddingProvider:
    if provider == "hash":
        return DeterministicHashEmbeddingProvider(dimension)
    if provider == "sentence-transformers":
        return SentenceTransformersEmbeddingProvider(model_name)
    if provider == "openai":
        return OpenAIEmbeddingProvider(model_name, dimension)
    if provider == "auto":
        if importlib.util.find_spec("sentence_transformers") is not None:
            try:
                return SentenceTransformersEmbeddingProvider(model_name)
            except Exception:
                pass
        return DeterministicHashEmbeddingProvider(dimension)
    raise ValueError("embedding provider must be hash, auto, sentence-transformers, or openai")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_embeddings(
    texts: list[str],
    provider: EmbeddingProvider,
    output_directory: str | Path,
    *,
    source_checksum: str,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[np.ndarray, bool]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    embedding_path = directory / "embeddings.npy"
    manifest_path = directory / "embedding_cache.json"
    cache_key = hashlib.sha256(
        f"{source_checksum}|{provider.provider_id}|{len(texts)}".encode("utf-8")
    ).hexdigest()
    if not force and embedding_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") == cache_key:
            vectors = np.load(embedding_path, allow_pickle=False)
            if vectors.shape == (len(texts), provider.dimension):
                return vectors, True

    vectors = provider.encode_texts(texts, batch_size=batch_size)
    vectors = normalize_rows(vectors, name="embeddings").astype(np.float32)
    np.save(embedding_path, vectors, allow_pickle=False)
    manifest_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "source_checksum": source_checksum,
                "provider_id": provider.provider_id,
                "dimension": provider.dimension,
                "row_count": len(texts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return vectors, False
