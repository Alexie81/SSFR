from __future__ import annotations

import sys
import types

import numpy as np

from ssfr.embeddings import create_embedding_provider


def test_fast_hash_is_deterministic_normalized_and_batch_safe() -> None:
    provider = create_embedding_provider("fast-hash", dimension=64)
    texts = [
        "Laptop pentru programare și editare video",
        "Telefon cu baterie mare și cameră bună",
        "Laptop pentru programare și editare video",
    ]
    vectors = provider.encode_texts(texts, batch_size=2)
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.array_equal(vectors[0], vectors[2])
    assert not np.array_equal(vectors[0], vectors[1])
    assert np.array_equal(provider.encode_query(texts[0]), vectors[0])


def test_multilingual_e5_uses_asymmetric_retrieval_prefixes(monkeypatch) -> None:
    encoded_inputs: list[list[str]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts, **kwargs):
            encoded_inputs.append(list(texts))
            return np.tile(
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                (len(texts), 1),
            )

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = create_embedding_provider("multilingual-e5")
    documents = provider.encode_texts(["laptop pentru programare", "running shoes"])
    query = provider.encode_query("programming notebook")

    assert provider.provider_id == "multilingual-e5:intfloat/multilingual-e5-small:fp32"
    assert documents.shape == (2, 3)
    assert query.shape == (3,)
    assert encoded_inputs == [
        ["passage: laptop pentru programare", "passage: running shoes"],
        ["query: programming notebook"],
    ]
