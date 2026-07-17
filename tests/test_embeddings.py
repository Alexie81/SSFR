from __future__ import annotations

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
