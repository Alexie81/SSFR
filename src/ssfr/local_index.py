"""Local shard indexes with exact NumPy, HNSWlib, and FAISS backends."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from .certificates import top_indices
from .metrics import normalize_rows, normalize_vector


def backend_available(backend: str) -> bool:
    module = {"hnsw": "hnswlib", "faiss": "faiss"}.get(backend)
    return True if module is None else importlib.util.find_spec(module) is not None


class LocalShardIndex:
    """A persistent local index whose scores are always "higher is better"."""

    def __init__(
        self,
        backend: str = "exact",
        distance_metric: str = "cosine",
        *,
        hnsw_ef_construction: int = 200,
        hnsw_m: int = 16,
        hnsw_ef_search: int = 64,
    ) -> None:
        if backend not in {"auto", "exact", "hnsw", "faiss", "diskann"}:
            raise ValueError("backend must be auto, exact, hnsw, faiss, or diskann")
        if distance_metric not in {"cosine", "euclidean"}:
            raise ValueError("distance_metric must be cosine or euclidean")
        if backend == "auto":
            backend = "hnsw" if backend_available("hnsw") else "exact"
        if backend == "diskann":
            raise NotImplementedError(
                "DiskANN is represented by an adapter boundary but is not implemented"
            )
        if backend in {"hnsw", "faiss"} and not backend_available(backend):
            package = "hnswlib" if backend == "hnsw" else "faiss-cpu"
            raise RuntimeError(
                f"{backend} backend requested but {package} is not installed; "
                "use backend='exact' or install the optional package"
            )
        self.backend = backend
        self.distance_metric = distance_metric
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self._index: object | None = None

    @property
    def built(self) -> bool:
        return self.vectors is not None and self.ids is not None

    @property
    def item_count(self) -> int:
        return 0 if self.ids is None else int(self.ids.size)

    @property
    def dimension(self) -> int:
        if self.vectors is None:
            raise RuntimeError("local index has not been built")
        return int(self.vectors.shape[1])

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        identifiers = np.asarray(ids)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError("vectors must have shape (item_count, dimension)")
        if identifiers.ndim != 1 or identifiers.shape[0] != matrix.shape[0]:
            raise ValueError("ids must be one-dimensional and match vectors")
        if len(np.unique(identifiers)) != identifiers.size:
            raise ValueError("ids must be unique within a local shard")
        if self.distance_metric == "cosine":
            matrix = normalize_rows(matrix, name="vectors").astype(np.float32)
        elif not np.all(np.isfinite(matrix)):
            raise ValueError("vectors contain non-finite values")
        self.vectors = np.ascontiguousarray(matrix)
        self.ids = identifiers.copy()
        self._build_backend()

    def _build_backend(self) -> None:
        if not self.built:
            raise RuntimeError("vectors and ids are unavailable")
        assert self.vectors is not None
        if self.backend == "exact":
            self._index = None
            return
        if self.backend == "hnsw":
            import hnswlib

            space = "cosine" if self.distance_metric == "cosine" else "l2"
            index = hnswlib.Index(space=space, dim=self.dimension)
            index.init_index(
                max_elements=self.item_count,
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_m,
                random_seed=42,
            )
            index.add_items(self.vectors, np.arange(self.item_count, dtype=np.int64))
            index.set_ef(max(self.hnsw_ef_search, 1))
            self._index = index
            return
        if self.backend == "faiss":
            import faiss

            if self.distance_metric == "cosine":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = faiss.IndexFlatL2(self.dimension)
            index.add(self.vectors)
            self._index = index

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("local index must be built before search")
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(f"query must have shape ({self.dimension},)")
        if self.distance_metric == "cosine":
            vector = normalize_vector(vector, name="query").astype(np.float32)
        elif not np.all(np.isfinite(vector)):
            raise ValueError("query contains non-finite values")
        return vector

    def _exact_search(
        self,
        query: np.ndarray,
        top_k: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.vectors is not None and self.ids is not None
        vectors = self.vectors if mask is None else self.vectors[mask]
        identifiers = self.ids if mask is None else self.ids[mask]
        if vectors.shape[0] == 0:
            return identifiers[:0], np.empty(0, dtype=np.float64)
        if self.distance_metric == "cosine":
            scores = vectors @ query
        else:
            scores = -np.linalg.norm(vectors - query, axis=1)
        selected = top_indices(scores, min(top_k, scores.size))
        return identifiers[selected], np.asarray(scores[selected], dtype=np.float64)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        allowed_ids: np.ndarray | set[object] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        vector = self._prepare_query(query)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        assert self.ids is not None
        if allowed_ids is not None:
            allowed = np.asarray(list(allowed_ids) if isinstance(allowed_ids, set) else allowed_ids)
            mask = np.isin(self.ids, allowed)
            return self._exact_search(vector, top_k, mask)
        if self.backend == "exact":
            return self._exact_search(vector, top_k)
        count = min(top_k, self.item_count)
        if self.backend == "hnsw":
            if self._index is None:
                self._build_backend()
            labels, distances = self._index.knn_query(vector, k=count)  # type: ignore[union-attr]
            labels = labels[0].astype(np.int64)
            distances = distances[0]
            scores = (
                1.0 - distances
                if self.distance_metric == "cosine"
                else -np.sqrt(np.maximum(distances, 0.0))
            )
            return self.ids[labels], np.asarray(scores, dtype=np.float64)
        if self.backend == "faiss":
            if self._index is None:
                self._build_backend()
            distances, labels = self._index.search(vector[None, :], count)  # type: ignore[union-attr]
            scores = (
                distances[0]
                if self.distance_metric == "cosine"
                else -np.sqrt(np.maximum(distances[0], 0.0))
            )
            return self.ids[labels[0]], np.asarray(scores, dtype=np.float64)
        raise RuntimeError(f"unsupported backend: {self.backend}")

    def estimated_candidates(self, allowed_ids: np.ndarray | None = None) -> int:
        if allowed_ids is not None and self.ids is not None:
            return int(np.count_nonzero(np.isin(self.ids, allowed_ids)))
        return self.item_count

    def save(self, path: str | Path) -> None:
        if not self.built:
            raise RuntimeError("local index must be built before save")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        assert self.vectors is not None and self.ids is not None
        np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        np.save(directory / "ids.npy", self.ids, allow_pickle=False)
        metadata = {
            "backend": self.backend,
            "distance_metric": self.distance_metric,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_search": self.hnsw_ef_search,
            "item_count": self.item_count,
            "dimension": self.dimension,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if self.backend == "hnsw" and self._index is not None:
            self._index.save_index(str(directory / "index.hnsw"))  # type: ignore[union-attr]
        elif self.backend == "faiss" and self._index is not None:
            import faiss

            faiss.write_index(self._index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, path: str | Path) -> "LocalShardIndex":
        directory = Path(path)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        index = cls(
            backend=metadata["backend"],
            distance_metric=metadata["distance_metric"],
            hnsw_ef_construction=metadata["hnsw_ef_construction"],
            hnsw_m=metadata["hnsw_m"],
            hnsw_ef_search=metadata["hnsw_ef_search"],
        )
        index.vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        index.ids = np.load(directory / "ids.npy", allow_pickle=False)
        if index.backend == "hnsw" and (directory / "index.hnsw").exists():
            import hnswlib

            native = hnswlib.Index(
                space="cosine" if index.distance_metric == "cosine" else "l2",
                dim=index.dimension,
            )
            native.load_index(str(directory / "index.hnsw"), max_elements=index.item_count)
            native.set_ef(index.hnsw_ef_search)
            index._index = native
        elif index.backend == "faiss" and (directory / "index.faiss").exists():
            import faiss

            index._index = faiss.read_index(str(directory / "index.faiss"))
        else:
            index._build_backend()
        return index
