"""Keyword-prefiltered visual search and globally normalized descriptor fusion."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.descriptors import AVAILABLE_DESCRIPTOR_NAMES
from backend.app.core.index.brute_force import knn_search as brute_force
from backend.app.core.index.registry import IndexRegistry
from backend.app.core.search.text_search import candidate_image_ids, dataset_image_count
from backend.app.db.models import Image


SMALL_CANDIDATE_LIMIT = 50_000
MAX_OVERFETCH = 50_000


@dataclass
class SearchHit:
    image: Image
    distance: float
    descriptor_distances: dict[str, float | None]


class SearchEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.registry = IndexRegistry(data_dir / "indexes")

    @staticmethod
    def _project(vector: np.ndarray, mean: np.ndarray, components: np.ndarray,
                 normalize: bool = False) -> np.ndarray:
        if components.size:
            vector = (vector - mean) @ components.T
            if normalize:
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector = vector / norm
        return vector

    def _candidate_rows(self, registered, candidates: list[int] | None) -> np.ndarray:
        """Map DB image IDs to descriptor rows, dropping images without vectors."""
        if not len(registered.image_ids):
            return np.empty(0, dtype=np.int64)
        if candidates is None:
            return np.arange(len(registered.image_ids), dtype=np.int64)
        if not candidates:
            return np.empty(0, dtype=np.int64)
        wanted = np.asarray(candidates, dtype=np.int64)
        positions = np.searchsorted(registered.sorted_image_ids, wanted)
        valid = positions < len(registered.sorted_image_ids)
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] &= registered.sorted_image_ids[positions[valid_positions]] == wanted[valid_positions]
        return registered.sorted_rows[positions[valid]].astype(np.int64, copy=False)

    def _small_subset_neighbors(self, registered, query: np.ndarray,
                                candidate_ids: list[int] | None, k: int):
        rows = self._candidate_rows(registered, candidate_ids)
        if not len(rows):
            return []
        vectors = np.asarray(registered.tree.vectors[rows], dtype=np.float32)
        local = brute_force(vectors, query, min(k, len(rows)), metric=registered.tree.metric)
        return [(int(rows[position]), distance) for position, distance in local]

    def _tree_filtered_neighbors(self, registered, query: np.ndarray,
                                 candidate_set: set[int] | None, k: int):
        n = len(registered.image_ids)
        if not n:
            return []
        fetch_cap = min(n, max(k, min(MAX_OVERFETCH, k * 100)))
        fetch = min(fetch_cap, max(k, k * 10))
        filtered: list[tuple[int, float]] = []
        while True:
            neighbors = registered.tree.knn_search(query, fetch)
            filtered = [(row, distance) for row, distance in neighbors
                        if candidate_set is None or int(registered.image_ids[row]) in candidate_set]
            if len(filtered) >= k or fetch >= fetch_cap or fetch >= n:
                return filtered[:k]
            fetch = min(fetch_cap, n, fetch * 2)

    def search(self, db: Session, dataset: str, keywords: list[str], mode: str,
               query_image_id: int | None, selections: list[tuple[str, float]], k: int,
               query_vectors: dict[str, np.ndarray] | None = None) -> list[SearchHit]:
        filtered_ids = candidate_image_ids(db, dataset, keywords, mode)
        candidate_count = len(filtered_ids) if filtered_ids is not None else dataset_image_count(db, dataset)
        if query_image_id is not None and db.query(Image.id).filter_by(id=query_image_id, dataset=dataset).first() is None:
            raise LookupError(f"Unknown query_image_id {query_image_id} for dataset '{dataset}'")

        # Text-only search remains useful when no visual query was supplied.
        if not selections or (query_image_id is None and not query_vectors):
            ids = filtered_ids
            if ids is None:
                ids = [int(row[0]) for row in db.query(Image.id).filter_by(dataset=dataset).order_by(Image.id).limit(k).all()]
            chosen = ids[:k]
            images = db.query(Image).filter(Image.id.in_(chosen)).all() if chosen else []
            by_id = {image.id: image for image in images}
            return [SearchHit(by_id[image_id], 0.0, {}) for image_id in chosen if image_id in by_id]

        candidate_set = set(filtered_ids) if filtered_ids is not None and candidate_count > SMALL_CANDIDATE_LIMIT else None
        loaded: dict[str, tuple[object, np.ndarray, float]] = {}
        candidate_pool: set[int] = set()
        for name, weight in selections:
            if name not in AVAILABLE_DESCRIPTOR_NAMES:
                raise ValueError(f"Unknown descriptor '{name}'")
            registered = self.registry.get(dataset, name)
            if registered is None:
                raise FileNotFoundError(f"No built index and ids.npy for dataset '{dataset}', descriptor '{name}'")

            if query_vectors is not None and name in query_vectors:
                raw_query = np.asarray(query_vectors[name], dtype=np.float32)
            elif query_image_id is not None:
                q_index = registered.row_for_image_id(query_image_id)
                if q_index is None:
                    raise LookupError(f"No vector for query_image_id {query_image_id}, descriptor '{name}', dataset '{dataset}'")
                original = np.load(self.data_dir / "descriptors" / dataset / f"{name}.npy", mmap_mode="r")
                raw_query = np.asarray(original[q_index], dtype=np.float32)
            else:
                continue
            query = self._project(raw_query, registered.pca_mean, registered.pca_components,
                                  normalize=name == "vgg16")
            loaded[name] = (registered, query, weight)

            if candidate_count <= SMALL_CANDIDATE_LIMIT:
                neighbors = self._small_subset_neighbors(registered, query, filtered_ids, k)
            else:
                neighbors = self._tree_filtered_neighbors(registered, query, candidate_set, k)
            for row, distance in neighbors:
                candidate_pool.add(int(registered.image_ids[row]))

        if not candidate_pool:
            return []
        pool_ids = sorted(candidate_pool)
        distance_by_image: dict[int, dict[str, float]] = {image_id: {} for image_id in pool_ids}
        weights_by_image: dict[int, dict[str, float]] = {image_id: {} for image_id in pool_ids}
        for name, (registered, query, weight) in loaded.items():
            rows = self._candidate_rows(registered, pool_ids)
            if not len(rows):
                continue
            vectors = np.asarray(registered.tree.vectors[rows], dtype=np.float32)
            if registered.tree.metric == "l1":
                distances = np.abs(vectors - query).sum(axis=1, dtype=np.float32)
            else:
                delta = vectors - query
                distances = np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float32))
            for row, distance in zip(rows, distances):
                image_id = int(registered.image_ids[row])
                distance_by_image[image_id][name] = float(distance)
                weights_by_image[image_id][name] = weight

        normalized_by_image: dict[int, float] = {}
        descriptor_values: dict[int, dict[str, float | None]] = {}
        for image_id, per_descriptor in distance_by_image.items():
            weighted_sum = 0.0
            weight_sum = 0.0
            details: dict[str, float | None] = {}
            for name, raw_distance in per_descriptor.items():
                registered = self.registry.get(dataset, name)
                normalized = (raw_distance - registered.tree.distance_mean) / registered.tree.distance_std
                weight = weights_by_image[image_id][name]
                weighted_sum += weight * normalized
                weight_sum += weight
                details[name] = raw_distance
            for name, _ in selections:
                details.setdefault(name, None)
            normalized_by_image[image_id] = weighted_sum / max(weight_sum, 1e-12)
            descriptor_values[image_id] = details

        ranked_ids = sorted(normalized_by_image, key=lambda image_id: (normalized_by_image[image_id], image_id))[:k]
        images = db.query(Image).filter(Image.id.in_(ranked_ids)).all()
        by_id = {image.id: image for image in images}
        return [SearchHit(by_id[image_id], normalized_by_image[image_id], descriptor_values[image_id])
                for image_id in ranked_ids if image_id in by_id]
