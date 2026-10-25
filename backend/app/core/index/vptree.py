"""Iterative, flat-array VP-Tree with vectorized leaf scans."""
from __future__ import annotations

from collections import deque
import heapq
from pathlib import Path
import numpy as np


class VPTree:
    """VP-Tree for float32 vectors, built iteratively and saved as NPZ artifacts."""
    FORMAT_VERSION = 2

    def __init__(self, vectors: np.ndarray, metric: str = "l2", leaf_size: int = 128,
                 sample_size: int = 32, seed: int = 17):
        if metric not in {"l1", "l2"}:
            raise ValueError("metric must be 'l1' or 'l2'")
        if leaf_size < 2:
            raise ValueError("leaf_size must be at least 2")
        self.vectors = vectors if isinstance(vectors, np.memmap) else np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.dtype != np.float32:
            raise ValueError("vectors must be a 2D float32 matrix")
        self.metric, self.leaf_size = metric, int(leaf_size)
        self.sample_size, self.seed = max(2, int(sample_size)), int(seed)
        self._build()

    def _distances(self, rows: np.ndarray, point: np.ndarray) -> np.ndarray:
        delta = rows - point
        if self.metric == "l1":
            return np.abs(delta).sum(axis=1, dtype=np.float32)
        return np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float32))

    def _indexed_distances(self, indices: np.ndarray, point: np.ndarray,
                           chunk_size: int = 65_536) -> np.ndarray:
        """Compute distances from indexed rows without a full N-by-D temporary."""
        output = np.empty(len(indices), dtype=np.float32)
        for start in range(0, len(indices), chunk_size):
            end = min(start + chunk_size, len(indices))
            output[start:end] = self._distances(self.vectors[indices[start:end]], point)
        return output

    def _build(self) -> None:
        n = len(self.vectors)
        self.order = np.arange(n, dtype=np.int32)
        vp, radius, inner_min, inner_max, outer_min, outer_max = [], [], [], [], [], []
        left, right, leaf_start, leaf_end = [], [], [], []
        self.root = -1
        if n == 0:
            self._set_arrays(vp, radius, inner_min, inner_max, outer_min, outer_max,
                             left, right, leaf_start, leaf_end)
            return
        rng = np.random.default_rng(self.seed)
        tasks = deque([(0, n, -1, 0)])
        while tasks:
            start, end, parent, side = tasks.popleft()
            node = len(vp)
            vp.append(-1); radius.append(0.0); inner_min.append(0.0); inner_max.append(0.0)
            outer_min.append(0.0); outer_max.append(0.0); left.append(-1); right.append(-1)
            leaf_start.append(-1); leaf_end.append(-1)
            if parent < 0: self.root = node
            elif side == 0: left[parent] = node
            else: right[parent] = node
            size = end - start
            if size <= self.leaf_size:
                leaf_start[node], leaf_end[node] = start, end
                continue

            rows = self.order[start:end]
            candidate_count = min(size, self.sample_size)
            candidates_pos = rng.choice(size, size=candidate_count, replace=False)
            probe_count = min(size, max(candidate_count, 64))
            probes = self.vectors[rows[rng.choice(size, size=probe_count, replace=False)]]
            candidate_vectors = self.vectors[rows[candidates_pos]]
            delta = candidate_vectors[:, None, :] - probes[None, :, :]
            if self.metric == "l1":
                sample_distances = np.abs(delta).sum(axis=2, dtype=np.float32)
            else:
                sample_distances = np.sqrt(np.einsum("cqd,cqd->cq", delta, delta, dtype=np.float32))
            best_pos = int(candidates_pos[int(np.argmax(sample_distances.var(axis=1, dtype=np.float64)))])
            vantage = int(rows[best_pos])
            vp[node] = vantage
            distances = self._indexed_distances(rows, self.vectors[vantage])
            vantage_pos = int(np.flatnonzero(rows == vantage)[0])
            other_rows = np.concatenate((rows[:vantage_pos], rows[vantage_pos + 1:]))
            other_distances = np.concatenate((distances[:vantage_pos], distances[vantage_pos + 1:]))
            middle = len(other_rows) // 2
            partition = np.argpartition(other_distances, middle)
            ordered_rows = other_rows[partition]
            ordered_distances = other_distances[partition]
            self.order[start] = vantage
            self.order[start + 1:end] = ordered_rows
            boundary = start + 1 + middle
            radius[node] = float(ordered_distances[middle])
            if middle:
                inner_min[node] = float(ordered_distances[:middle].min())
                inner_max[node] = float(ordered_distances[:middle].max())
            if len(ordered_distances) > middle:
                outer_min[node] = float(ordered_distances[middle:].min())
                outer_max[node] = float(ordered_distances[middle:].max())
            tasks.append((start + 1, boundary, node, 0))
            tasks.append((boundary, end, node, 1))
        self._set_arrays(vp, radius, inner_min, inner_max, outer_min, outer_max,
                         left, right, leaf_start, leaf_end)

    def _set_arrays(self, vp, radius, inner_min, inner_max, outer_min, outer_max,
                    left, right, leaf_start, leaf_end):
        self.vp_index = np.asarray(vp, dtype=np.int32)
        self.radius = np.asarray(radius, dtype=np.float32)
        self.inner_min = np.asarray(inner_min, dtype=np.float32)
        self.inner_max = np.asarray(inner_max, dtype=np.float32)
        self.outer_min = np.asarray(outer_min, dtype=np.float32)
        self.outer_max = np.asarray(outer_max, dtype=np.float32)
        self.left = np.asarray(left, dtype=np.int32)
        self.right = np.asarray(right, dtype=np.int32)
        self.leaf_start = np.asarray(leaf_start, dtype=np.int64)
        self.leaf_end = np.asarray(leaf_end, dtype=np.int64)

    @property
    def leaf_count(self) -> int:
        return int(np.count_nonzero(self.leaf_start >= 0))

    @property
    def memory_bytes(self) -> int:
        return int(sum(a.nbytes for a in (self.order, self.vp_index, self.radius, self.left,
                                          self.inner_min, self.inner_max, self.outer_min,
                                          self.outer_max, self.right, self.leaf_start, self.leaf_end)))

    def knn_search(self, query: np.ndarray, k: int, max_visited_leaves: int | None = None,
                   exclude_index: int | None = None) -> list[tuple[int, float]]:
        """Exact by default; set a leaf cap for approximate best-first traversal."""
        if k <= 0 or self.root < 0: return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.size != self.vectors.shape[1]:
            raise ValueError(f"query dimension {query.size} != index dimension {self.vectors.shape[1]}")
        cap = None if max_visited_leaves is None else max(0, int(max_visited_leaves))
        if cap == 0: return []
        frontier: list[tuple[float, int]] = [(0.0, self.root)]
        # Negative (distance, row) turns heapq into a max-heap whose root is
        # the current worst neighbor, with larger row IDs losing equal ties.
        best: list[tuple[float, int]] = []
        visited = 0

        def offer(index: int, distance: float) -> None:
            if exclude_index == index or not np.isfinite(distance): return
            entry = (-distance, -index)
            if len(best) < k:
                heapq.heappush(best, entry)
            elif entry > best[0]:
                heapq.heapreplace(best, entry)

        while frontier:
            bound, node = heapq.heappop(frontier)
            tau = -best[0][0] if len(best) == k else float("inf")
            if len(best) == k and bound > tau + 1e-6: break
            leaf_from = int(self.leaf_start[node])
            if leaf_from >= 0:
                if cap is not None and visited >= cap: break
                visited += 1
                leaf_to = int(self.leaf_end[node])
                indices = self.order[leaf_from:leaf_to]
                distances = self._distances(self.vectors[indices], query)
                if exclude_index is not None: distances[indices == exclude_index] = np.inf
                take_count = min(k, len(distances))
                if take_count < len(distances):
                    partial = np.argpartition(distances, take_count - 1)[:take_count]
                    cutoff = float(distances[partial].max())
                    # Keep all boundary ties so final ordering by row ID is stable.
                    take = np.flatnonzero(distances <= cutoff)
                else: take = np.arange(len(distances))
                for pos in take: offer(int(indices[pos]), float(distances[pos]))
                continue

            vantage = int(self.vp_index[node])
            distance = float(self._distances(self.vectors[vantage:vantage + 1], query)[0])
            offer(vantage, distance)
            radius = float(self.radius[node])
            # Child distance intervals retain exact bounds despite median ties.
            def interval_bound(low: float, high: float) -> float:
                if distance < low: value = low - distance
                elif distance > high: value = distance - high
                else: return 0.0
                return max(0.0, value - 1e-5 * (1.0 + abs(distance) + abs(high)))
            children = ((int(self.left[node]), interval_bound(float(self.inner_min[node]), float(self.inner_max[node]))),
                        (int(self.right[node]), interval_bound(float(self.outer_min[node]), float(self.outer_max[node]))))
            for child, lower_bound in children:
                if child >= 0 and (len(best) < k or lower_bound <= tau):
                    heapq.heappush(frontier, (lower_bound, child))
        ranked = sorted(((-neg_index, -neg_distance) for neg_distance, neg_index in best),
                        key=lambda item: (item[1], item[0]))
        return ranked

    def save(self, path: Path, pca_mean: np.ndarray | None = None,
             pca_components: np.ndarray | None = None, distance_mean: float = 0.0,
             distance_std: float = 1.0) -> None:
        """Save tree arrays and optional PCA transform in a compressed NPZ archive."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, format_version=np.asarray([self.FORMAT_VERSION], np.int32),
            metric=np.asarray(self.metric), leaf_size=np.asarray([self.leaf_size], np.int32),
            root=np.asarray([self.root], np.int32), order=self.order, vp_index=self.vp_index,
            radius=self.radius, left=self.left, right=self.right, leaf_start=self.leaf_start,
            leaf_end=self.leaf_end, inner_min=self.inner_min, inner_max=self.inner_max,
            outer_min=self.outer_min, outer_max=self.outer_max,
            pca_mean=np.asarray([] if pca_mean is None else pca_mean, np.float32),
            pca_components=np.asarray([] if pca_components is None else pca_components, np.float32),
            distance_mean=np.asarray([distance_mean], np.float32),
            distance_std=np.asarray([max(float(distance_std), 1e-8)], np.float32))

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray) -> "VPTree":
        """Load tree arrays from NPZ and leave vectors backed by the caller's memmap."""
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["format_version"][0]) != cls.FORMAT_VERSION:
                raise ValueError(f"Unsupported VP-Tree format: {path}")
            tree = cls.__new__(cls)
            tree.metric = str(archive["metric"].item())
            tree.leaf_size = int(archive["leaf_size"][0])
            tree.root = int(archive["root"][0])
            for name in ("order", "vp_index", "radius", "inner_min", "inner_max", "outer_min",
                         "outer_max", "left", "right", "leaf_start", "leaf_end"):
                setattr(tree, name, archive[name].copy())
            tree.pca_mean = archive["pca_mean"].copy()
            tree.pca_components = archive["pca_components"].copy()
            tree.distance_mean = float(archive["distance_mean"][0])
            tree.distance_std = max(float(archive["distance_std"][0]), 1e-8)
        tree.vectors = vectors
        tree.seed, tree.sample_size = 0, 32
        return tree

    @staticmethod
    def fit_pca(vectors: np.ndarray, output_dim: int, sample_size: int = 50_000,
                seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
        """Fit PCA with SVD on a bounded random sample."""
        if output_dim <= 0 or output_dim > vectors.shape[1]:
            raise ValueError("pca_dim must be between 1 and input dimension")
        rng = np.random.default_rng(seed)
        sample_count = min(len(vectors), max(output_dim + 1, sample_size))
        sample = np.asarray(vectors[rng.choice(len(vectors), sample_count, replace=False)], dtype=np.float32)
        mean = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
        _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
        return mean, vt[:output_dim].astype(np.float32)

    @staticmethod
    def project(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray,
                output: np.ndarray | None = None, chunk_size: int = 100_000) -> np.ndarray:
        """Project in chunks, allowing direct writes into an output memmap."""
        if output is None: output = np.empty((len(vectors), components.shape[0]), dtype=np.float32)
        for start in range(0, len(vectors), chunk_size):
            end = min(start + chunk_size, len(vectors))
            output[start:end] = (np.asarray(vectors[start:end], dtype=np.float32) - mean) @ components.T
        return output
