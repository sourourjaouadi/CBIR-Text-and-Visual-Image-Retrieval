"""Brute-force exact kNN baseline."""
import numpy as np

def knn_search(vectors: np.ndarray, query: np.ndarray, k: int, metric: str = "l2") -> list[tuple[int, float]]:
    delta = np.asarray(vectors, dtype=np.float32) - np.asarray(query, dtype=np.float32)
    if metric == "l1":
        distances = np.abs(delta).sum(axis=1, dtype=np.float32)
    elif metric == "l2":
        distances = np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float32))
    else:
        raise ValueError("metric must be 'l1' or 'l2'")
    if k <= 0 or not len(distances): return []
    ids = np.argpartition(distances, min(k, len(distances)) - 1)[:k]
    return sorted(((int(i), float(distances[i])) for i in ids), key=lambda item: (item[1], item[0]))
