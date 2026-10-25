"""Benchmark VP-Tree build, memory, latency, and recall at increasing N."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from backend.app.core.index.brute_force import knn_search as brute_force
from backend.app.core.index.vptree import VPTree


def benchmark_size(n: int, dim: int, queries: int, k: int, metric: str,
                   leaf_size: int, seed: int, workdir: Path) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed + n)
    vector_path = workdir / f"vectors_{n}_{dim}.npy"
    vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(n, dim))
    # Chunk generation avoids an additional full-size temporary matrix.
    for start in range(0, n, 100_000):
        end = min(start + 100_000, n)
        vectors[start:end] = rng.standard_normal((end - start, dim), dtype=np.float32)
    vectors.flush()
    vectors = np.load(vector_path, mmap_mode="r")

    start = perf_counter()
    tree = VPTree(vectors, metric=metric, leaf_size=leaf_size, seed=seed)
    build_seconds = perf_counter() - start
    queries_data = rng.standard_normal((queries, dim), dtype=np.float32)
    vp_times, brute_times, recalls = [], [], []
    for query in queries_data:
        start = perf_counter()
        found = tree.knn_search(query, k)
        vp_times.append(perf_counter() - start)
        start = perf_counter()
        exact = brute_force(vectors, query, k, metric=metric)
        brute_times.append(perf_counter() - start)
        exact_ids = {index for index, _ in exact}
        recalls.append(len(exact_ids.intersection(index for index, _ in found)) / max(1, k))

    archive = workdir / f"tree_{n}_{dim}.npz"
    tree.save(archive)
    result = {
        "n": n, "dim": dim, "metric": metric, "build_seconds": build_seconds,
        "tree_metadata_mib": tree.memory_bytes / 1024**2,
        "vector_matrix_mib": vectors.nbytes / 1024**2,
        "index_archive_mib": archive.stat().st_size / 1024**2,
        "leaf_count": tree.leaf_count,
        "vp_avg_query_ms": 1000 * float(np.mean(vp_times)),
        "brute_avg_query_ms": 1000 * float(np.mean(brute_times)),
        "recall_at_k": float(np.mean(recalls)),
    }
    del tree, vectors
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--metric", choices=("l1", "l2"), default="l2")
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workdir", type=Path, default=Path(".benchmark"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.dim <= 0 or args.dim > 64:
        parser.error("--dim must be between 1 and 64")
    args.workdir.mkdir(parents=True, exist_ok=True)
    results = [benchmark_size(n, args.dim, args.queries, args.k, args.metric,
                              args.leaf_size, args.seed, args.workdir) for n in args.sizes]
    for row in results:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
