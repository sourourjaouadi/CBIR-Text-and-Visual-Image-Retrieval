"""Build per-descriptor flat VP-Trees with optional sample-fitted PCA."""
import argparse
from pathlib import Path
import numpy as np

from backend.app.config import get_settings
from backend.app.core.index.vptree import VPTree


def estimate_distance_stats(vectors: np.ndarray, metric: str, sample_pairs: int = 50_000,
                            seed: int = 29) -> tuple[float, float]:
    """Estimate a corpus-level distance scale from random vector pairs."""
    if len(vectors) < 2:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    count = min(sample_pairs, max(1, len(vectors)))
    left = rng.integers(0, len(vectors), size=count)
    right = rng.integers(0, len(vectors), size=count)
    distances = np.empty(count, dtype=np.float32)
    for start in range(0, count, 50_000):
        end = min(start + 50_000, count)
        delta = np.asarray(vectors[left[start:end]], dtype=np.float32) - np.asarray(vectors[right[start:end]], dtype=np.float32)
        if metric == "l1":
            distances[start:end] = np.abs(delta).sum(axis=1, dtype=np.float32)
        else:
            distances[start:end] = np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float32))
    mean = float(distances.mean(dtype=np.float64))
    std = float(distances.std(dtype=np.float64))
    return mean, max(std, 1e-8)


def build_one(source: Path, output_dir: Path, metric: str, leaf_size: int,
              pca_dim: int | None, seed: int, pca_sample_size: int = 50_000) -> None:
    vectors = np.load(source, mmap_mode="r")
    if vectors.dtype != np.float32:
        raise ValueError(f"Expected float32 feature matrix: {source}")
    mean = components = None
    indexed_path = source
    indexed_vectors = vectors
    if pca_dim is not None:
        effective_pca_dim = min(pca_dim, vectors.shape[1], max(1, len(vectors) - 1))
        if effective_pca_dim != pca_dim:
            print(f"{source.stem}: reducing PCA dimension from {pca_dim} to {effective_pca_dim} for this corpus")
        pca_dim = effective_pca_dim
        mean, components = VPTree.fit_pca(vectors, pca_dim, sample_size=pca_sample_size, seed=seed)
        indexed_path = output_dir / f"{source.stem}.projected.npy"
        projected = np.lib.format.open_memmap(indexed_path, mode="w+", dtype=np.float32,
                                               shape=(len(vectors), pca_dim))
        VPTree.project(vectors, mean, components, output=projected)
        if source.stem == "vgg16":
            for start in range(0, len(projected), 8192):
                end = min(len(projected), start + 8192)
                norms = np.linalg.norm(projected[start:end], axis=1, keepdims=True)
                np.divide(projected[start:end], norms, out=projected[start:end], where=norms > 0)
            projected.flush()
        else:
            projected.flush()
        indexed_vectors = np.load(indexed_path, mmap_mode="r")
    else:
        stale_projection = output_dir / f"{source.stem}.projected.npy"
        if stale_projection.exists():
            stale_projection.unlink()
    distance_mean, distance_std = estimate_distance_stats(indexed_vectors, metric, seed=seed)
    tree = VPTree(indexed_vectors, metric=metric, leaf_size=leaf_size, seed=seed)
    tree.save(output_dir / f"{source.stem}.npz", pca_mean=mean, pca_components=components,
              distance_mean=distance_mean, distance_std=distance_std)
    print(f"indexed {source.stem}: {len(vectors)} rows, dim={indexed_vectors.shape[1]}, "
          f"leaves={tree.leaf_count}, tree_metadata={tree.memory_bytes / 1024**2:.2f} MiB, "
          f"distance_mean={distance_mean:.5f}, distance_std={distance_std:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="objects")
    parser.add_argument("--metric", choices=("l1", "l2"), default=None,
                        help="apply one metric to every descriptor (overrides descriptor defaults)")
    parser.add_argument("--descriptor-metric", action="append", default=[], metavar="NAME=METRIC",
                        help="override one descriptor metric; can be repeated")
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--pca-sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    metric_overrides: dict[str, str] = {}
    for item in args.descriptor_metric:
        try:
            name, metric = item.split("=", maxsplit=1)
        except ValueError:
            parser.error(f"Invalid --descriptor-metric {item!r}; expected NAME=l1 or NAME=l2")
        if metric not in {"l1", "l2"}:
            parser.error(f"Invalid metric {metric!r}; use l1 or l2")
        metric_overrides[name] = metric
    root = get_settings().data_dir
    output_dir = root / "indexes" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((root / "descriptors" / args.dataset).glob("*.npy")):
        if source.name.endswith((".projected.npy", ".partial.npy")):
            continue
        pca_dim = args.pca_dim if args.pca_dim is not None else (128 if source.stem == "vgg16" else None)
        from backend.app.core.descriptors import DESCRIPTORS
        default_metric = DESCRIPTORS[source.stem].metric if source.stem in DESCRIPTORS else "l2"
        metric = args.metric or metric_overrides.get(source.stem, default_metric)
        build_one(source, output_dir, metric, args.leaf_size, pca_dim, args.seed,
                  args.pca_sample_size)


if __name__ == "__main__":
    main()
