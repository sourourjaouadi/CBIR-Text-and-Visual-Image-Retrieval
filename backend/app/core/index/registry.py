"""Lazy per-dataset, per-descriptor index registry."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from backend.app.core.index.vptree import VPTree

@dataclass
class RegisteredIndex:
    tree: VPTree
    image_ids: np.ndarray
    sorted_image_ids: np.ndarray
    sorted_rows: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray

    def row_for_image_id(self, image_id: int) -> int | None:
        """Resolve an external image ID even when source rows are not ID-sorted."""
        position = int(np.searchsorted(self.sorted_image_ids, image_id))
        if position >= len(self.sorted_image_ids) or int(self.sorted_image_ids[position]) != image_id:
            return None
        return int(self.sorted_rows[position])

class IndexRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._loaded: dict[tuple[str, str], RegisteredIndex] = {}
    def get(self, dataset: str, descriptor: str) -> RegisteredIndex | None:
        key = (dataset, descriptor)
        if key not in self._loaded:
            path = self.root / dataset / f"{descriptor}.npz"
            vectors_path = self.root / dataset / f"{descriptor}.projected.npy"
            if not vectors_path.exists():
                vectors_path = self.root.parent / "descriptors" / dataset / f"{descriptor}.npy"
            if not path.exists() or not vectors_path.exists(): return None
            ids_path = self.root.parent / "descriptors" / dataset / descriptor / "ids.npy"
            if not ids_path.exists(): return None
            vectors = np.load(vectors_path, mmap_mode="r")
            tree = VPTree.load(path, vectors=vectors)
            image_ids = np.load(ids_path, mmap_mode="r")
            if len(image_ids) != len(tree.vectors):
                raise ValueError(f"Index/ID row count mismatch for {dataset}/{descriptor}")
            if len(image_ids) < 2 or np.all(image_ids[:-1] <= image_ids[1:]):
                sorted_rows = np.arange(len(image_ids), dtype=np.int32)
                sorted_image_ids = image_ids
            else:
                sorted_rows = np.argsort(image_ids, kind="stable").astype(np.int32, copy=False)
                sorted_image_ids = np.asarray(image_ids[sorted_rows], dtype=np.int64)
            self._loaded[key] = RegisteredIndex(tree=tree, image_ids=image_ids,
                sorted_image_ids=sorted_image_ids, sorted_rows=sorted_rows,
                pca_mean=tree.pca_mean, pca_components=tree.pca_components)
        return self._loaded[key]
