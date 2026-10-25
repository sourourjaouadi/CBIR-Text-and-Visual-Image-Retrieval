"""Compute object descriptors into aligned NumPy descriptor matrices."""
import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from backend.app.config import get_settings
from backend.app.core.descriptors import DESCRIPTORS
from backend.app.db.models import Image
from backend.app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def compute_vgg16(root: Path, dataset: str, rows: list[tuple[int, Path]], limit: int | None,
                  batch_size: int, device: str | None, num_workers: int | None) -> None:
    """Resumably extract VGG16 features, compacting valid rows at completion."""
    from backend.app.core.descriptors.vgg16 import VGG16Descriptor

    descriptor = VGG16Descriptor(device=device, num_workers=num_workers)
    dimension = 4096 if descriptor.layer == "fc7" else 512
    if limit is not None:
        rows = rows[:limit]
    output_dir = root / "descriptors" / dataset
    id_dir = output_dir / "vgg16"
    output_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "vgg16.npy"
    partial = output_dir / "vgg16.partial.npy"
    partial_ids = id_dir / "ids.partial.npy"
    checkpoint = id_dir / "checkpoint.json"
    metadata = {"dataset": dataset, "rows": len(rows), "dimension": dimension,
                "layer": descriptor.layer}
    last_input = -1
    written = 0
    if checkpoint.exists() and partial.exists() and partial_ids.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError("VGG16 checkpoint settings differ from this run; remove checkpoint artifacts to restart")
        last_input = int(saved["last_completed_input_row"])
        written = int(saved["written_rows"])
        vectors = np.load(partial, mmap_mode="r+")
        ids = np.load(partial_ids, mmap_mode="r+")
        logger.info("Resuming VGG16 at input row %d (%d vectors written)", last_input + 1, written)
    else:
        vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                             shape=(len(rows), dimension))
        ids = np.lib.format.open_memmap(partial_ids, mode="w+", dtype=np.int64,
                                        shape=(len(rows),))

    chunk_size = max(1, batch_size)
    for start in tqdm(range(last_input + 1, len(rows), chunk_size), desc="VGG16", unit="batch"):
        chunk = rows[start:start + chunk_size]
        features, valid_paths = descriptor.extract_batch([path for _, path in chunk], batch_size=chunk_size)
        ids_by_path = {path.resolve(): image_id for image_id, path in chunk}
        valid_ids = [ids_by_path[path.resolve()] for path in valid_paths]
        invalid = set(ids_by_path) - {path.resolve() for path in valid_paths}
        for path in invalid:
            logger.warning("Skipping unreadable image: %s", path)
        end = written + len(features)
        vectors[written:end] = features
        ids[written:end] = valid_ids
        vectors.flush(); ids.flush()
        written = end
        last_input = start + len(chunk) - 1
        checkpoint.write_text(json.dumps({**metadata, "last_completed_input_row": last_input,
                                          "written_rows": written}), encoding="utf-8")

    # Compact successful rows into the final NumPy file, since corrupt files are omitted.
    final_vectors = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32,
                                               shape=(written, dimension))
    final_ids = np.lib.format.open_memmap(id_dir / "ids.npy", mode="w+", dtype=np.int64,
                                           shape=(written,))
    for start in range(0, written, 8192):
        end = min(written, start + 8192)
        final_vectors[start:end] = vectors[start:end]
        final_ids[start:end] = ids[start:end]
    final_vectors.flush(); final_ids.flush()
    del final_vectors, final_ids, vectors, ids
    partial.unlink(missing_ok=True); partial_ids.unlink(missing_ok=True); checkpoint.unlink(missing_ok=True)
    print(f"vgg16: {written} vectors -> {target}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="objects"); parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument("--descriptor", choices=[*DESCRIPTORS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(); root = get_settings().data_dir
    with SessionLocal() as db:
        images = db.query(Image).filter_by(dataset=args.dataset).order_by(Image.id).all()
        image_ids = [image.id for image in images]
        paths = [(image_id, root / "images" / image.url) for image_id, image in zip(image_ids, images)]
    # Keep routine demo/seed runs offline; neural weights are downloaded only by explicit request.
    selected = ({name: descriptor for name, descriptor in DESCRIPTORS.items() if name != "vgg16"}
                if args.descriptor == "all" else {args.descriptor: DESCRIPTORS[args.descriptor]})
    if "vgg16" in selected:
        compute_vgg16(root, args.dataset, paths, args.limit, args.batch_size, args.device, args.num_workers)
    for name, descriptor in selected.items():
        if name == "vgg16":
            continue
        rows = []
        row_ids = []
        for image_id, path in paths[:args.limit]:
            image = cv2.imread(str(path))
            if image is not None:
                rows.append(descriptor.extract(image))
                row_ids.append(image_id)
        if rows:
            output = root / "descriptors" / args.dataset / f"{name}.npy"; output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, np.stack(rows).astype(np.float32))
            id_dir = root / "descriptors" / args.dataset / name
            id_dir.mkdir(parents=True, exist_ok=True)
            np.save(id_dir / "ids.npy", np.asarray(row_ids, dtype=np.int64))
            print(f"{name}: {len(rows)} vectors -> {output}")

if __name__ == "__main__": main()
