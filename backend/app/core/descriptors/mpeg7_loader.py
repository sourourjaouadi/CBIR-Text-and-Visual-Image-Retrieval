"""Memory-bounded loader for row-oriented numeric MPEG-7 exports."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _parse_line(line: str, delimiter: str | None) -> np.ndarray | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "%")):
        return None
    values = np.fromstring(stripped, dtype=np.float32, sep=delimiter or " ")
    if delimiter:
        # np.fromstring supports one separator, so normalize common CSV whitespace.
        if delimiter != " ":
            values = np.fromstring(stripped.replace(delimiter, " "), dtype=np.float32, sep=" ")
    if not values.size:
        raise ValueError(f"Non-numeric descriptor row: {stripped[:120]!r}")
    return values


def load_mpeg7_text(source: Path, output: Path, delimiter: str | None = None,
                    image_ids: np.ndarray | None = None) -> np.memmap:
    """Stream a numeric text export into float32 ``.npy`` and save aligned IDs.

    The export must contain one descriptor vector per line. IDs must be supplied
    explicitly in the same row order; the loader deliberately makes no implicit
    row-to-database-ID assumption.

    TODO: add parsers for the project's exact XML/binary MPEG-7 export format once
    representative source files and their image-ID manifest are available.
    """
    source, output = Path(source), Path(output)
    if image_ids is None:
        raise ValueError("Provide image_ids in exactly the descriptor row order")

    rows = 0
    dimension: int | None = None
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            values = _parse_line(line, delimiter)
            if values is None:
                continue
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise ValueError(f"Descriptor dimension mismatch at line {line_number}: "
                                 f"expected {dimension}, found {len(values)}")
            rows += 1
    if dimension is None or rows == 0:
        raise ValueError(f"No numeric descriptor rows found in {source}")

    ids = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    if len(ids) != rows:
        raise ValueError(f"IDs row count ({len(ids)}) does not match descriptors ({rows})")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("image_ids must be unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                       shape=(rows, dimension))
    row = 0
    with source.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            values = _parse_line(line, delimiter)
            if values is not None:
                matrix[row] = values
                row += 1
    matrix.flush()
    id_dir = output.parent / output.stem
    id_dir.mkdir(parents=True, exist_ok=True)
    np.save(id_dir / "ids.npy", ids)
    logger.info("Loaded %d MPEG-7 vectors (dim=%d) to %s", rows, dimension, output)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="descriptor matrix .npy path")
    parser.add_argument("--ids", type=Path, required=True,
                        help="one sorted database image ID per line, aligned with source rows")
    parser.add_argument("--delimiter", default=None,
                        help="optional column delimiter; default accepts whitespace-separated values")
    args = parser.parse_args()
    ids = np.loadtxt(args.ids, dtype=np.int64, ndmin=1)
    load_mpeg7_text(args.source, args.output, delimiter=args.delimiter, image_ids=ids)


if __name__ == "__main__":
    main()
