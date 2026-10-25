"""Evaluate descriptor and descriptor-combination retrieval against label relevance."""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import func

from backend.app.config import get_settings
from backend.app.core.index.brute_force import knn_search as brute_force
from backend.app.core.index.vptree import VPTree
from backend.app.db.models import Image
from backend.app.db.session import SessionLocal


def _project(raw: np.ndarray, tree: VPTree, descriptor_name: str = "") -> np.ndarray:
    if tree.pca_components.size:
        projected = (raw - tree.pca_mean) @ tree.pca_components.T
        if descriptor_name == "vgg16":
            norm = float(np.linalg.norm(projected))
            if norm > 0:
                projected = projected / norm
        return projected
    return raw


def _fetch_labels(db, ids: set[int], dataset: str) -> dict[int, str | None]:
    if not ids:
        return {}
    found: dict[int, str | None] = {}
    values = list(ids)
    for start in range(0, len(values), 800):
        rows = (db.query(Image.id, Image.label).filter(Image.dataset == dataset,
                Image.id.in_(values[start:start + 800])).all())
        found.update({int(image_id): label for image_id, label in rows})
    return found


def _id_row_map(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact lookup index from external IDs to descriptor rows."""
    if len(ids) < 2 or np.all(ids[:-1] <= ids[1:]):
        return ids, np.arange(len(ids), dtype=np.int32)
    rows = np.argsort(ids, kind="stable").astype(np.int32, copy=False)
    return np.asarray(ids[rows], dtype=np.int64), rows


def _row_for_id(mapping: tuple[np.ndarray, np.ndarray], image_id: int) -> int | None:
    sorted_ids, sorted_rows = mapping
    position = int(np.searchsorted(sorted_ids, image_id))
    if position >= len(sorted_ids) or int(sorted_ids[position]) != image_id:
        return None
    return int(sorted_rows[position])


def _metrics(ranked_ids: list[int], labels: dict[int, str | None], query_label: str,
             relevant_total: int, k: int) -> tuple[float, float, float, list[float], list[float]]:
    chosen = ranked_ids[:k]
    relevance = [labels.get(image_id) == query_label for image_id in chosen]
    cumulative = np.cumsum(relevance, dtype=np.int64)
    precisions = [float(cumulative[i] / (i + 1)) for i in range(len(relevance))]
    recalls = [float(cumulative[i] / relevant_total) if relevant_total else 0.0 for i in range(len(relevance))]
    precision_at_k = float(sum(relevance) / k)
    recall_at_k = float(sum(relevance) / relevant_total) if relevant_total else 0.0
    average_precision = float(sum(p for p, hit in zip(precisions, relevance) if hit) / min(relevant_total, k)) if relevant_total else 0.0
    return precision_at_k, recall_at_k, average_precision, precisions, recalls


def _rank_combination(dataset: str, descriptor_names: tuple[str, ...], pool_ids: set[int],
                      query_data: dict[str, dict], trees: dict[str, VPTree],
                      ids_by_descriptor: dict[str, np.ndarray],
                      row_maps: dict[str, tuple[np.ndarray, np.ndarray]], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for image_id in pool_ids:
        total = 0.0
        used = 0
        for name in descriptor_names:
            ids = ids_by_descriptor[name]
            row = _row_for_id(row_maps[name], image_id)
            if row is None:
                continue
            query = query_data[name]["query"]
            vector = trees[name].vectors[row:row + 1]
            if trees[name].metric == "l1":
                distance = float(np.abs(vector - query).sum(axis=1, dtype=np.float32)[0])
            else:
                delta = vector - query
                distance = float(np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float32)[0]))
            total += (distance - trees[name].distance_mean) / trees[name].distance_std
            used += 1
        if used:
            scores[image_id] = total / used
    return sorted(scores, key=lambda image_id: (scores[image_id], image_id))[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="objects")
    parser.add_argument("--queries", type=int, default=100, help="random query image count")
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--descriptors", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = get_settings().data_dir
    output = args.output or root / "evaluation" / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    descriptor_root = root / "descriptors" / args.dataset
    candidates = sorted(path.stem for path in descriptor_root.glob("*.npy")
                        if not path.name.endswith((".projected.npy", ".partial.npy")))
    names = args.descriptors or candidates
    if not names:
        parser.error(f"No descriptor matrices found for dataset {args.dataset}")
    trees: dict[str, VPTree] = {}
    ids_by_descriptor: dict[str, np.ndarray] = {}
    row_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        archive = root / "indexes" / args.dataset / f"{name}.npz"
        projected = root / "indexes" / args.dataset / f"{name}.projected.npy"
        matrix = projected if projected.exists() else descriptor_root / f"{name}.npy"
        if not archive.exists() or not matrix.exists():
            parser.error(f"Missing index artifacts for descriptor {name}; run build_indexes.py first")
        trees[name] = VPTree.load(archive, vectors=np.load(matrix, mmap_mode="r"))
        ids_by_descriptor[name] = np.load(descriptor_root / name / "ids.npy", mmap_mode="r")
        row_maps[name] = _id_row_map(ids_by_descriptor[name])

    query_ids_source = ids_by_descriptor[names[0]]
    rng = np.random.default_rng(args.seed)
    sampled_rows = rng.choice(len(query_ids_source), size=min(args.queries, len(query_ids_source)), replace=False)
    query_ids = [int(query_ids_source[row]) for row in sampled_rows]
    configs = [combo for length in range(1, len(names) + 1) for combo in combinations(names, length)]
    cutoffs = list(range(1, args.k + 1))
    curve_values: dict[tuple[str, int], list[tuple[float, float]]] = {}
    csv_rows: list[dict[str, object]] = []

    with SessionLocal() as db:
        label_counts = dict(db.query(Image.label, func.count(Image.id)).filter(
            Image.dataset == args.dataset, Image.label.is_not(None)).group_by(Image.label).all())
        for query_id in query_ids:
            labels = _fetch_labels(db, {query_id}, args.dataset)
            query_label = labels.get(query_id)
            if query_label is None:
                continue
            q_data: dict[str, dict] = {}
            query_result_ids: set[int] = set()
            for name in names:
                ids = ids_by_descriptor[name]
                row = _row_for_id(row_maps[name], query_id)
                if row is None:
                    continue
                tree = trees[name]
                original = np.load(descriptor_root / f"{name}.npy", mmap_mode="r")
                query = _project(np.asarray(original[row], dtype=np.float32), tree, name)
                pool_k = min(len(ids) - 1, max(args.k, args.k * 10))
                started = perf_counter()
                vp_neighbors = tree.knn_search(query, pool_k, exclude_index=row)
                vp_ms = (perf_counter() - started) * 1000
                started = perf_counter()
                brute_neighbors = brute_force(tree.vectors, query, min(len(ids), pool_k + 1), metric=tree.metric)
                brute_neighbors = [(idx, dist) for idx, dist in brute_neighbors if idx != row][:pool_k]
                brute_ms = (perf_counter() - started) * 1000
                q_data[name] = {"query": query, "vp": vp_neighbors, "brute": brute_neighbors,
                                "vp_ms": vp_ms, "brute_ms": brute_ms, "row": row}
                query_result_ids.update(int(ids[idx]) for idx, _ in vp_neighbors)
                query_result_ids.update(int(ids[idx]) for idx, _ in brute_neighbors)
            if not q_data:
                continue
            result_labels = _fetch_labels(db, query_result_ids, args.dataset)
            relevant_total = max(0, int(label_counts.get(query_label, 0)) - 1)
            for combo in configs:
                available_combo = tuple(name for name in combo if name in q_data)
                if not available_combo:
                    continue
                if len(available_combo) == 1:
                    name = available_combo[0]
                    ranked = [int(ids_by_descriptor[name][idx]) for idx, _ in q_data[name]["vp"]]
                    vp_ms = float(q_data[name]["vp_ms"])
                    brute_ranked = [int(ids_by_descriptor[name][idx]) for idx, _ in q_data[name]["brute"]]
                    brute_ms = float(q_data[name]["brute_ms"])
                else:
                    pool = set().union(*(set(int(ids_by_descriptor[name][idx]) for idx, _ in q_data[name]["vp"])
                                         for name in available_combo))
                    ranked = _rank_combination(args.dataset, available_combo, pool, q_data, trees,
                                               ids_by_descriptor, row_maps, args.k)
                    vp_ms = sum(float(q_data[name]["vp_ms"]) for name in available_combo)
                    brute_pool = set().union(*(set(int(ids_by_descriptor[name][idx]) for idx, _ in q_data[name]["brute"])
                                               for name in available_combo))
                    brute_ranked = _rank_combination(args.dataset, available_combo, brute_pool, q_data,
                                                     trees, ids_by_descriptor, row_maps, args.k)
                    brute_ms = sum(float(q_data[name]["brute_ms"]) for name in available_combo)
                p_at_k, r_at_k, ap_at_k, precisions, recalls = _metrics(
                    ranked, result_labels, query_label, relevant_total, args.k)
                brute_p_at_k, brute_r_at_k, brute_ap_at_k, _, _ = _metrics(
                    brute_ranked, result_labels, query_label, relevant_total, args.k)
                config_name = "+".join(available_combo)
                csv_rows.append({"query_image_id": query_id, "label": query_label,
                    "descriptors": config_name, "precision_at_k": p_at_k,
                    "recall_at_k": r_at_k, "ap_at_k": ap_at_k,
                    "brute_precision_at_k": brute_p_at_k,
                    "brute_recall_at_k": brute_r_at_k,
                    "brute_ap_at_k": brute_ap_at_k,
                    "vp_tree_ms": vp_ms, "brute_force_ms": brute_ms})
                for cutoff, precision, recall in zip(cutoffs, precisions, recalls):
                    curve_values.setdefault((config_name, cutoff), []).append((precision, recall))

    csv_path = output / "metrics.csv"
    fields = ["query_image_id", "label", "descriptors", "precision_at_k", "recall_at_k",
              "ap_at_k", "brute_precision_at_k", "brute_recall_at_k", "brute_ap_at_k",
              "vp_tree_ms", "brute_force_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(csv_rows)

    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        summary_fields = ["descriptors", "query_count", "mean_precision_at_k", "mean_recall_at_k",
                          "map_at_k", "brute_mean_precision_at_k", "brute_mean_recall_at_k",
                          "brute_map_at_k", "mean_vp_tree_ms", "mean_brute_force_ms"]
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for config_name in sorted({str(row["descriptors"]) for row in csv_rows}):
            group = [row for row in csv_rows if row["descriptors"] == config_name]
            writer.writerow({"descriptors": config_name, "query_count": len(group),
                "mean_precision_at_k": float(np.mean([float(row["precision_at_k"]) for row in group])),
                "mean_recall_at_k": float(np.mean([float(row["recall_at_k"]) for row in group])),
                "map_at_k": float(np.mean([float(row["ap_at_k"]) for row in group])),
                "brute_mean_precision_at_k": float(np.mean([float(row["brute_precision_at_k"]) for row in group])),
                "brute_mean_recall_at_k": float(np.mean([float(row["brute_recall_at_k"]) for row in group])),
                "brute_map_at_k": float(np.mean([float(row["brute_ap_at_k"]) for row in group])),
                "mean_vp_tree_ms": float(np.mean([float(row["vp_tree_ms"]) for row in group])),
                "mean_brute_force_ms": float(np.mean([float(row["brute_force_ms"]) for row in group]))})

    fig, ax = plt.subplots(figsize=(9, 6))
    config_names = ["+".join(combo) for combo in configs]
    for config_name in config_names:
        points = [curve_values.get((config_name, cutoff), []) for cutoff in cutoffs]
        points = [point for point in points if point]
        if points:
            ax.plot([float(np.mean([p for p, _ in group])) for group in points],
                    [float(np.mean([r for _, r in group])) for group in points],
                    marker=".", label=config_name)
    ax.set(xlabel="Precision", ylabel="Recall", title=f"Precision–Recall by cutoff (k={args.k})")
    ax.grid(True, alpha=.25); ax.legend(fontsize="small", ncol=2); fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=160); plt.close(fig)

    if csv_rows:
        names_for_plot = sorted({str(row["descriptors"]) for row in csv_rows})
        vp_means = [np.mean([float(row["vp_tree_ms"]) for row in csv_rows if row["descriptors"] == name]) for name in names_for_plot]
        brute_means = [np.mean([float(row["brute_force_ms"]) for row in csv_rows if row["descriptors"] == name]) for name in names_for_plot]
        positions = np.arange(len(names_for_plot)); width = .38
        fig, ax = plt.subplots(figsize=(max(9, len(names_for_plot) * .5), 6))
        ax.bar(positions - width / 2, vp_means, width, label="VP-Tree")
        ax.bar(positions + width / 2, brute_means, width, label="Brute force")
        ax.set_ylabel("Average query time (ms)"); ax.set_title("Query time by descriptor configuration")
        ax.set_xticks(positions, names_for_plot, rotation=70, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); fig.savefig(output / "query_time.png", dpi=160); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.text(.5, .5, "No labeled queries available", ha="center", va="center")
        ax.set_axis_off(); fig.tight_layout(); fig.savefig(output / "query_time.png", dpi=160); plt.close(fig)
    print(f"Saved {len(csv_rows)} per-query metric rows to {csv_path}")
    print(f"Plots: {output / 'precision_recall.png'}, {output / 'query_time.png'}")


if __name__ == "__main__":
    main()
