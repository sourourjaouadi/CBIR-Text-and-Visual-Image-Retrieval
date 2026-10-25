"""Regression tests for IDs, URL routing, descriptor validation, and query lookup."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api import routes_descriptors, routes_images, routes_search
from backend.app.core.index.vptree import VPTree
from backend.app.db.models import Image, Keyword
from backend.app.db.session import Base
from backend.app.core.search.engine import SearchEngine
from backend.scripts import compute_descriptors, ingest_images


@pytest.fixture
def demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "objects"
    image_dir.mkdir(parents=True)
    for name, color in (("red_0.jpg", (0, 0, 255)), ("green_1.jpg", (0, 255, 0))):
        assert cv2.imwrite(str(image_dir / name), np.full((16, 16, 3), color, dtype=np.uint8))

    db_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(db_engine)
    TestingSession = sessionmaker(bind=db_engine, expire_on_commit=False)
    with TestingSession() as db:
        red = Image(url="objects/red_0.jpg", dataset="objects", label="red_0")
        red.keywords.append(Keyword(value="red"))
        green = Image(url="objects/green_1.jpg", dataset="objects", label="green_1")
        green.keywords.append(Keyword(value="green"))
        db.add_all([red, green]); db.commit()
        ids = [red.id, green.id]

    descriptor_dir = data_dir / "descriptors" / "objects"
    descriptor_dir.mkdir(parents=True)
    # Descriptor rows intentionally differ from image-ID order.
    matrix = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    np.save(descriptor_dir / "color_hsv.npy", matrix)
    id_dir = descriptor_dir / "color_hsv"
    id_dir.mkdir()
    np.save(id_dir / "ids.npy", np.asarray([ids[1], ids[0]], dtype=np.int64))
    index_dir = data_dir / "indexes" / "objects"
    VPTree(matrix).save(index_dir / "color_hsv.npz")

    monkeypatch.setattr(routes_search, "engine", SearchEngine(data_dir))
    monkeypatch.setattr(routes_search, "SessionLocal", TestingSession)
    def image_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()
    monkeypatch.setattr(routes_images, "get_db", image_db)

    configured_data_dir = data_dir
    class TestSettings:
        public_base_url = "http://testserver"
        data_dir = configured_data_dir

    monkeypatch.setattr("backend.app.api.routes_search.get_settings", lambda: TestSettings())
    monkeypatch.setattr("backend.app.api.routes_images.get_settings", lambda: TestSettings())
    monkeypatch.setattr("backend.app.api.routes_descriptors.get_settings", lambda: TestSettings())
    app = FastAPI()
    app.include_router(routes_search.router, prefix="/api")
    app.include_router(routes_images.router, prefix="/api")
    app.include_router(routes_descriptors.router, prefix="/api")
    from fastapi.staticfiles import StaticFiles
    app.mount("/static/images", StaticFiles(directory=data_dir / "images"), name="images")
    return TestClient(app), data_dir, ids, TestingSession


def test_search_maps_rows_using_persisted_image_ids(demo):
    client, _, ids, _ = demo
    response = client.post("/api/search", json={
        "dataset": "objects", "query_image_id": ids[1],
        "descriptors": [{"name": "color_hsv", "weight": 1}], "k": 2,
    })
    assert response.status_code == 200
    assert response.json()["results"][0]["image_id"] == ids[1]


def test_flat_vptree_exact_matches_brute_force_for_l1_and_l2(tmp_path):
    from backend.app.core.index.brute_force import knn_search as brute
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(300, 8)).astype(np.float32)
    for metric in ("l1", "l2"):
        tree = VPTree(vectors, metric=metric, leaf_size=16, seed=2)
        for query in rng.normal(size=(8, 8)).astype(np.float32):
            actual = tree.knn_search(query, 7)
            expected = brute(vectors, query, 7, metric=metric)
            assert np.allclose([d for _, d in actual], [d for _, d in expected])


def test_flat_tree_npz_load_uses_memmap_and_pca(tmp_path):
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(80, 6)).astype(np.float32)
    mean, components = VPTree.fit_pca(vectors, 3, sample_size=40)
    projected_path = tmp_path / "projected.npy"
    projected = np.lib.format.open_memmap(projected_path, mode="w+", dtype=np.float32, shape=(len(vectors), 3))
    VPTree.project(vectors, mean, components, output=projected)
    tree = VPTree(np.load(projected_path, mmap_mode="r"), leaf_size=8)
    archive = tmp_path / "tree.npz"
    tree.save(archive, pca_mean=mean, pca_components=components)
    loaded = VPTree.load(archive, np.load(projected_path, mmap_mode="r"))
    query = (vectors[3] - mean) @ components.T
    assert loaded.knn_search(query, 1)[0][0] == 3
    assert np.array_equal(loaded.pca_components, components)


def test_leaf_cap_bounds_approximate_search_work():
    vectors = np.arange(2048, dtype=np.float32).reshape(-1, 1)
    tree = VPTree(vectors, leaf_size=32)
    assert tree.knn_search(np.array([22.0], np.float32), 5, max_visited_leaves=1)


def test_ingestion_stores_path_relative_to_data_images(tmp_path, monkeypatch):
    from types import SimpleNamespace
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "objects"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "demo_0.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    db_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(db_engine)
    TestingSession = sessionmaker(bind=db_engine, expire_on_commit=False)
    monkeypatch.setattr(ingest_images, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(ingest_images, "Base", Base)
    monkeypatch.setattr(ingest_images, "engine", db_engine)
    monkeypatch.setattr(ingest_images, "SessionLocal", TestingSession)
    monkeypatch.setattr("sys.argv", ["ingest_images", "--dataset", "objects", "--directory", str(image_dir)])
    ingest_images.main()
    with TestingSession() as db:
        assert db.query(Image).one().url == "objects/demo_0.jpg"


def test_descriptor_script_saves_per_descriptor_image_ids(demo, monkeypatch):
    client, data_dir, ids, TestingSession = demo
    del client
    from types import SimpleNamespace
    from backend.scripts import compute_descriptors
    # The fixture images and DB paths are aligned with the fixture's image IDs.
    monkeypatch.setattr(compute_descriptors, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(compute_descriptors, "SessionLocal", TestingSession)
    monkeypatch.setattr("sys.argv", ["compute_descriptors", "--dataset", "objects"])
    compute_descriptors.main()
    for name in ("color_hsv", "lbp", "hog"):
        stored_ids = np.load(data_dir / "descriptors" / "objects" / name / "ids.npy")
        assert stored_ids.tolist() == ids


def test_ingested_path_is_returned_as_public_static_url(demo):
    client, _, ids, _ = demo
    response = client.get("/api/images?dataset=objects")
    assert response.status_code == 200
    image = next(item for item in response.json()["items"] if item["image_id"] == ids[0])
    assert image["url"] == "http://testserver/static/images/objects/red_0.jpg"
    assert client.get("/static/images/objects/red_0.jpg").status_code == 200


def test_unknown_descriptor_returns_400(demo):
    client, _, ids, _ = demo
    response = client.post("/api/search", json={
        "dataset": "objects", "query_image_id": ids[0],
        "descriptors": [{"name": "made_up", "weight": 1}], "k": 2,
    })
    assert response.status_code == 400
    assert "Unknown descriptor 'made_up'" in response.json()["detail"]


def test_missing_index_returns_400(demo):
    client, data_dir, ids, _ = demo
    (data_dir / "indexes" / "objects" / "color_hsv.npz").unlink()
    routes_search.engine = SearchEngine(data_dir)
    response = client.post("/api/search", json={
        "dataset": "objects", "query_image_id": ids[0],
        "descriptors": [{"name": "color_hsv", "weight": 1}], "k": 2,
    })
    assert response.status_code == 400
    assert "No built index" in response.json()["detail"]


def test_unknown_query_image_id_returns_404(demo):
    client, _, _, _ = demo
    response = client.post("/api/search", json={"dataset": "objects", "query_image_id": 9999, "k": 2})
    assert response.status_code == 404
    assert "Unknown query_image_id" in response.json()["detail"]


def test_descriptor_picker_endpoint_only_lists_built_descriptors(demo):
    client, _, _, _ = demo
    response = client.get("/api/descriptors?dataset=objects")
    assert response.status_code == 200
    descriptors = response.json()["descriptors"]
    assert [item["name"] for item in descriptors] == ["color_hsv"]
    assert descriptors[0]["metric"] == "l2"  # metric recorded by this fixture's index
