"""Search endpoint."""
"""JSON and multipart search endpoints."""
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from backend.app.config import get_settings, public_image_url
from backend.app.core.descriptors import AVAILABLE_DESCRIPTOR_NAMES, DESCRIPTORS
from backend.app.core.search.engine import SearchEngine
from backend.app.db.models import Image
from backend.app.db.session import SessionLocal
from backend.app.schemas import DescriptorWeight, SearchRequest, SearchResponse, SearchResult
router = APIRouter()
engine = SearchEngine(get_settings().data_dir)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    start = perf_counter()
    try:
        if payload.query_image_id is not None and db.query(Image.id).filter_by(
                id=payload.query_image_id, dataset=payload.dataset).first() is None:
            raise LookupError(f"Unknown query_image_id {payload.query_image_id} for dataset '{payload.dataset}'")
        for selection in payload.descriptors:
            if selection.name not in AVAILABLE_DESCRIPTOR_NAMES:
                raise ValueError(f"Unknown descriptor '{selection.name}'")
            if engine.registry.get(payload.dataset, selection.name) is None:
                raise FileNotFoundError(f"No built index and ids.npy for dataset '{payload.dataset}', descriptor '{selection.name}'")
        found = engine.search(db, payload.dataset, payload.keywords, payload.keyword_mode,
                              payload.query_image_id, [(item.name, item.weight) for item in payload.descriptors], payload.k)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    query_image_url = None
    if payload.query_image_id is not None:
        query_image = db.query(Image).filter_by(id=payload.query_image_id, dataset=payload.dataset).one_or_none()
        if query_image is not None:
            query_image_url = public_image_url(query_image.url)
    return _response(payload.model_dump(), found, query_image_url, start)


def _response(query: dict, found, query_image_url: str | None, started: float) -> SearchResponse:
    results = [SearchResult(image_id=hit.image.id, url=public_image_url(hit.image.url),
                            distance=float(hit.distance), rank=rank,
                            descriptor_distances=hit.descriptor_distances,
                            keywords=[keyword.value for keyword in hit.image.keywords])
               for rank, hit in enumerate(found, start=1)]
    return SearchResponse(query=query, query_image_url=query_image_url, results=results,
                          took_ms=(perf_counter() - started) * 1000)


@router.post("/search/upload", response_model=SearchResponse)
async def search_upload(
    image: UploadFile = File(...),
    dataset: str = Form(...),
    keywords: str = Form("[]"),
    keyword_mode: str = Form("and"),
    descriptors: str = Form("[]"),
    k: int = Form(10),
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Search using an uploaded query image and the same fields as JSON search."""
    started = perf_counter()
    if dataset == "mpeg7_1m":
        raise HTTPException(status_code=422,
                            detail="Image uploads are unsupported for mpeg7_1m unless an MPEG-7 extractor is configured.")
    try:
        keyword_values = json.loads(keywords)
        descriptor_values = [DescriptorWeight.model_validate(item) for item in json.loads(descriptors)]
        payload = SearchRequest(dataset=dataset, keywords=keyword_values, keyword_mode=keyword_mode,
                                query_image_id=None, descriptors=descriptor_values, k=k)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid multipart search fields: {exc}") from exc
    for item in payload.descriptors:
        if item.name not in DESCRIPTORS:
            raise HTTPException(status_code=400, detail=f"Unknown descriptor '{item.name}'")
        if engine.registry.get(payload.dataset, item.name) is None:
            raise HTTPException(status_code=400,
                                detail=f"No built index and ids.npy for dataset '{payload.dataset}', descriptor '{item.name}'")

    raw = await image.read()
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image.")
    query_vectors = {item.name: DESCRIPTORS[item.name].extract(decoded) for item in payload.descriptors}

    # Keep the upload addressable through the same static image route as result images.
    image_dir = get_settings().data_dir / "images" / "queries"
    image_dir.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", decoded)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not encode uploaded query image.")
    relative_path = Path("queries") / f"{uuid4().hex}.jpg"
    (image_dir / relative_path.name).write_bytes(encoded.tobytes())
    query_image_url = public_image_url((Path("queries") / relative_path.name).as_posix())

    try:
        found = engine.search(db, payload.dataset, payload.keywords, payload.keyword_mode,
                              None, [(item.name, item.weight) for item in payload.descriptors],
                              payload.k, query_vectors=query_vectors)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _response(payload.model_dump(), found, query_image_url, started)
