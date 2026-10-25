"""Image listing endpoint."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from backend.app.api.routes_search import get_db
from backend.app.db.models import Image, Keyword
from backend.app.config import public_image_url
router = APIRouter()

@router.get("/images")
def list_images(dataset: str = "objects", keyword: str | None = None,
                page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    query = db.query(Image).filter(Image.dataset == dataset)
    if keyword: query = query.join(Image.keywords).filter(func.lower(Keyword.value) == keyword.lower())
    rows = query.order_by(Image.id).offset((page - 1) * size).limit(size).all()
    return {"items": [{"image_id": row.id, "url": public_image_url(row.url), "label": row.label,
                       "keywords": [word.value for word in row.keywords]} for row in rows], "page": page, "size": size}


@router.get("/images/{image_id}")
def get_image(image_id: int, dataset: str = "objects", db: Session = Depends(get_db)):
    row = db.query(Image).filter_by(id=image_id, dataset=dataset).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id {image_id} for dataset '{dataset}'")
    return {"image_id": row.id, "url": public_image_url(row.url), "label": row.label,
            "keywords": [keyword.value for keyword in row.keywords]}
