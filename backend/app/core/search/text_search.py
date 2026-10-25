"""Database-backed keyword candidate ID filtering."""
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.models import Image, Keyword


def candidate_image_ids(db: Session, dataset: str, words: list[str], mode: str) -> list[int] | None:
    """Return sorted candidate IDs, or None when the whole dataset is selected."""
    lowered = [word.strip().lower() for word in words if word.strip()]
    if not lowered:
        return None
    query = db.query(Image.id).filter(Image.dataset == dataset)
    if mode == "or":
        query = query.join(Image.keywords).filter(func.lower(Keyword.value).in_(lowered)).distinct()
    else:
        for word in lowered:
            query = query.filter(Image.keywords.any(func.lower(Keyword.value) == word))
    return [int(row[0]) for row in query.order_by(Image.id).all()]


def dataset_image_count(db: Session, dataset: str) -> int:
    return int(db.query(func.count(Image.id)).filter(Image.dataset == dataset).scalar() or 0)
