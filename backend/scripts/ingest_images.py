"""Batch-ingest image paths and filename-derived labels/keywords into the database."""
import argparse
from itertools import islice
import logging
from pathlib import Path

from sqlalchemy.orm import selectinload

from backend.app.config import get_settings
from backend.app.db.models import Image, Keyword
from backend.app.db.session import Base, SessionLocal, engine

logger = logging.getLogger(__name__)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _batches(items, size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    Base.metadata.create_all(bind=engine)
    image_root = (get_settings().data_dir / "images").resolve()
    paths = (path for path in args.directory.rglob("*")
             if path.is_file() and path.suffix.lower() in EXTENSIONS)

    inserted = updated = 0
    with SessionLocal() as db:
        for batch in _batches(paths, args.batch_size):
            records: list[tuple[str, str, list[str]]] = []
            for path in batch:
                resolved = path.resolve()
                try:
                    relative_path = resolved.relative_to(image_root).as_posix()
                except ValueError as exc:
                    raise ValueError(f"Image {resolved} must be inside DATA_DIR/images ({image_root})") from exc
                tokens = list(dict.fromkeys(part for part in path.stem.lower().split("_") if part))
                records.append((relative_path, tokens[0] if tokens else path.stem.lower(), tokens))

            urls = [record[0] for record in records]
            existing_rows = (db.query(Image).options(selectinload(Image.keywords))
                             .filter(Image.dataset == args.dataset, Image.url.in_(urls)).all())
            existing = {row.url: row for row in existing_rows}
            words = {word for _, _, tokens in records for word in tokens}
            keyword_rows = db.query(Keyword).filter(Keyword.value.in_(words)).all() if words else []
            keywords = {row.value: row for row in keyword_rows}
            missing_words = sorted(words - keywords.keys())
            new_keywords = [Keyword(value=value) for value in missing_words]
            db.add_all(new_keywords)
            if new_keywords:
                db.flush()
                keywords.update({row.value: row for row in new_keywords})

            for url, label, tokens in records:
                row = existing.get(url)
                if row is None:
                    row = Image(url=url, dataset=args.dataset, label=label)
                    db.add(row)
                    inserted += 1
                else:
                    row.label = label
                    updated += 1
                current_words = {keyword.value for keyword in row.keywords}
                row.keywords.extend(keywords[word] for word in tokens
                                    if word in keywords and word not in current_words)
            db.commit()
    logger.info("Ingestion finished for %s: inserted=%d updated=%d", args.dataset, inserted, updated)
    print(f"{args.dataset}: inserted={inserted}, updated={updated}")


if __name__ == "__main__":
    main()
