"""FastAPI application entrypoint."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.api import routes_descriptors, routes_images, routes_search
from backend.app.config import get_settings
from backend.app.db.session import Base, engine
from backend.app.db import models  # noqa: F401

settings = get_settings()
logging.basicConfig(level=settings.log_level)
Base.metadata.create_all(bind=engine)
app = FastAPI(title="CBIR Search API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
static_root = settings.data_dir / "images"
static_root.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=static_root), name="images")
app.include_router(routes_search.router, prefix="/api")
app.include_router(routes_images.router, prefix="/api")
app.include_router(routes_descriptors.router, prefix="/api")

@app.get("/health")
def health(): return {"status": "ok"}
