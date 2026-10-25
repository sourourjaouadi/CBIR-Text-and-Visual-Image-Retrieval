"""Relational image and keyword metadata."""
from sqlalchemy import ForeignKey, String, Table, Column, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.session import Base

image_keywords = Table("image_keywords", Base.metadata,
    Column("image_id", ForeignKey("images.id", ondelete="CASCADE"), primary_key=True),
    Column("keyword_id", ForeignKey("keywords.id", ondelete="CASCADE"), primary_key=True),
    Index("ix_image_keywords_keyword_image", "keyword_id", "image_id"))

class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    dataset: Mapped[str] = mapped_column(String(32), index=True)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    keywords: Mapped[list["Keyword"]] = relationship(secondary=image_keywords, back_populates="images")

class Keyword(Base):
    __tablename__ = "keywords"
    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column("word", String(100), unique=True, index=True)
    images: Mapped[list[Image]] = relationship(secondary=image_keywords, back_populates="keywords")
