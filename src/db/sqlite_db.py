import os
from typing import List, Optional, Set, Tuple

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
)

from src.schemas.fact_schema import ExtractedFact


# Load environment variables
load_dotenv()


# =============================================================
# Database configuration
# =============================================================

DATABASE_PATH = os.getenv(
    "SQLITE_DB_PATH",
    "data/facts.sqlite",
)

# Make sure the database directory exists
os.makedirs(
    os.path.dirname(DATABASE_PATH) or ".",
    exist_ok=True,
)

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


Base = declarative_base()


# =============================================================
# Database models
# =============================================================


class Document(Base):
    __tablename__ = "documents"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    file_name = Column(
        String,
        nullable=False,
        unique=True,
    )

    document_title = Column(
        String,
        nullable=True,
    )

    reporting_period = Column(
        String,
        nullable=True,
    )

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False,
    )

    page_number = Column(
        Integer,
        nullable=True,
    )

    chunk_index = Column(
        Integer,
        nullable=True,
    )

    content = Column(
        Text,
        nullable=False,
    )

    document = relationship(
        "Document",
        back_populates="chunks",
    )


class Fact(Base):
    __tablename__ = "facts"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False,
    )

    chunk_id = Column(
        Integer,
        ForeignKey("chunks.id"),
        nullable=True,
    )

    entity = Column(
        String,
        nullable=False,
    )

    attribute = Column(
        String,
        nullable=False,
    )

    raw_value = Column(
        String,
        nullable=False,
    )

    unit = Column(
        String,
        nullable=True,
    )

    time = Column(
        String,
        nullable=True,
    )

    scope = Column(
        String,
        nullable=True,
    )

    qualifier = Column(
        String,
        nullable=True,
    )

    exact_quote = Column(
        Text,
        nullable=False,
    )

    confidence_score = Column(
        Integer,
        nullable=False,
    )

    normalized_value = Column(
        String,
        nullable=True,
    )

    status = Column(
        String,
        nullable=False,
        default="VALID",
    )


# =============================================================
# Database initialization
# =============================================================


def init_db():
    """
    Create all database tables if they do not already exist.
    """

    Base.metadata.create_all(
        bind=engine,
    )


# =============================================================
# Document operations
# =============================================================


def create_document(
    file_name: str,
    document_title: Optional[str] = None,
    reporting_period: Optional[str] = None,
) -> int:
    """
    Create a document record and return its ID.
    """

    session = SessionLocal()

    try:
        document = Document(
            file_name=file_name,
            document_title=document_title,
            reporting_period=reporting_period,
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        return document.id

    finally:
        session.close()


def get_document_by_filename(
    file_name: str,
) -> Optional[Document]:
    """
    Retrieve an existing document by filename.

    Used to make PDF processing incremental/idempotent.
    """

    session = SessionLocal()

    try:
        return (
            session.query(Document)
            .filter(
                Document.file_name == file_name
            )
            .first()
        )

    finally:
        session.close()


# =============================================================
# Chunk operations
# =============================================================


def create_chunk(
    document_id: int,
    page_number: Optional[int],
    chunk_index: Optional[int],
    content: str,
) -> int:
    """
    Store a document chunk and return its ID.
    """

    session = SessionLocal()

    try:
        chunk = Chunk(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            content=content,
        )

        session.add(chunk)
        session.commit()
        session.refresh(chunk)

        return chunk.id

    finally:
        session.close()


def get_processed_chunk_keys(
    document_id: int,
) -> Set[Tuple[Optional[int], Optional[int]]]:
    """
    Return the (page_number, chunk_index) pairs that have
    already been processed for a document.

    This allows the pipeline to skip chunks that have already
    been stored in SQLite.
    """

    session = SessionLocal()

    try:
        chunks = (
            session.query(Chunk)
            .filter(
                Chunk.document_id == document_id
            )
            .all()
        )

        return {
            (
                chunk.page_number,
                chunk.chunk_index,
            )
            for chunk in chunks
        }

    finally:
        session.close()


# =============================================================
# Fact operations
# =============================================================


def create_fact(
    document_id: int,
    chunk_id: Optional[int],
    fact: ExtractedFact,
) -> int:
    """
    Store an extracted fact and return its database ID.
    """

    session = SessionLocal()

    try:
        db_fact = Fact(
            document_id=document_id,
            chunk_id=chunk_id,
            entity=fact.entity,
            attribute=fact.attribute,
            raw_value=fact.raw_value,
            unit=fact.unit,
            time=fact.time,
            scope=fact.scope,
            qualifier=fact.qualifier,
            exact_quote=fact.exact_quote,
            confidence_score=fact.confidence_score,
            normalized_value=(
                str(fact.normalized_value)
                if fact.normalized_value is not None
                else None
            ),
            status=fact.status,
        )

        session.add(db_fact)
        session.commit()
        session.refresh(db_fact)

        return db_fact.id

    finally:
        session.close()


# =============================================================
# Fact retrieval
# =============================================================


def get_valid_facts() -> List[Fact]:
    """
    Retrieve all facts that passed grounding validation.
    """

    session = SessionLocal()

    try:
        return (
            session.query(Fact)
            .filter(
                Fact.status == "VALID"
            )
            .all()
        )

    finally:
        session.close()