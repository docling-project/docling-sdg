"""Define the models for question-answering (Q&A)."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from docling_core.transforms.chunker import DocChunk, DocMeta
from docling_core.types.doc import DocItemLabel
from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
)


class Status(str, Enum):
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"


class SampleOptions(BaseModel):
    """Passage sampling options for Q&A generation."""

    sample_file: Annotated[
        Path, Field(description="Path to the target file to store the sample passages.")
    ]
    chunker: Literal["hybrid", "hierarchical"] = Field(
        default="hybrid",
        description="Docling chunker to create passages.",
        examples=["HierarchicalChunker", "HybridChunker"],
    )
    min_token_count: int = Field(
        default=10,
        ge=0,
        le=512,
        description="Only consider passages with at least this number of tokens.",
    )
    max_passages: int = Field(
        default=50, ge=0, description="Maximum number of passages to sample."
    )
    doc_items: Optional[list[DocItemLabel]] = Field(
        default=[DocItemLabel.TEXT, DocItemLabel.PARAGRAPH],
        min_length=1,
        description=(
            "Only consider passages that include these doc items. If None, no "
            "constraint will be applied."
        ),
    )
    seed: int = Field(default=0, description="Random seed for sampling.")


class BaseResult(BaseModel):
    status: Annotated[Status, Field(description="Status of the running process.")]
    timing: Annotated[float, Field(description="Processing time in seconds.")]


class SampleResult(BaseResult):
    output: Annotated[
        Path, Field(description="Path to the file containing the sample passages.")
    ]
    num_passages: Annotated[
        NonNegativeInt, Field(description="Number of passages added to the file.")
    ]


class QaMeta(DocMeta):
    """Data model for question-answering chunk metadata."""

    chunk_id: Annotated[
        str,
        Field(description="Unique identifier of this chunk within a Q&A collection."),
    ]
    doc_id: Annotated[
        str,
        Field(description="Unique identifier of the document containing this chunk."),
    ]


class QaChunk(DocChunk):
    """Chunk in a question-answering collection."""

    meta: QaMeta

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, QaChunk):
            return self.meta.chunk_id == other.meta.chunk_id
        else:
            return NotImplemented
