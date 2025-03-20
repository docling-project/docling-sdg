"""Define the models for question-answering (Q&A)."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from llama_index.llms.ibm.base import GenTextParamsMetaNames
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    NonNegativeInt,
    SecretStr,
)

from docling_core.transforms.chunker import DocChunk, DocMeta
from docling_core.types.doc import DocItemLabel
from docling_core.types.nlp.qa import QAPair

from docling_sdg.qa.prompts.generation_prompts import (
    QaPromptTemplate,
    default_combined_question_qa_prompt,
)


class Status(str, Enum):
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"


class SampleOptions(BaseModel):
    """Passage sampling options for Q&A generation."""

    sample_file: Path = Field(
        default=Path("docling_sdg_sample.jsonl"),
        description="Path to the target file to store the sample passages.",
    )
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
        default=50, gt=0, description="Maximum number of passages to sample."
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


class LlmOptions(BaseModel):
    """Generative AI options for Q&A generation.

    Currently, only support watsonx.ai.
    """

    url: AnyUrl = Field(
        default=AnyUrl("https://us-south.ml.cloud.ibm.com"),
        description="Url to Watson Machine Learning or CPD instance.",
    )
    project_id: Optional[SecretStr] = Field(
        default=None, description="ID of the Watson Studio project."
    )
    api_key: Optional[SecretStr] = Field(
        default=None, description="API key to Watson Machine Learning or CPD instance."
    )
    model_id: str = Field(
        default="mistralai/mixtral-8x7b-instruct-v01",
        description="Type of model to use.",
    )
    max_new_tokens: int = Field(
        default=512, ge=0, description="The maximum number of tokens to generate."
    )
    additional_params: Optional[dict[str, Any]] = Field(
        default={
            GenTextParamsMetaNames.DECODING_METHOD: "sample",
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 50,
            GenTextParamsMetaNames.TEMPERATURE: 0.0,
            GenTextParamsMetaNames.TOP_K: 50,
            GenTextParamsMetaNames.TOP_P: 0.95,
        },
        description="Additional generation params for the watsonx.ai models.",
    )


class GenerateOptions(LlmOptions):
    generated_file: Path = Field(
        default=Path("docling_sdg_generated_qac.jsonl"),
        description="Path to the target file to store the generated Q&A.",
    )
    max_qac: int = Field(
        default=100, gt=0, description="Maximum number of Q&A items to generate."
    )
    prompts: list[QaPromptTemplate] = Field(
        default=[default_combined_question_qa_prompt],
        description="List of Q&A prompt templates.",
    )


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


class GenerateResult(BaseResult):
    output: Annotated[
        Path, Field(description="Path to the file containing the generated Q&A items.")
    ]
    num_qac: Annotated[
        NonNegativeInt, Field(description="Number of Q&A items added to the file.")
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


class GenQAC(QAPair[BaseModel]):
    """Generated question-answering-context object."""

    doc_id: str
    chunk_id: str
    qac_id: str
    # prompts: dict[str, str] = {}
    critiques: dict[str, str | int] = {}
