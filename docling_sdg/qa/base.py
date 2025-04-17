"""Define the models for question-answering (Q&A)."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

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

from docling_sdg.qa.prompts.critique_prompts import (
    CritiquePromptTemplate,
    default_critique_templates,
)
from docling_sdg.qa.prompts.generation_prompts import (
    QaPromptTemplate,
    default_combined_question_qa_prompt,
)


class Status(str, Enum):
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"


class Chunker(str, Enum):
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class SampleOptions(BaseModel):
    """Passage sampling options for Q&A generation."""

    sample_file: Path = Field(
        default=Path("docling_sdg_sample.jsonl"),
        description="Path to the target file to store the sample passages.",
    )
    chunker: Chunker = Field(
        default=Chunker.HYBRID,
        description="Docling chunker to create passages.",
        examples=["hybrid", "hierarchical"],
    )
    min_token_count: int = Field(
        default=20,
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


class LlmProvider(str, Enum):
    OPENAI = "openai"
    OPENAI_LIKE = "openai_like"
    WATSONX = "watsonx"


class LlmOptions(BaseModel):
    """Generative AI options for Q&A generation."""

    provider: LlmProvider = Field(
        default=LlmProvider.OPENAI_LIKE,
        description=f"LLM provider: [{','.join(LlmProvider)}]",
    )
    url: AnyUrl = Field(
        default=AnyUrl("http://127.0.0.1:11434/v1"),
        description="Url to LLM API endpoint",
    )
    project_id: Optional[SecretStr] = Field(
        default=None, description="ID of the Watson Studio project."
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key to Watson Machine Learning or CPD instance.",
    )
    model_id: str = Field(
        default="mistralai/mixtral-8x7b-instruct-v01",
        description="Which model to use.",
    )
    max_new_tokens: int = Field(
        default=512, ge=0, description="The maximum number of tokens to generate."
    )
    additional_params: dict[str, Any] = Field(
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


class CritiqueOptions(LlmOptions):
    critiqued_file: Path = Field(
        default=Path("docling_sdg_critiqued_qac.jsonl"),
        description="Path to the target file to store the critiqued Q&A.",
    )
    max_qac: int = Field(
        default=100, gt=0, description="Maximum number of Q&A items to critique."
    )
    prompts: list[CritiquePromptTemplate] = Field(
        default=default_critique_templates,
        description="List of critique prompt templates.",
    )


class BaseResult(BaseModel):
    status: Annotated[Status, Field(description="Status of the running process.")]
    time_taken: Annotated[float, Field(description="Processing time in seconds.")]


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


class CritiqueResult(BaseResult):
    output: Annotated[
        Path,
        Field(
            description=(
                "Path to the file containing the generated Q&A item with critiques."
            )
        ),
    ]
    num_qac: Annotated[
        NonNegativeInt,
        Field(description=("Number of critiqued Q&A items added to the file.")),
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


class Critique(BaseModel):
    evaluation: Optional[str] = Field(
        None, description=("An explanation of the critique.")
    )
    rating: Annotated[
        Optional[int],
        Field(
            description=(
                "A number indicating the evaluation result. A higher rating means "
                "better quality."
            )
        ),
    ]


class GenQAC(QAPair[BaseModel]):
    """Generated question-answering-context object."""

    doc_id: Annotated[str, Field(description="The unique identifier of a document.")]
    chunk_id: Annotated[
        str,
        Field(description="The unique identifier of a passage within the document."),
    ]
    qac_id: Annotated[
        str,
        Field(description="The unique identifier of a question-answer-contex item."),
    ]
    critiques: dict[str, Critique] = Field(
        default={}, description="A set of critiques for each supported dimension."
    )
