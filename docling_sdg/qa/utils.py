"""Utility functions for question-answering (Q&A)."""

import hashlib
import os
import re
from pathlib import Path
from typing import Callable, Final, Iterator, Optional, TypeVar, cast

import jsonlines
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel

from docling_core.transforms.chunker import DocChunk

from docling_sdg.qa.base import GenQAC, QaChunk, QaMeta

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


_FORBIDDEN_PREFIXES: Final = ["this", "the", "given", "following"]
_FORBIDDEN_TEXTUALS: Final = ["paragraph", "context", "text", "passage"]
_FORBIDDEN_TERMS: Final = [
    f"{fp} {ft}" for fp in _FORBIDDEN_PREFIXES for ft in _FORBIDDEN_TEXTUALS
]


def get_qa_chunks(
    dl_id: str,
    chunks: Iterator[DocChunk],
    filter: Optional[list[Callable[[DocChunk], bool]]],
) -> Iterator[QaChunk]:
    ids: set[str] = set()
    for item in chunks:
        if not item.text:
            continue

        if filter is not None and all(func(item) for func in filter):
            chunk_id: str = hashlib.sha256(item.text.encode()).hexdigest()
            if chunk_id not in ids:
                qa_meta = QaMeta(
                    doc_items=item.meta.doc_items,
                    headings=item.meta.headings,
                    captions=item.meta.captions,
                    origin=item.meta.origin,
                    chunk_id=chunk_id,
                    doc_id=dl_id,
                )
                qa_chunk = QaChunk(text=item.text, meta=qa_meta)
                ids.add(chunk_id)

                yield qa_chunk


def retrieve_stored_passages(in_file: Path) -> Iterator[QaChunk]:
    if os.path.isfile(in_file):
        with open(in_file, encoding="utf-8") as file_obj:
            for line in file_obj:
                line += line.strip()
                if line:
                    yield QaChunk.model_validate_json(line)


def retrieve_stored_qac_ids(in_file: Path) -> Iterator[tuple[str, str]]:
    if os.path.isfile(in_file):
        with open(in_file, encoding="utf-8") as qac_file:
            for line in qac_file:
                line += line.strip()
                if line:
                    qac = GenQAC.model_validate_json(line)

                    yield qac.qac_id, qac.chunk_id


def postprocess_question(question: str) -> Optional[str]:
    question = question.strip()

    questions_rgx = [
        r"^Question \d+:",
        r"^Question\:",
        r"^\d+\.\s?Question\:",
        r"^\d+\.Q\:",
        r"^\\d+\\.",
        r"^\d+",
        r"^Q\:",
        r"^Q\d+\:",
        r"\*\*Question (\d+):\*\*",
    ]

    for rgx in questions_rgx:
        question = re.sub(rgx, "", question).strip()

    if not question.endswith("?"):
        return None

    question_lower = question.lower()
    for terms in _FORBIDDEN_TERMS:
        if terms in question_lower:
            return None

    return question


def postprocess_answer(answer: str) -> str:
    answer = str(answer).strip()

    answers_rgx = [
        r"^(\-|\*)",
        r"^[A-Z](\:|\.)",
        r"^Answer \d+\:",
        r"^Answer\s?:",
        r"^A(\d+)?\s?\:",
        r"^assistant\:",
    ]

    for rgx in answers_rgx:
        answer = re.sub(rgx, "", answer).strip()

    return answer


def save_to_file(objects: list[BaseModelT], out_file: Path) -> None:
    mode = "a" if os.path.isfile(out_file) else "w"

    with jsonlines.open(out_file, mode=mode) as writer:
        for item in objects:
            cast(jsonlines.Writer, writer).write(item.model_dump())


class ChatAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.model_id = llm.metadata.model_name

    def ask(self, question: str) -> str:
        response = self.llm.chat([ChatMessage(content=question)])
        answer = str(response)
        return answer
