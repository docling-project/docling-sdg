"""Utility functions for question-answering (Q&A)."""

import hashlib
from typing import Callable, Iterator, Optional

from docling_core.transforms.chunker import DocChunk

from docling_sdg.qa.base import QaChunk, QaMeta


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
