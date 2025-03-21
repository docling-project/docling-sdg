import logging
import random
import time
from pathlib import Path
from typing import Iterator, Optional, Union, cast

from pydantic import ConfigDict, validate_call
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import DocChunk, HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_sdg.qa.base import QaChunk, SampleOptions, SampleResult, Status
from docling_sdg.qa.utils import get_qa_chunks, retrieve_stored_passages, save_to_file

_log = logging.getLogger(__name__)


class PassageSampler:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __init__(self, sample_options: Optional[SampleOptions] = None):
        self.options = sample_options or SampleOptions()
        self.chunker = (
            HierarchicalChunker(heading_as_metadata=True)
            if self.options.chunker == "hierarchical"
            else HybridChunker()
        )

    def _filter_by_count(self, chunk: DocChunk) -> bool:
        if self.options.min_token_count == 0:
            return True

        tokenized = self.tokenizer(
            text=chunk.text, truncation=True, max_length=self.tokenizer.model_max_length
        )

        return len(tokenized.tokens()) >= self.options.min_token_count

    def _filter_by_docitem(self, chunk: DocChunk) -> bool:
        if self.options.doc_items is None:
            return True

        chunk_labels = [item.label for item in chunk.meta.doc_items]

        return any(item in chunk_labels for item in self.options.doc_items)

    @validate_call(config=ConfigDict(strict=True))
    def sample(self, source: list[Union[Path, str, DocumentStream]]) -> SampleResult:
        _log.debug(f"Output file: {self.options.sample_file.absolute()}")
        start_time = time.time()

        passages: list[QaChunk] = list(
            retrieve_stored_passages(in_file=self.options.sample_file)
        )

        num_exported_passages: int = 0
        if len(passages) >= self.options.max_passages:
            end_time = time.time()
        else:
            qa_chunks: list[QaChunk] = []
            _log.info("Loading and chunking documents from source...")
            converter = DocumentConverter()
            results = converter.convert_all(source)
            for res in results:
                if res.status != ConversionStatus.SUCCESS:
                    _log.warning(f"Could not parse document: {res.errors}")
                    continue
                doc: DoclingDocument = res.document
                doc_id = str(doc.origin.binary_hash) if doc.origin else doc.name
                _log.debug(f"Parsed document {doc_id}.")
                chunk_iter = self.chunker.chunk(dl_doc=doc)
                qa_chunks.extend(
                    item
                    for item in get_qa_chunks(
                        doc_id,
                        cast(Iterator[DocChunk], chunk_iter),
                        [self._filter_by_count, self._filter_by_docitem],
                    )
                    if item not in qa_chunks
                )
                _log.debug(
                    f"Got {len(qa_chunks)} chunks from {doc_id} after filtering."
                )

            num_extra_passages = self.options.max_passages - len(passages)
            if len(qa_chunks) > num_extra_passages:
                random.seed(self.options.seed)
                sample_passages = random.sample(qa_chunks, num_extra_passages)
                save_to_file(sample_passages, self.options.sample_file)
                num_exported_passages = len(sample_passages)
            else:
                save_to_file(qa_chunks, self.options.sample_file)
                num_exported_passages = len(qa_chunks)

            end_time = time.time()

        sample_res = SampleResult(
            status=Status.SUCCESS,
            time_taken=(end_time - start_time),
            num_passages=num_exported_passages,
            output=self.options.sample_file,
        )

        return sample_res
