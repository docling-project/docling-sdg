import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import tqdm
from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.utils import format_string
from pydantic import ConfigDict, validate_call

from docling_core.types.nlp.qa_labels import QAInformationLabel, QALabelling

from docling_sdg.qa.base import (
    GenerateOptions,
    GenerateResult,
    GenQAC,
    QaChunk,
    Status,
)
from docling_sdg.qa.prompts.generation_prompts import PromptTypes
from docling_sdg.qa.utils import (
    ChatAgent,
    initialize_llm,
    postprocess_answer,
    postprocess_question,
    retrieve_stored_passages,
    retrieve_stored_qac_ids,
    save_to_file,
)

_log = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        generate_options: GenerateOptions,
    ):
        self.options = generate_options

        self.qac_types = list(
            {label for prt in self.options.prompts for label in prt.labels or []}
        )

        self.llm: LLM = initialize_llm(generate_options)
        self.agent = ChatAgent(llm=self.llm)

    def generate_from_prompt(
        self,
        key_dict: dict[str, str],
        question_types: list[QAInformationLabel],
        prompt_type: PromptTypes,
    ) -> tuple[str, str] | tuple[None, None]:
        if not set(question_types).issubset(set(self.qac_types)):
            _log.warning(
                f"{question_types} not found in list of available "
                f"types: {self.qac_types}"
            )
            return (None, None)
        if prompt_type.value not in [
            prt.type_
            for prt in self.options.prompts
            if set(prt.labels) == set(question_types)
        ]:
            _log.warning(
                f"{prompt_type} prompt not available for types {question_types}"
            )
            return (None, None)

        template = [
            prt.template
            for prt in self.options.prompts
            if (set(prt.labels) == set(question_types) and prt.type_ == prompt_type)
        ][0]
        prompt_template = PromptTemplate(template=template)
        prompt = format_string(prompt_template.template, **key_dict).strip()

        return (
            self.agent.ask(question=prompt, max_tokens=self.options.max_new_tokens)
            .replace("\n", " ")
            .strip(),
            prompt.strip(),
        )

    @validate_call(config=ConfigDict(strict=True))
    def generate_from_sample(self, source: Path) -> GenerateResult:
        _log.debug(f"Output file: {self.options.generated_file.absolute()}")
        start_time = time.time()

        passages: Iterator[QaChunk] = retrieve_stored_passages(in_file=source)

        result = self.generate_from_chunks(passages)
        end_time = time.time()
        result.time_taken = end_time - start_time

        return result

    def generate_from_chunks(  # noqa: C901
        self, stored_chunks: Iterator[QaChunk]
    ) -> GenerateResult:
        start_time = time.time()

        stored_chunk_ids: list[str] = []
        num_stored_qac: int = 0
        for _, chunk_id in retrieve_stored_qac_ids(self.options.generated_file):
            num_stored_qac += 1
            if chunk_id not in stored_chunk_ids:
                stored_chunk_ids.append(chunk_id)

        num_exported_qac: int = 0
        if num_stored_qac >= self.options.max_qac:
            end_time = time.time()
        else:
            num_extra_qac: int = self.options.max_qac - num_stored_qac
            for chunk in tqdm.tqdm(stored_chunks):
                if num_exported_qac >= num_extra_qac:
                    break

                # If a passage is in QAC file, skip it
                if chunk.meta.chunk_id in stored_chunk_ids:
                    continue

                # Generate question
                question, question_prompt = self.generate_from_prompt(
                    key_dict={"context_str": chunk.text},
                    question_types=self.qac_types,
                    prompt_type=PromptTypes.QUESTION,
                )
                if question is None or question_prompt is None:
                    continue

                generated_qas: dict[str, str] = {}
                if len(self.qac_types) > 1:
                    str_dict = question[
                        question.find("{") : question.rfind("}") + 1
                    ].replace(r"\_", "_")
                    try:
                        generated_qas = json.loads(str_dict)
                    except json.decoder.JSONDecodeError:
                        _log.warning(
                            f"Failed parsing JSON from generated question: {str_dict}"
                        )
                        continue
                else:
                    generated_qas = {self.qac_types[0]: question}

                for this_type, raw_question in generated_qas.items():
                    if this_type not in self.qac_types:
                        _log.debug(
                            f"Unexpected generated question type: {this_type}."
                            f"Requested types were: {self.qac_types}"
                        )
                        continue

                    if not isinstance(raw_question, str):
                        _log.debug(f"Generated question is not string: {raw_question}")
                        continue

                    this_question = postprocess_question(question=raw_question)

                    if this_question is None:
                        continue

                    raw_answer: str | None = None
                    answer_prompt: str | None = None
                    if len(self.qac_types) > 1:  # combined, a JSON is expected
                        answer_type = this_type + "_answer"
                        answer_prompt = ""
                        if answer_type in generated_qas:
                            raw_answer = generated_qas[answer_type]

                    if raw_answer is None:
                        raw_answer, answer_prompt = self.generate_from_prompt(
                            key_dict={
                                "context_str": chunk.text,
                                "question_str": question,
                            },
                            question_types=[this_type],  # type: ignore[list-item]
                            prompt_type=PromptTypes.ANSWER,
                        )

                    if raw_answer is None or answer_prompt is None:
                        continue

                    this_answer = postprocess_answer(answer=raw_answer)

                    if this_answer is None:
                        continue

                    num_exported_qac += 1
                    qa_labels = QALabelling(information=this_type)

                    qac_txt = this_question + this_answer + chunk.text
                    qac_id: str = hashlib.sha256(qac_txt.encode()).hexdigest()
                    save_to_file(
                        objects=[
                            GenQAC(
                                doc_id=chunk.meta.doc_id,
                                qac_id=qac_id,
                                context=chunk.text,
                                question=this_question,
                                answer=this_answer,
                                generated_question=True,
                                generated_answer=True,
                                created=datetime.now(),
                                model=self.options.model_id,
                                paths=[""],
                                chunk_id=chunk.meta.chunk_id,
                                labels=qa_labels,
                            )
                        ],
                        out_file=self.options.generated_file,
                    )
            end_time = time.time()

        generate_res = GenerateResult(
            status=Status.SUCCESS,
            time_taken=(end_time - start_time),
            num_qac=num_exported_qac,
            output=self.options.generated_file,
        )

        return generate_res
