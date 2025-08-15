"""Conceptual question generation.

Conceptual question generation produces questions using an abstract description
of the content.  The goal is to simulate what an actual user of an AI application
would know about the content when asking a question in order to optimize how realistic
the questions are for that application.  Note that this also includes generating some
questions that can't be answered by the content, which is useful for testing how a
system responds to such questions.
"""

# This file has been modified with the assistance of AI Tools:
#  Cursor using claude-4-sonnet and Google Gemini

import hashlib
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import jsonlines
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.llms.types import TextBlock
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import ConfigDict, validate_call
from tqdm import tqdm

from docling_core.types.nlp.qa_labels import QALabelling

from docling_sdg.qa.base import (
    ConceptualGenerateOptions,
    GenerateResult,
    GenQAC,
    Status,
    UserProfile,
)
from docling_sdg.qa.prompts.generation_prompts import (
    QaPromptTemplate,
)
from docling_sdg.qa.utils import (
    ChatAgent,
    initialize_llm,
    retrieve_stored_qac,
    save_to_file,
)

_log = logging.getLogger(__name__)


# Configuration for the reference answer generation.
# These are hard coded for now, but we might make them configurable in the future.
EMBED_MODEL_ID = "ibm-granite/granite-embedding-125m-english"
NUMBER_OF_SEARCH_RESULTS_TO_CONSIDER = 25
NUMBER_OF_SEARCH_RESULTS_TO_SELECT_FOR_ANSWER_GENERATION = 7
REFERENCE_ANSWER_GENERATION_RERANKER_BATCH_SIZE = 10


class ConceptualGenerator:
    def __init__(self, generate_options: ConceptualGenerateOptions):
        self.options = generate_options

        self.qac_types = list(
            {
                label
                for prt in self.options.question_prompts
                for label in prt.labels or []
            }
        )
        self.llm: LLM = initialize_llm(generate_options)
        self.agent = ChatAgent(llm=self.llm)

    @validate_call(config=ConfigDict(strict=True))
    def generate_questions_from_content_description(
        self, content_description: str
    ) -> GenerateResult:
        _log.debug(f"Output file: {self.options.generated_question_file.absolute()}")
        start_time = time.time()

        topics = self.generate_topics(content_description)

        result = self.generate_questions_from_topics(content_description, topics)
        end_time = time.time()
        result.time_taken = end_time - start_time

        return result

    def chat(self, message: str) -> str:
        messages = [ChatMessage(role="user", content=message)]
        resp = self.llm.chat(messages)
        block = resp.message.blocks[0]
        if isinstance(block, TextBlock):
            response_text = block.text
        else:
            _log.warning(f"Unexpected block type: {type(block)}")
            response_text = ""
        return response_text

    def generate_topics(self, content_description: str) -> dict[UserProfile, list[str]]:
        user_profiles = self.options.user_profiles
        topic_prompts = self.options.topic_prompts
        profiles_with_topics: dict[UserProfile, list[str]] = {}
        for user_profile in user_profiles:
            for prompt in topic_prompts:
                message = prompt.format(
                    content_description_str=content_description,
                    num_topics=user_profile.number_of_topics,
                    user_profile_str=user_profile.description,
                )
                response_text = self.chat(message)
                topic_list = _extract_list_items(response_text)
                if user_profile not in profiles_with_topics:
                    profiles_with_topics[user_profile] = topic_list
                else:
                    profiles_with_topics[user_profile].extend(topic_list)
        return profiles_with_topics

    def generate_questions_from_topics(
        self,
        content_description: str,
        profiles_with_topics: dict[UserProfile, list[str]],
    ) -> GenerateResult:
        question_prompts = self.options.question_prompts
        additional_instructions = self.options.additional_instructions
        start_time = time.time()
        num_questions_expected = _compute_num_questions_expected(
            profiles_with_topics, question_prompts
        )
        i = 0

        # If the file already exists, we remove it so that we don't get duplicate
        # questions.  Note that this is different from the behavior of the
        # generate_questions_from_chunks method in generate.py, which allows you to
        # resume generation from the last question that was generated if it fails.
        # That is a nice feature, but it would be more difficult to implement for
        # conceptual generation because we don't have chunk IDs.  In principle, we
        # could produce IDs for user_profile, topic, and question_prompt, and iteration
        # number tuples, but it would be more complex to implement, and it is not clear
        # that it is needed because question generation is fairly lightweight.
        if os.path.exists(self.options.generated_question_file):
            _log.debug(
                f"Removing existing file: {self.options.generated_question_file}"
            )
            os.remove(self.options.generated_question_file)

        questions: set[str] = set()

        progress_bar = tqdm(total=num_questions_expected, desc="Generating questions")
        for user_profile, topics in profiles_with_topics.items():
            for topic in topics:
                for question_prompt in question_prompts:
                    existing_questions_for_topic_and_prompt: list[str] = []
                    for _ in range(user_profile.number_of_iterations_per_topic):
                        additional_instruction = additional_instructions[
                            i % len(additional_instructions)
                        ]
                        existing_questions_str = (
                            "\n".join(existing_questions_for_topic_and_prompt)
                            if existing_questions_for_topic_and_prompt
                            else "NONE"
                        )
                        message = question_prompt.template.format(
                            content_description_str=content_description,
                            topic_str=topic,
                            existing_questions_str=existing_questions_str,
                            user_profile_str=user_profile.description,
                            additional_instructions_str=additional_instruction,
                        )
                        question_text = self.chat(message)
                        existing_questions_for_topic_and_prompt.append(question_text)

                        qa_labels = QALabelling(information=question_prompt.labels[0])
                        q_id: str = hashlib.sha256(question_text.encode()).hexdigest()
                        metadata = {
                            "conceptual": True,
                            "topic": topic,
                            "user_profile": user_profile.description,
                            "additional_instruction": additional_instruction,
                        }

                        # Sometimes we get duplicate questions either because the same
                        # question is generated for different profiles or topics or
                        # questions types or because the model ignored the instruction
                        # to not generate the same question.  We do not save duplicates.
                        if question_text in questions:
                            _log.debug(f"Skipping duplicate question: {question_text}")
                            continue
                        questions.add(question_text)
                        save_to_file(
                            objects=[
                                GenQAC(
                                    doc_id="",
                                    qac_id=q_id,
                                    context="",
                                    question=question_text,
                                    answer="",
                                    generated_question=True,
                                    generated_answer=False,
                                    retrieved_context=False,
                                    created=datetime.now(),
                                    model=self.options.model_id,
                                    paths=[""],
                                    chunk_id="",
                                    labels=qa_labels,
                                    metadata=metadata,
                                )
                            ],
                            out_file=self.options.generated_question_file,
                        )
                        progress_bar.update(1)
                        i += 1
        progress_bar.close()
        end_time = time.time()

        generate_res = GenerateResult(
            status=Status.SUCCESS,
            time_taken=(end_time - start_time),
            num_qac=i,
            output=self.options.generated_question_file,
        )

        return generate_res

    def generate_answers_using_retrieval(self, chunk_file: Path) -> GenerateResult:
        _log.debug(f"Input file: {self.options.generated_question_file.absolute()}")
        _log.debug(f"Output file: {self.options.generated_qac_file.absolute()}")

        start_time = time.time()

        index = self._make_index(chunk_file)

        with open(self.options.generated_question_file, "r") as f:
            num_lines = sum(1 for line in f)

        qac_collection: Iterator[GenQAC] = retrieve_stored_qac(
            in_file=self.options.generated_question_file
        )
        num_qac = 0
        for qac in tqdm(qac_collection, total=num_lines, desc="Generating answers"):
            num_qac += 1

            # If a QAC already has a generated answer, skip it
            if qac.generated_answer:
                continue

            answer, context = self._generate_answer(qac, index)

            save_to_file(
                objects=[
                    GenQAC(
                        doc_id=qac.doc_id,
                        qac_id=qac.qac_id,
                        context=context,
                        question=qac.question,
                        answer=answer,
                        generated_question=True,
                        generated_answer=True,
                        retrieved_context=True,
                        created=datetime.now(),
                        model=self.options.model_id,
                        paths=qac.paths,
                        chunk_id=qac.chunk_id,
                        labels=qac.labels,
                        metadata=qac.metadata,
                    )
                ],
                out_file=self.options.generated_qac_file,
            )

        end_time = time.time()

        generate_res = GenerateResult(
            status=Status.SUCCESS,
            time_taken=(end_time - start_time),
            num_qac=num_qac,
            output=self.options.generated_qac_file,
        )
        return generate_res

    def _generate_answer(self, qac: GenQAC, index: VectorStoreIndex) -> tuple[str, str]:
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=NUMBER_OF_SEARCH_RESULTS_TO_CONSIDER,
        )

        # Create LLMRerank postprocessor
        reranker = LLMRerank(
            llm=self.llm,
            top_n=NUMBER_OF_SEARCH_RESULTS_TO_SELECT_FOR_ANSWER_GENERATION,
            # Process nodes in batches for efficiency.  You don't want to make this the
            # batch size too big because you'll run out of context window size for the
            # model or just confuse the model with more context than it can handle.
            choice_batch_size=REFERENCE_ANSWER_GENERATION_RERANKER_BATCH_SIZE,
        )

        # Retrieve initial set of nodes
        nodes = retriever.retrieve(qac.question)

        if not nodes:
            return "", ""

        # Use LLMRerank to rerank and select the best nodes
        query_bundle = QueryBundle(query_str=qac.question)
        reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

        # Note we get no reranked nodes if   the question is not relevant to the index:
        # the reranker will return an empty list and then we have no reference contexts.
        # We could still ask the generator model to generate an answer, but it will be a
        # generic answer that is not useful for evaluation of RAG solutions.
        if not reranked_nodes:
            return "", ""

        # Extract the text from the reranked nodes
        reference_contexts = [_format_context_node(node) for node in reranked_nodes]
        context_str = "\n-------\n".join(reference_contexts)

        # Generate the reference answer using the selected contexts
        message_text = DEFAULT_TEXT_QA_PROMPT_TMPL.format(
            **{"context_str": context_str, "query_str": qac.question}
        )
        answer = self.chat(message_text)
        return answer, context_str

    def _make_index(self, chunk_file: Path) -> VectorStoreIndex:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)

        # Create an empty VectorStoreIndex
        index = VectorStoreIndex([], embed_model=embed_model)

        with open(chunk_file, "r") as f:
            num_lines = sum(1 for line in f)

        with (
            tqdm(total=num_lines, desc="Adding chunks to index") as progress_bar,
            jsonlines.open(chunk_file) as reader,
        ):
            for chunk_data in reader:
                progress_bar.update(1)
                # Extract text and metadata from each chunk
                text = chunk_data.get("text", "")
                meta = chunk_data.get("meta", {})

                # Create a Document object
                doc = Document(
                    text=text,
                    metadata={
                        "headings": meta.get("headings", []),
                    },
                )

                # Add the document to the index
                index.insert(doc)
        return index


def _format_context_node(node: Any) -> str:
    result = ""
    text = node.node.text
    headings = node.node.metadata.get("headings", [])
    for i, heading in enumerate(headings):
        result += f"{'#' * (i + 1)} {heading}\n"
    result += "\n" + text
    return result


def _compute_num_questions_expected(
    profiles_with_topics: dict[UserProfile, list[str]],
    question_prompts: list[QaPromptTemplate],
) -> int:
    num_question_prompts = len(question_prompts)
    n = 0
    for user_profile, topics in profiles_with_topics.items():
        n += (
            len(topics)
            * user_profile.number_of_iterations_per_topic
            * num_question_prompts
        )
    return n


def _extract_list_items(text_block: str) -> list[str]:
    """Extracts items from a multi-line string.

    It handles:
    - Optional blank lines between items.
    - Optional numbering (e.g., "1.", "1 ", "2.") at the start of lines.
    It returns a list of strings, with numbers and blank lines removed,
    and each item stripped of leading/trailing whitespace.

    Args:
        text_block: The multi-line string to process.

    Returns:
        A list of extracted string values.
    """
    items = []
    # Regex to identify leading numbers + an optional period and optional whitespace.
    # Example: "1.", "1 ", "  1. ", "2 "
    # This pattern is applied to lines that have already had outer whitespace stripped.
    # ^      Matches the beginning of the string (the stripped line).
    # \d+    Matches one or more digits (the number).
    # \.?    Matches an optional literal period.
    # \s* Matches zero or more whitespace characters following the number/period.
    number_prefix_pattern = re.compile(r"^\d+\.?\s*")

    for line in text_block.splitlines():
        # 1. Remove leading/trailing whitespace from the current line.
        stripped_line = line.strip()

        # 2. If the line is blank after stripping, skip it.
        if not stripped_line:
            continue

        # 3. Remove the number prefix, if present.
        #    The sub() method replaces the matched pattern with an empty string.
        item_text = number_prefix_pattern.sub("", stripped_line)

        # 4. Strip any leading/trailing whitespace that might remain on the item_text.
        #    This is important if the original item had spaces after the number,
        #    or if the item itself had leading/trailing spaces (which strip() in step 1
        #    would have handled if no number was present, but this ensures cleanliness
        #    after potential prefix removal).
        final_item_text = item_text.strip()

        # 5. Add the cleaned item to the list, only if it's not empty.
        #    (e.g., a line like "1." would become "" after processing).
        if final_item_text:
            items.append(final_item_text)

    return items
