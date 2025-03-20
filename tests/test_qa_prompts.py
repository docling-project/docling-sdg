"""Test package core/qa/prompts."""

from llama_index.core import PromptTemplate

from docling_sdg.qa.prompts.generation_prompts import DEFAULT_SUMMARY_QUESTION_PROMPT


def test_generation_prompts() -> None:
    context = (
        "Retrieval augmented generation (RAG) is a type of generative artificial "
        "intelligence that has information retrieval capabilities.Retrieval augmented "
        "generation (RAG) is a type of generative artificial intelligence that has "
        "information retrieval capabilities."
    )
    prompt = DEFAULT_SUMMARY_QUESTION_PROMPT.format(context_str=context)
    assert prompt

    li_template = PromptTemplate(DEFAULT_SUMMARY_QUESTION_PROMPT)
    li_prompt = li_template.format(context_str=context)

    assert prompt == li_prompt
