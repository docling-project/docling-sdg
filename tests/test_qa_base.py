"""Test module core/qa/base.py."""

import pytest
from llama_index.llms.ibm import WatsonxLLM
from llama_index.llms.openai_like import OpenAILike

from docling_sdg.qa.base import QaPromptTemplate, LlmOptions, LlmProviders, initialize_llm
from docling_sdg.qa.prompts.generation_prompts import PromptTypes

def test_llm_init() -> None:
    options = LlmOptions()
    options.provider = LlmProviders.WATSONX
    llm = initialize_llm(options)
    assert isinstance(llm, WatsonxLLM)

    options.provider = LlmProviders.OPENAI_LIKE
    llm = initialize_llm(options)
    assert isinstance(llm, OpenAILike)

def test_qa_prompt_template() -> None:
    template = (
        "Reply 'yes' if the following sentence is a question.\nSentence: {question}"
    )
    keys = ["question"]

    prompt = QaPromptTemplate(
        template=template, keys=keys, type_=PromptTypes.QUESTION, labels=["fact_single"]
    )
    assert prompt.template == template
    assert prompt.keys == keys
    assert prompt

    keys = ["question", "answer"]
    with pytest.raises(ValueError, match="key answer not found in template"):
        QaPromptTemplate(template=template, keys=keys, type_="question")
