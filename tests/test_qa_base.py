"""Test module core/qa/base.py."""

import pytest
from llama_index.llms.openai_like import OpenAILike
from pydantic import SecretStr

from docling_sdg.qa.base import (
    LlmOptions,
    LlmProvider,
    QaPromptTemplate,
)
from docling_sdg.qa.prompts.generation_prompts import PromptTypes
from docling_sdg.qa.utils import initialize_llm


def test_llm_init() -> None:
    options = LlmOptions(
        project_id=SecretStr("any_project_id"),
        api_key=SecretStr("fake"),
    )

    options.provider = LlmProvider.OPENAI_LIKE
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
