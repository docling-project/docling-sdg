"""Test module core/qa/base.py."""

import pytest

from docling_sdg.qa.base import QaPromptTemplate
from docling_sdg.qa.prompts.generation_prompts import PromptTypes


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
