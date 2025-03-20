"""Set of prompts for question-answering generation."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from docling_core.types.nlp.qa_labels import QAInformationLabel

DEFAULT_COMBINED_QUESTION_PROMPT = (
    "I will provide you a text passage. I need you to generate three questions that "
    "must be answered only with information contained in this passage, and nothing "
    "else.\n"
    'The first question is of type "fact_single", which means that the answer to this '
    "question is a simple, single piece of factual information contained in the "
    "context.\n"
    'The second question is of type "summary", which means that the answer to this '
    "question summarizes different pieces of factual information contained in the "
    "context.\n"
    'The third question is of type "reasoning", which is a question that requires the '
    "reader to think critically and make an inference or draw a conclusion based on "
    "the information provided in the passage.\n"
    "Make sure that the three questions are different.\n"
    "\n"
    "You will format your generation as a python dictionary, such as:\n"
    '{"fact_single": <The "fact_single" type question you thought of>, '
    '"fact_single_answer: <Answer to the "fact_single" question>, "summary": <the '
    '"summary" type question you thought of>, "summary_answer": <Answer to the '
    '"summary" question>, "reasoning": <the "reasoning" type question you thought '
    'of>, "reasoning_answer": <Answer to the "reasoning" question>}\n'
    "\n"
    "Only provide the python dictionary as your output.\n"
    "\n"
    "Context: {context_str}"
)

DEFAULT_FACT_SINGLE_QUESTION_PROMPT = (
    'A "single-fact" question is a question with the following properties:\n'
    "- It is a natural language question.\n"
    "- It is answered with a single piece of factual information.\n"
    "\n"
    "I will provide you with a context.\n"
    "Think of a single-fact question that must be answered using only information "
    "contained in the given context.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "What question did you think about? Do not say anything other than the question."
)

DEFAULT_FACT_SINGLE_ANSWER_PROMPT = (
    'A "single-fact" answer is an answer with the following properties:\n'
    "- It contains a single piece of factual information.\n"
    "\n"
    "I will provide you with a context and a question that should be answered with a "
    '"single-fact" answer.\n'
    'This "single-fact" answer should only contain information from the context, '
    "and nothing else.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "Question: {question_str}\n"
    "\n"
    "What is your answer? Do not say anything other than the answer."
)

DEFAULT_SUMMARY_QUESTION_PROMPT = (
    'A "summary" question is a question with the following properties:\n'
    "- It is a natural language question.\n"
    "- It is answered with a summary of multiple pieces of information.\n"
    "- It cannot be answered with a single piece of factual information.\n"
    "\n"
    "I will provide you with a context.\n"
    'Think of a "summary" question that must be answered using only information '
    "contained in the given context.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "What question did you think about? Do not say anything other than the question."
)

DEFAULT_SUMMARY_ANSWER_PROMPT = (
    'A "summary" answer is an answer with the following properties:\n'
    "- It contains a summary of multiple pieces of information.\n"
    "- It contains more information than a single piece of factual information.\n"
    "\n"
    "I will provide you with a context and a question that should be answered with a "
    '"summary" answer.\n'
    'This "summary" answer should only contain information from the context, and '
    "nothing else.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "Question: {question_str}\n"
    "\n"
    "What is your answer? Do not say anything other than the answer."
)

DEFAULT_REASONING_QUESTION_PROMPT = (
    'A "reasoning" question is a question with the following properties:\n'
    "- It is a natural language question.\n"
    "- It requires the reader to think critically and make an inference or draw a "
    "conclusion based on the information provided in the passage.\n"
    "\n"
    "I will provide you with a context.\n"
    'Think of a "reasoning" question that must be answered using only information '
    "contained in the given context.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "What question did you think about? Do not say anything other than the question."
)

DEFAULT_REASONING_ANSWER_PROMPT = (
    'A "reasoning" answer is an answer with the following properties:\n'
    "- It contains a summary of multiple pieces of information.\n"
    "- It clearly states an inference or conclusion that can be drawn from a "
    "passage.\n"
    "\n"
    "I will provide you with a context and a question that should be answered with a "
    '"reasoning" answer.\n'
    'This "reasoning" answer should only contain information from the context, and '
    "nothing else.\n"
    "\n"
    "Context: {context_str}\n"
    "\n"
    "Question: {question_str}\n"
    "\n"
    "What is your answer? Do not say anything other than the answer."
)


class PromptTypes(str, Enum):
    """Supported prompt types for Q&A generation."""

    QUESTION = "question"
    ANSWER = "answer"


class QaPromptTemplate(BaseModel):
    """Prompt template for Q&A generation."""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    template: str
    keys: list[str] = []
    labels: list[QAInformationLabel] = []
    type_: PromptTypes = Field(alias="type")

    @model_validator(mode="after")
    def check_keys_in_template(self) -> Self:
        if self.keys is not None:
            for item in self.keys:
                placeholder = "{" + item + "}"
                if placeholder not in self.template:
                    raise ValueError(f"key {item} not found in template")
        return self


default_combined_question_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_COMBINED_QUESTION_PROMPT,
    keys=["context_str"],
    labels=["fact_single", "summary", "reasoning"],
    type_="question",
)
default_fact_single_question_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_FACT_SINGLE_QUESTION_PROMPT,
    keys=["context_str"],
    labels=["fact_single"],
    type_="question",
)
default_fact_single_answer_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_FACT_SINGLE_ANSWER_PROMPT,
    keys=["context_str", "question_str"],
    labels=["fact_single"],
    type_="answer",
)
default_summary_question_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_SUMMARY_QUESTION_PROMPT,
    keys=["context_str"],
    labels=["summary"],
    type_="question",
)
default_summary_answer_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_SUMMARY_ANSWER_PROMPT,
    keys=["context_str", "question_str"],
    labels=["summary"],
    type_="answer",
)
default_reasoning_question_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_REASONING_QUESTION_PROMPT,
    keys=["context_str"],
    labels=["reasoning"],
    type_="question",
)
default_reasoning_answer_qa_prompt: QaPromptTemplate = QaPromptTemplate(
    template=DEFAULT_REASONING_ANSWER_PROMPT,
    keys=["context_str", "question_str"],
    labels=["reasoning"],
    type_="answer",
)

default_prompt_templates: list[QaPromptTemplate] = [
    default_combined_question_qa_prompt,
    default_fact_single_question_qa_prompt,
    default_fact_single_answer_qa_prompt,
    default_summary_question_qa_prompt,
    default_summary_answer_qa_prompt,
    default_reasoning_question_qa_prompt,
    default_reasoning_answer_qa_prompt,
]


# Specifying Context Prompts

SPECIFY_CONTEXT_ROLE_DESCRIPTION = (
    "You are a filtering machine, that selects the exact sentences within a given "
    "'context' paragraph, that a given 'question' and 'answer' is based upon.\n"
    "The context paragraph is already split into a dictionary with its indexed "
    "individual sentences."
    "Select the one or many sentences that are required to answer the given "
    "question\n"
    "Output only index number(s). Nothing else!"
    "If the question and answer are based on multiple sentences, output the page "
    "numbers, separated by a comma.\n"
    "Text example:\n"
    "Input:\n\n"
    "Question: Is Ministernotomy for AVR generally a safe method?\n"
    "Answer: Yes\n"
    "Context:\n"
    "{'[1]': 'Material and methods: The total study population was divided into 2 "
    "demographically homogeneous groups: mini-AVR (n = 74) and con-AVR (n = 76).', "
    "'[2]': 'There were no statistically significant differences in preoperative "
    "echocardiography.', '[3]': 'Results: Aortic cross-clamp time and "
    "cardiopulmonary bypass time were significantly longer in the mini-AVR group.',"
    " '[4]': 'Shorter mechanical ventilation time, hospital stay and lower "
    "postoperative drainage were observed in the mini-AVR group (p < 0.05).', "
    "'[5]': 'Biological prostheses were more frequently implanted in the mini-AVR "
    "group (p < 0.05).', '[6]': 'Patients from the mini-AVR group reported less "
    "postoperative pain.', '[7]': 'No significant differences were found in the "
    "diameter of the implanted aortic prosthesis, the amount of inotropic agents "
    "and painkillers, postoperative left ventricular ejection fraction (LVEF), "
    "medium and maximum transvalvular gradient or the number of transfused blood "
    "units.', '[8]': 'There were no differences in the frequency of postopera tive "
    "complications such as mortality, stroke, atrial fibrillation, renal failure, "
    "wound infection, sternal instability, or the need for rethoracotomy.', '[9]': "
    "'Ministernotomy for AVR is a safe method and does not increase morbidity and "
    "mortality.', '[10]': 'It significantly re duces post-operative blood loss and "
    "shortens hospital stay.', '[11]': 'Ministernotomy can be successfully used as "
    "an alternative method to sternotomy.']"
    "LLM Output:\n\n"
    "9"
)
