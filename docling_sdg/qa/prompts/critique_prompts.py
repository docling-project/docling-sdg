"""Set of prompts for question-answering critique."""

from pydantic import BaseModel, model_validator
from typing_extensions import Self

DEFAULT_Q_TO_C_GROUNDENESS_PROMPT = (
    "You will be given a context and a sentence that should be a question.\n"
    "Your task is to provide a 'total rating' scoring on how well one can answer the "
    "given question unambiguously with the given context.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the question is not "
    "answerable at all given the context, and 5 means that the question is clearly "
    "and unambiguously answerable with the context.\n"
    "If the sentence provided is not actually a question, rate it as 1.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here are the question and context.\n"
    "\n"
    'Question: "{question_str}"\n'
    "\n"
    'Context: "{context_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_A_TO_C_GROUNDENESS_PROMPT = (
    "You will be given a context and a passage.\n"
    "Your task is to provide a 'total rating' scoring on how well the statements in "
    "the provided passage can be infered from the provided context.\n"
    "Give your rating on a scale of 1 to 5, where 1 means that none of the statements "
    "in the passage can be inferred from the provided context, while 5 means that all "
    "of the statements in the passage can be unambiguously and entirely obtained from "
    "the context.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here are the context and passage.\n"
    "\n"
    'Context: "{context_str}"\n'
    "\n"
    'Passage: "{answer_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_Q_FEASIBILITY_PROMPT = (
    "You will be given a context and a question.\n"
    "This context is extracted from a collection of passages, and the question will "
    "be used to find it.\n"
    "Your task is to provide a 'total rating' scoring how well this context can be "
    "retrieved based on the specificity and pertinence of the question.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that it will be difficult "
    "to find this context from this question due to lack of specificity or "
    "pertinence, and 5 means that the context can clearly be found with information "
    "contained in the question.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here are the question and context.\n"
    "\n"
    'Question: "{question_str}"\n'
    "\n"
    'Context: "{context_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_STAND_ALONE_PROMPT = (
    "You will be given a question.\n"
    "Your task is to provide a 'total rating' representing how context-independent "
    "this question is.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the question depends "
    "on additional information to be understood, and 5 means that the question makes "
    "sense by itself.\n"
    "For instance, if the question refers to a particular setting, like 'in the "
    "context' or 'in the document', the rating must be 1.\n"
    "The questions can contain obscure technical nouns or acronyms and still be a 5: "
    "it must simply be clear to an operator with access to documentation what the "
    "question is about.\n"
    "\n"
    'For instance, "What is the name of the checkpoint from which the ViT model is '
    'imported?" should receive a 1, since there is an implicit mention of a context, '
    "thus the question is not independent from the context.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here is the question.\n"
    "\n"
    'Question: "{question_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_Q_USEFULNESS_PROMPT = (
    "You will be given a question.\n"
    "This question is to be used to find information in a collection of documents.\n"
    "Your task is to provide a 'total rating' representing how useful this question "
    "can be to a user with domain knowledge on the subject covered by the document "
    "collection.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the question is not "
    "useful at all, and 5 means that the question is extremely useful.\n:"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here is the question.\n"
    "\n"
    'Question: "{question_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_C_USEFULNESS_PROMPT = (
    "You will be given a context.\n"
    "This context is a part of a collection of contexts that users can query.\n"
    "Your task is to provide a 'total rating' representing how useful this context "
    "can be to extract statements for a user with domain knowledge on the subject "
    "covered by the context collection.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the context does not "
    "contain any useful statements, and 5 means that the context contains multiple "
    "statements that provide the user with different pieces of information.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here is the context.\n"
    "\n"
    'Context: "{context_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_C_CLARITY_PROMPT = (
    "You will be given a context.\n"
    "This context is a part of a collection of contexts that users can query.\n"
    "Your task is to provide a 'total rating' representing clarity of the information "
    "contained in the context.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the context contains "
    "incomplete, unclear or poorly formatted information, and 5 means that the context "
    "contains only complete, clear and well formatted statements.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here is the context.\n"
    "\n"
    'Context: "{context_str}"\n'
    "\n"
    "Answer:::"
)

DEFAULT_QA_TAUTOLOGY_PROMPT = (
    "You will be given a question and passage its answer.\n"
    "Your question is to judge whether this question and answer pair form a "
    "tautological exchange.\n"
    "Give your answer on a scale of 1 to 5, where 1 means that the question and "
    "answer repeat the same information, and 5 means that the answer is made of "
    "entirely new information.\n"
    "\n"
    "Provide your answer as a python dictionary as follows:\n"
    "\n"
    "Answer:::\n"
    '{{"evaluation": "Your rationale for the rating, as a brief and concise '
    'text", "rating": "your rating, as a number between 1 and 5"}}\n'
    "\n"
    "You MUST provide values for 'evaluation' and 'rating' in your answer. Provide "
    "ONLY the python dictionary as your answer.\n"
    "\n"
    "Now here are the question and its answer.\n"
    "\n"
    'Question: "{question_str}"\n'
    "\n"
    'Answer: "{answer_str}"\n'
    "\n"
    "Answer:::"
)


class CritiquePromptTemplate(BaseModel):
    """Prompt template for Q&A critique."""

    template: str
    keys: list[str] = []
    name: str

    @model_validator(mode="after")
    def check_keys_in_template(self) -> Self:
        if self.keys is not None:
            for item in self.keys:
                placeholder = "{" + item + "}"
                if placeholder not in self.template:
                    raise ValueError(f"key {item} not found in template")
        return self


default_q_to_c_groundedness_prompt = CritiquePromptTemplate(
    template=DEFAULT_Q_TO_C_GROUNDENESS_PROMPT,
    keys=["question_str", "context_str"],
    name="q_to_c_groundedness",
)
default_a_to_c_groundedness_prompt = CritiquePromptTemplate(
    template=DEFAULT_A_TO_C_GROUNDENESS_PROMPT,
    keys=["context_str", "answer_str"],
    name="a_to_c_groundedness",
)
default_q_feasibility_prompt = CritiquePromptTemplate(
    template=DEFAULT_Q_FEASIBILITY_PROMPT,
    keys=["question_str", "context_str"],
    name="q_feasibility",
)
default_stand_alone_prompt = CritiquePromptTemplate(
    template=DEFAULT_STAND_ALONE_PROMPT,
    keys=["question_str"],
    name="stand_alone",
)
default_q_usefulness_prompt = CritiquePromptTemplate(
    template=DEFAULT_Q_USEFULNESS_PROMPT,
    keys=["question_str"],
    name="q_usefulness",
)
default_c_usefulness_prompt = CritiquePromptTemplate(
    template=DEFAULT_C_USEFULNESS_PROMPT,
    keys=["context_str"],
    name="c_usefulness",
)
default_c_clarity_prompt = CritiquePromptTemplate(
    template=DEFAULT_C_CLARITY_PROMPT,
    keys=["context_str"],
    name="c_clarity",
)
default_qa_tautology_prompt = CritiquePromptTemplate(
    template=DEFAULT_QA_TAUTOLOGY_PROMPT,
    keys=["question_str", "answer_str"],
    name="qa_tautology",
)

default_critique_templates = [
    default_q_to_c_groundedness_prompt,
    default_a_to_c_groundedness_prompt,
    default_q_feasibility_prompt,
    default_stand_alone_prompt,
    default_q_usefulness_prompt,
    default_c_usefulness_prompt,
    default_c_clarity_prompt,
    default_qa_tautology_prompt,
]

# Modified Prompt
MODIFIED_Q_STAND_ALONE_PROMPT = (
    "You will be given a question.\n"
    "Your task is to provide a 'total rating' representing how inherently "
    "answerable this question is.\n"
    "Give your answer on a scale of 1 to 5."
    "A score of 1 means that the question is NOT inherently understandable on "
    "its own. That means, it requires more context or detail within the "
    "question to make sense on its own. Such questions have an undefined scope "
    "or refer to entities that are not clearly defined. For example they ask "
    "about a study, document, group, author, patient, quiz etc. without "
    "mentioning what specific study, document, group, author, patient, quiz "
    "they refer to.\n"
    "Examples are:\n"
    "- What was the primary objective of the study?\n"
    "- What is the percentage of students who scored 955 or above in the "
    "examination?\n"
    "- What is the aim of the study presented in the document?\n"
    "A score of 5 means that the question is a standalone question, that is "
    "inherently understandable. It makes sense without providing more context "
    "within the question. Such questions have a complete and closed scope. "
    "They can be answerable by finding a specific piece of information.\n"
    "Examples are:\n"
    "- What is the effect of mesenchymal stem cell therapy on aging frailty?\n"
    "- How is the INR related to the blood's tendency to clot, and how does "
    "warfarin affect the INR?\n"
    "- How many patients were included in the Yechoor study conducted in 2006 "
    "and what method was used to assess AS severity?\n"
)
