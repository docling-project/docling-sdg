import logging
import time
from pathlib import Path
from typing import Iterator

import tqdm
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.utils import format_string
from pydantic import ConfigDict, ValidationError, validate_call

from docling_sdg.qa.base import (
    Critique,
    CritiqueOptions,
    CritiqueResult,
    GenQAC,
    Status,
)
from docling_sdg.qa.utils import (
    ChatAgent,
    initialize_llm,
    retrieve_stored_qac,
    retrieve_stored_qac_ids,
    save_to_file,
)

_log = logging.getLogger(__name__)


class Judge:
    def __init__(
        self,
        critique_options: CritiqueOptions,
    ):
        self.options = critique_options or CritiqueOptions()
        self.llm: LLM = initialize_llm(critique_options)
        self.agent = ChatAgent(llm=self.llm)

    @staticmethod
    def _get_eval_and_score(reply: str) -> Critique:
        required: list[str] = ["evaluation", "rating", "{", "}"]
        if not reply or any(item not in reply for item in required):
            critique: Critique = Critique(evaluation="non-valid", rating=None)
            _log.debug(f"Invalid critique reply: {reply}")
        else:
            dict_str = reply[reply.find("{") : reply.rfind("}") + 1]
            try:
                critique = Critique.model_validate_json(dict_str)
            except ValidationError as ve:
                _log.debug(f"Unable to instantiate a Critique from reply: {repr(ve)}")
                critique = Critique(evaluation="non-valid", rating=None)

        return critique

    def _critique_qac(self, qac: GenQAC) -> dict[str, Critique]:
        judge_result: dict[str, Critique] = {}

        for prompt_template in self.options.prompts:
            prompt = format_string(
                prompt_template.template,
                **{
                    x: getattr(qac, x.removesuffix("_str"))
                    for x in prompt_template.keys
                },
            )
            reply = self.agent.ask(prompt, self.options.max_new_tokens)
            judge_result[prompt_template.name] = Judge._get_eval_and_score(reply)

        return judge_result

    @validate_call(config=ConfigDict(strict=True))
    def critique(self, source: Path) -> CritiqueResult:
        _log.debug(f"Output file: {self.options.critiqued_file.absolute()}")
        start_time = time.time()

        qac_collection: Iterator[GenQAC] = retrieve_stored_qac(in_file=source)

        stored_critique_ids: list[str] = [
            qac_id for qac_id, _ in retrieve_stored_qac_ids(self.options.critiqued_file)
        ]
        num_stored_critiques: int = len(stored_critique_ids)

        num_exported_critiques: int = 0
        if num_stored_critiques >= self.options.max_qac:
            end_time = time.time()
        else:
            num_extra_critiques: int = self.options.max_qac - num_stored_critiques
            for qac in tqdm.tqdm(qac_collection):
                if num_exported_critiques >= num_extra_critiques:
                    break

                # If a QAC has already been critiqued, skip it
                if qac.qac_id in stored_critique_ids:
                    continue

                critique: dict[str, Critique] = self._critique_qac(qac)
                new_qac = qac.model_copy(update={"critiques": critique}, deep=True)

                num_exported_critiques += 1
                save_to_file(objects=[new_qac], out_file=self.options.critiqued_file)

            end_time = time.time()

        critique_res = CritiqueResult(
            status=Status.SUCCESS,
            time_taken=(end_time - start_time),
            num_qac=num_exported_critiques,
            output=self.options.critiqued_file,
        )

        return critique_res
