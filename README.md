<p align="center">
  <a href="https://github.com/docling-project/docling-sdg">
    <img loading="lazy" alt="Docling" src="https://github.com/docling-project/docling-sdg/raw/main/docs/assets/docling-sdg-pic.png" width="40%"/>
  </a>
</p>

# Docling SDG

[![Platforms](https://img.shields.io/badge/platform-macos%20|%20linux%20|%20windows-blue)](https://github.com/docling-project/docling-parse/)
[![PyPI version](https://img.shields.io/pypi/v/docling-sdg)](https://pypi.org/project/docling-sdg/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-sdg)](https://pypi.org/project/docling-sdg/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-parse)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling-sdg/month)](https://pepy.tech/projects/docling-sdg)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling for Synthetic Data Generation (SDG) provides a set of tools to create artificial data from documents, leveraging generative AI and Docling's parsing capabilities.

## Features

* 🧬 Generation of question-answering pairs from passages of [multiple document formats][supported_formats] including 
PDF, HTML, or DOCX, leveraging Docling's parsing capabilities
* ⚖️ LLM as a judge for high quality question-answering pairs
* 💻 Simple and convenient CLI

### Coming soon

* 📝 Integrations with Llama Stack and vLLM
* 📝 SDG on tabular data
* 📝 Documentation

## Installation

To use Docling SDG, simply install `docling-sdg` from your package manager, e.g., pip:

```bash
pip install docling-sdg
```

Alternatively, you can clone this repository and use [uv](https://docs.astral.sh/uv) for
creating a virtual environment, installing the packages, and running the project commands.

```bash
git clone git@github.com:docling-project/docling-sdg.git
cd docling-sdg
uv sync
```

## Getting started

You can create synthetically-generated questions and answers from relevant parts of one or several documents.
These question-answer pairs may be used in AI applications, such as evaluating a RAG application or generating
ground truth to train a language model.


### Sample

Generating and judging data with LLMs may be computationally intense. Since document collections may be large,
you may want to chunk the documents into passages, filter them based on length and content criteria, and sample
a bunch of them to have a manageable dataset.

```python
from docling_sdg.qa.sample import PassageSampler

source = "https://en.wikipedia.org/wiki/Duck"
passage_sampler = PassageSampler()
print(passage_sampler.sample(source))
```

By default, the results will be exported to the file `docling_sdg_sample.jsonl`. Every line represents a document passage.

### Generate

For each passage created in the previous step, we can prompt an LLM and generate 3 different questions of the following
types: _simple fact_, _summary_, and _reasoning_.

The `GenerateOptions` class controls which model provider is used for Q&A generation by setting the `provider` attribute, as shown below. Three options are available:

* `LlmProvider.WATSONX` for [watsonx.ai](https://www.ibm.com/products/watsonx-ai);, you will need to provide a watsonx.ai instance ID and an API key.
* `LlmProvider.OPENAI` for OpenAI; you will need to provide an OpenAI API key
* `LlmProvider.OPENAI_LIKE` for any model provider with OpenAI compatible APIs; if no API key is needed (such as when running against `ollama` locally), set `api_key` to any string, e.g. `"fake"`

```python
import os
from docling_sdg.qa.base import GenerateOptions, LlmProvider
from docling_sdg.qa.generate import Generator
from pathlib import Path

options = GenerateOptions(
    provider=LlmProvider.WATSONX,
    project_id=os.environ.get("WATSONX_PROJECT_ID"),
    api_key=os.environ.get("WATSONX_APIKEY"),
    url=os.environ.get("WATSONX_URL"),
)

generator = Generator(generate_options=options)
print(generator.generate_from_sample(Path("docling_sdg_sample.jsonl")))
```

By default, the results will be exported to the file `docling_sdg_generated_qac.jsonl`. Every line represents a generated
question-answer-context item with additional information like the question type.


### Critique

Certain applications may require certain quality in the generated data. The last step consists of using an LLM to judge
the generated data and provide both qualitative and quantiative evaluations of the question-answer-context items. Using
those evaluations, we can filter the generated dataset to the required quality levels.

```python
import os
from docling_sdg.qa.base import CritiqueOptions, LlmProvider
from docling_sdg.qa.critique import Judge
from pathlib import Path

options = CritiqueOptions(
    provider=LlmProvider.WATSONX,
    project_id=os.environ.get("WATSONX_PROJECT_ID"),
    api_key=os.environ.get("WATSONX_APIKEY"),
    url=os.environ.get("WATSONX_URL"),
)

judge = Judge(critique_options=options)
print(judge.critique(Path("docling_sdg_generated_qac.jsonl")))
```

By default, the results will be exported to the file `docling_sdg_critiqued_qac.jsonl`. The file content is similar to 
the one created in the [Generate](#generate) step, but it additionally contains the critique evaluation on several dimensions such as
_question to context groundness_, _question feasibility_ or _context usefulness_.


## CLI

Docling SDG has a built-in CLI to run the 3 steps of the question-answering data generation.

```bash
docling-sdg qa sample https://en.wikipedia.org/wiki/Duck
docling-sdg qa generate docling_sdg_sample.jsonl
docling-sdg qa critique docling_sdg_generated.jsonl
```

Find out more about optional parameters with the help argument. For instance:

```bash
docling-sdg qa generate --help
```

## Get help and support

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling/discussions).

## Technical report

For more details on Docling SDG's inner workings, check out the paper [Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG System](https://aclanthology.org/2025.coling-industry.4.pdf), as well as [Docling Technical Report](https://arxiv.org/abs/2408.09869).

## Contributing

Please read [Contributing to Docling SDG](https://github.com/docling-project/docling-sdg/blob/main/CONTRIBUTING.md) for details.

## References

If you use Docling SDG in your projects, please consider citing the following:

```bib
@inproceedings{teixeira-de-lima-etal-2025-know,
    title={Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems}, 
    author={Rafael Teixeira de Lima and Shubham Gupta and Cesar Berrospi and Lokesh Mishra and Michele Dolfi and Peter Staar and Panagiotis Vagenas},
    year={2025},
    month={jan},
    booktitle={Proceedings of the 31st International Conference on Computational Linguistics: Industry Track},
    publisher={Association for Computational Linguistics},
    url={https://aclanthology.org/2025.coling-industry.4/}
}
```

## License

The Docling SDG codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.

