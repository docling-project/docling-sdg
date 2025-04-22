import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional, Type, Union

import typer
from dotenv import load_dotenv
from llama_index.llms.ibm.base import GenTextParamsMetaNames
from pydantic import AnyUrl, TypeAdapter
from rich.console import Console

from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling_core.types.doc import DocItemLabel
from docling_core.types.io import DocumentStream
from docling_core.utils.file import resolve_source_to_path

from docling_sdg.qa.base import (
    Chunker,
    CritiqueOptions,
    CritiqueResult,
    GenerateOptions,
    GenerateResult,
    LlmOptions,
    LlmProvider,
    SampleOptions,
    SampleResult,
)
from docling_sdg.qa.critique import Judge
from docling_sdg.qa.generate import Generator
from docling_sdg.qa.sample import PassageSampler

_log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, add_completion=False)

console = Console()
err_console = Console(stderr=True)

QaOption = Union[SampleOptions, CritiqueOptions, CritiqueOptions]


def get_option_def(field: str, option: Type[QaOption]) -> Any:
    field_info = option.model_fields.get(field)
    if field_info is None:
        return None
    else:
        return field_info.default


def get_option_desc(field: str, option: Type[QaOption]) -> Optional[str]:
    field_info = option.model_fields.get(field)
    if field_info is None:
        return None
    else:
        return field_info.description


def set_watsonx_options(options: LlmOptions) -> None:
    if "WATSONX_URL" in os.environ:
        options.url = TypeAdapter(AnyUrl).validate_python(os.environ.get("WATSONX_URL"))
    if "WATSONX_MODEL_ID" in os.environ:
        options.model_id = TypeAdapter(str).validate_python(
            os.environ.get("WATSONX_MODEL_ID")
        )
    if "WATSONX_MAX_NEW_TOKENS" in os.environ:
        options.max_new_tokens = TypeAdapter(int).validate_python(
            os.environ.get("WATSONX_MAX_NEW_TOKENS")
        )
    if "WATSONX_DECODING_METHOD" in os.environ and options.additional_params:
        options.additional_params[GenTextParamsMetaNames.DECODING_METHOD] = TypeAdapter(
            str
        ).validate_python(os.environ.get("WATSONX_DECODING_METHOD"))
    if "WATSONX_MIN_NEW_TOKENS" in os.environ and options.additional_params:
        options.additional_params[GenTextParamsMetaNames.MIN_NEW_TOKENS] = TypeAdapter(
            int
        ).validate_python((os.environ.get("WATSONX_MIN_NEW_TOKENS")))
    if "WATSONX_TEMPERATURE" in os.environ and options.additional_params:
        options.additional_params[GenTextParamsMetaNames.TEMPERATURE] = TypeAdapter(
            float
        ).validate_python(os.environ.get("WATSONX_TEMPERATURE"))
    if "WATSONX_TOP_K" in os.environ and options.additional_params:
        options.additional_params[GenTextParamsMetaNames.TOP_K] = TypeAdapter(
            int
        ).validate_python((os.environ.get("WATSONX_TOP_K")))
    if "WATSONX_TOP_P" in os.environ and options.additional_params:
        options.additional_params[GenTextParamsMetaNames.TOP_P] = TypeAdapter(
            float
        ).validate_python(os.environ.get("WATSONX_TOP_P"))


@app.command(
    no_args_is_help=True,
    help=(
        "Prepare the data for SDG: parse and chunk documents to create a file "
        "with document passsages."
    ),
)
def sample(
    input_sources: Annotated[
        list[str],
        typer.Argument(
            ...,
            metavar="source",
            help=(
                "PDF files to convert, chunk, and sample. Can be local file / "
                "directory paths or URL."
            ),
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help=(
                "Set the verbosity level. -v for info logging, -vv for debug logging."
            ),
        ),
    ] = 0,
    sample_file: Annotated[
        Optional[Path],
        typer.Option(
            "--sample-file",
            "-f",
            help=get_option_desc("sample_file", SampleOptions),
        ),
    ] = get_option_def("sample_file", SampleOptions),
    chunker: Annotated[
        Optional[Chunker],
        typer.Option(
            "--chunker",
            "-c",
            help=get_option_desc("chunker", SampleOptions),
        ),
    ] = get_option_def("chunker", SampleOptions),
    min_token_count: Annotated[
        Optional[int],
        typer.Option(
            "--min-token-count",
            "-t",
            help=get_option_desc("min_token_count", SampleOptions),
        ),
    ] = get_option_def("min_token_count", SampleOptions),
    max_passages: Annotated[
        Optional[int],
        typer.Option(
            "--max-passages",
            "-p",
            help=get_option_desc("max_passages", SampleOptions),
        ),
    ] = get_option_def("max_passages", SampleOptions),
    doc_items: Annotated[
        Optional[list[DocItemLabel]],
        typer.Option(
            "--doc-items",
            "-d",
            help=get_option_desc("doc_items", SampleOptions),
        ),
    ] = get_option_def("doc_items", SampleOptions),
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            "-s",
            help=get_option_desc("seed", SampleOptions),
        ),
    ] = get_option_def("seed", SampleOptions),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    with tempfile.TemporaryDirectory() as tempdir:
        input_doc_paths: list[Path | str | DocumentStream] = []
        for src in input_sources:
            try:
                # check if we can fetch some remote url
                source = resolve_source_to_path(source=src, workdir=Path(tempdir))
                input_doc_paths.append(source)
            except FileNotFoundError as err:
                err_console.print(
                    f"[red]Error: The input file {src} does not exist.[/red]"
                )
                raise typer.Abort() from err
            except IsADirectoryError:
                # if the input matches to a file or a folder
                try:
                    local_path = TypeAdapter(Path).validate_python(src)
                    if local_path.exists() and local_path.is_dir():
                        for fmt in list(InputFormat):
                            for ext in FormatToExtensions[fmt]:
                                input_doc_paths.extend(
                                    list(local_path.glob(f"**/*.{ext}"))
                                )
                                input_doc_paths.extend(
                                    list(local_path.glob(f"**/*.{ext.upper()}"))
                                )
                    elif local_path.exists():
                        input_doc_paths.append(local_path)
                    else:
                        err_console.print(
                            f"[red]Error: The input file {src} does not exist.[/red]"
                        )
                        raise typer.Abort()
                except Exception as err:
                    err_console.print(f"[red]Error: Cannot read the input {src}.[/red]")
                    _log.info(err)  # will print more details if verbose is activated
                    raise typer.Abort() from err

        options: SampleOptions = SampleOptions(
            sample_file=sample_file,
            chunker=chunker,
            min_token_count=min_token_count,
            max_passages=max_passages,
            doc_items=doc_items,
            seed=seed,
        )
        sample = PassageSampler(sample_options=options)
        result: SampleResult = sample.sample(input_doc_paths)
        typer.echo(f"Q&A Sample finished: {result}")


@app.command(
    no_args_is_help=True,
    help=(
        "Run SDG on a set of document passages and create question-answering items "
        "of different types."
    ),
)
def generate(
    input_source: Annotated[
        Path,
        typer.Argument(
            ...,
            metavar="source",
            help="Path to a file with sample passages from documents.",
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    generated_file: Annotated[
        Optional[Path],
        typer.Option(
            "--generated-file",
            "-f",
            help=get_option_desc("generated_file", CritiqueOptions),
        ),
    ] = get_option_def("generated_file", CritiqueOptions),
    max_qac: Annotated[
        Optional[int],
        typer.Option(
            "--max-qac",
            "-q",
            help=get_option_desc("max_qac", CritiqueOptions),
        ),
    ] = get_option_def("max_qac", CritiqueOptions),
    watsonx: Annotated[
        Optional[Path],
        typer.Option(
            "--watsonx",
            "-w",
            help="Path to a file with the parameters for watsonx.ai.",
        ),
    ] = Path("./.env"),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    if not input_source.is_file():
        err_console.print(
            f"[red]Error: The input file {input_source} does not exist.[/red]"
        )
        raise typer.Abort()

    if not watsonx or not os.path.isfile(watsonx):
        err_console.print(
            f"[red]Error: The watsonx.ai file {watsonx} does not exist.[/red]"
        )
        raise typer.Abort()

    load_dotenv(watsonx)

    options = GenerateOptions(
        provider=LlmProvider.WATSONX,
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
        api_key=os.environ.get("WATSONX_APIKEY"),
    )

    set_watsonx_options(options)
    if generated_file:
        options.generated_file = generated_file
    if max_qac:
        options.max_qac = max_qac

    generator: Generator = Generator(generate_options=options)
    result: GenerateResult = generator.generate_from_sample(input_source)
    typer.echo(f"Q&A Generation finished: {result}")


@app.command(
    no_args_is_help=True,
    help="Use LLM as a judge to critique a set of SDG question-answering items.",
)
def critique(
    input_source: Annotated[
        Path,
        typer.Argument(
            ...,
            metavar="source",
            help="Path to a file with generated Q&A items from document passages.",
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    critiqued_file: Annotated[
        Optional[Path],
        typer.Option(
            "--critiqued-file",
            "-f",
            help=get_option_desc("critiqued_file", CritiqueOptions),
        ),
    ] = get_option_def("critiqued_file", CritiqueOptions),
    max_qac: Annotated[
        Optional[int],
        typer.Option(
            "--max-qac",
            "-q",
            help=get_option_desc("max_qac", CritiqueOptions),
        ),
    ] = get_option_def("max_qac", CritiqueOptions),
    watsonx: Annotated[
        Optional[Path],
        typer.Option(
            "--watsonx",
            "-w",
            help="Path to a file with the parameters for watsonx.ai.",
        ),
    ] = Path("./.env"),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    if not input_source.is_file():
        err_console.print(
            f"[red]Error: The input file {input_source} does not exist.[/red]"
        )
        raise typer.Abort()

    if not watsonx or not os.path.isfile(watsonx):
        err_console.print(
            f"[red]Error: The watsonx.ai file {watsonx} does not exist.[/red]"
        )
        raise typer.Abort()

    load_dotenv(watsonx)

    options = CritiqueOptions(
        provider=LlmProvider.WATSONX,
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
        api_key=os.environ.get("WATSONX_APIKEY"),
    )

    set_watsonx_options(options)
    if critiqued_file:
        options.critiqued_file = critiqued_file
    if max_qac:
        options.max_qac = max_qac

    judge: Judge = Judge(critique_options=options)
    result: CritiqueResult = judge.critique(input_source)
    typer.echo(f"Q&A Critique finished: {result}")
