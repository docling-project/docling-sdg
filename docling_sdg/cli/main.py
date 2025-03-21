import importlib
import logging
import platform
import sys
import warnings

import typer
from rich.console import Console

from docling_sdg.cli.qa import app as qa_app

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic")

_log = logging.getLogger(__name__)

err_console = Console(stderr=True)


app = typer.Typer(
    name="Docling SDG",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.command(name="version", help="Show version information.")
def get_version() -> None:
    docling_sdg_version = importlib.metadata.version("docling_sdg")
    platform_str = platform.platform()
    py_impl_version = sys.implementation.cache_tag
    py_lang_version = platform.python_version()
    typer.echo(f"Docling SDG version: {docling_sdg_version}")
    typer.echo(f"Python: {py_impl_version} ({py_lang_version})")
    typer.echo(f"Platform: {platform_str}")


app.add_typer(qa_app, name="qa", help="Interact with SDG for question-answering.")
