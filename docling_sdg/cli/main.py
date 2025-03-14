import importlib
import logging
import platform
import sys
import warnings
from typing import Annotated, Optional

import typer
from rich.console import Console

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic")

_log = logging.getLogger(__name__)

err_console = Console(stderr=True)


app = typer.Typer(
    name="Docling SDG",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool) -> None:
    if value:
        docling_sdg_version = importlib.metadata.version("docling_sdg")
        platform_str = platform.platform()
        py_impl_version = sys.implementation.cache_tag
        py_lang_version = platform.python_version()
        print(f"Docling SDG version: {docling_sdg_version}")
        print(f"Python: {py_impl_version} ({py_lang_version})")
        print(f"Platform: {platform_str}")
        raise typer.Exit()


@app.command(no_args_is_help=True)
def sdg(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version information.",
        ),
    ] = None,
) -> None:
    pass


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
