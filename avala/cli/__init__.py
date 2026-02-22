"""Avala CLI — command-line interface for the Avala API."""

from __future__ import annotations

import sys

try:
    import click
except ImportError:
    raise SystemExit("CLI dependencies not installed. Run: pip install avala[cli]")

from avala import Client


@click.group()
@click.option("--api-key", envvar="AVALA_API_KEY", default=None, help="Avala API key (or set AVALA_API_KEY)")
@click.option("--base-url", envvar="AVALA_BASE_URL", default=None, help="Avala API base URL (or set AVALA_BASE_URL)")
@click.pass_context
def main(ctx: click.Context, api_key: str | None, base_url: str | None) -> None:
    """Avala CLI — interact with the Avala API from your terminal."""
    ctx.ensure_object(dict)

    if ctx.invoked_subcommand == "configure":
        return

    try:
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        ctx.obj["client"] = Client(**kwargs)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Run 'avala configure' to set up your API key.", err=True)
        sys.exit(1)


# Import and register subcommands
from avala.cli.configure import configure  # noqa: E402
from avala.cli.datasets import datasets  # noqa: E402
from avala.cli.exports import exports  # noqa: E402
from avala.cli.projects import projects  # noqa: E402
from avala.cli.storage_configs import storage_configs  # noqa: E402
from avala.cli.tasks import tasks  # noqa: E402

main.add_command(configure)
main.add_command(datasets)
main.add_command(exports)
main.add_command(projects)
main.add_command(storage_configs)
main.add_command(tasks)
