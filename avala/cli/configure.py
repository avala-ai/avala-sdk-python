"""CLI configure command."""

from __future__ import annotations

import shlex

import click


@click.command()
def configure() -> None:
    """Configure API key and base URL for the Avala CLI."""
    click.echo("Configure your Avala CLI credentials.\n")

    api_key = click.prompt("API Key", type=str)
    base_url = click.prompt(
        "Base URL",
        type=str,
        default="https://api.avala.ai/api/v1",
        show_default=True,
    )

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 16 else "****"

    click.echo("\nAdd these to your shell profile (.bashrc, .zshrc, etc.):\n")
    click.echo(f"  export AVALA_API_KEY={shlex.quote(api_key)}")
    if base_url != "https://api.avala.ai/api/v1":
        click.echo(f"  export AVALA_BASE_URL={shlex.quote(base_url)}")
    click.echo(f"\n  (Key shown: {masked_key} — clear your terminal history after copying)")
    click.echo("\nOr pass them as flags: avala --api-key <key> <command>")
