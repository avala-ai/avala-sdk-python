"""CLI configure command."""

from __future__ import annotations

import os

import click

from avala._config import DEFAULT_BASE_URL


@click.command()
def configure() -> None:
    """Configure API key and base URL for the Avala CLI."""
    click.echo("Configure your Avala CLI credentials.\n")

    # hide_input=True — never echo the key to the terminal. Otherwise it
    # appears on screen (shoulder-surfing range) and in any terminal
    # recording or screen-share.
    api_key = click.prompt("API Key", type=str, hide_input=True)
    base_url = click.prompt(
        "Base URL",
        type=str,
        default=DEFAULT_BASE_URL,
        show_default=True,
    )

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 16 else "****"

    # Validate the key by making a test request
    click.echo("\nValidating API key... ", nl=False)
    try:
        from avala import Client

        client = Client(api_key=api_key, base_url=base_url)
        org_page = client.organizations.list()
        click.echo(click.style("OK", fg="green"))
        if org_page.items:
            click.echo(f"  Organization: {org_page.items[0].name}")
    except Exception as exc:
        click.echo(click.style("FAILED", fg="red"))
        msg = str(exc).split("\n")[0]  # First line only — avoid raw Pydantic tracebacks
        click.echo(f"  {msg}")
        if not click.confirm("\nKey validation failed. Save anyway?", default=False):
            raise click.Abort()

    # Never print the full key to stdout — it ends up in shell history,
    # terminal recordings, CI logs, and clipboard managers. Show only the
    # masked key and instruct the user to set the env var from the
    # (non-echoed) value they just entered.
    click.echo(f"\nKey validated ({masked_key}).\n")
    click.echo(f"Add this to your shell profile ({_shell_profile()}):\n")
    click.echo("  export AVALA_API_KEY='<paste your key here>'")
    if base_url != DEFAULT_BASE_URL:
        click.echo(f"  export AVALA_BASE_URL='{base_url}'")
    click.echo("\nOr pass the key as a flag: avala --api-key <key> <command>")


def _shell_profile() -> str:
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return ".zshrc"
    if "fish" in shell:
        return ".config/fish/config.fish"
    return ".bashrc"
