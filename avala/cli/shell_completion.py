"""Shell completion setup for the Avala CLI."""

from __future__ import annotations

import os

import click


_COMPLETIONS = {
    "bash": 'eval "$(_AVALA_COMPLETE=bash_source avala)"',
    "zsh": 'eval "$(_AVALA_COMPLETE=zsh_source avala)"',
    "fish": "_AVALA_COMPLETE=fish_source avala | source",
}


@click.command("shell-completion")
@click.argument("shell", required=False, default=None, type=click.Choice(["bash", "zsh", "fish"]))
def shell_completion(shell: str | None) -> None:
    """Print shell completion setup instructions.

    Run the output to enable tab-completion for all avala commands.

    \b
    Examples:
      avala shell-completion bash >> ~/.bashrc
      avala shell-completion zsh >> ~/.zshrc
      avala shell-completion fish > ~/.config/fish/completions/avala.fish
    """
    if shell is None:
        shell = _detect_shell()

    click.echo(_COMPLETIONS[shell])


def _detect_shell() -> str:
    shell_path = os.environ.get("SHELL", "")
    if "zsh" in shell_path:
        return "zsh"
    if "fish" in shell_path:
        return "fish"
    return "bash"
