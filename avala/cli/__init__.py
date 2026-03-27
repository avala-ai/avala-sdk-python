"""Avala CLI — command-line interface for the Avala API."""

from __future__ import annotations

try:
    import click
except ImportError:
    raise SystemExit("CLI dependencies not installed. Run: pip install avala[cli]")

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

from avala import Client
from avala.errors import AuthenticationError, AvalaError, RateLimitError


def _get_version() -> str:
    try:
        return _pkg_version("avala")
    except PackageNotFoundError:
        return "unknown"


class _CliExceptionHandler(click.Group):
    """Wraps invoke() to catch SDK exceptions and show friendly messages."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            super().invoke(ctx)
        except AuthenticationError as exc:
            raise click.ClickException(
                f"{exc.message}\n\n"
                "Set your API key with:\n"
                '  export AVALA_API_KEY="avk_..."\n'
                "  # or run: avala configure"
            ) from exc
        except RateLimitError as exc:
            hint = f" (retry after {exc.retry_after}s)" if exc.retry_after else ""
            raise click.ClickException(f"Rate limited{hint}: {exc.message}") from exc
        except AvalaError as exc:
            raise click.ClickException(f"API error ({exc.status_code}): {exc.message}") from exc
        except ValueError as exc:
            msg = str(exc)
            if "api key" in msg.lower():
                raise click.ClickException(
                    "No API key provided.\n\n"
                    "Set your API key with:\n"
                    '  export AVALA_API_KEY="avk_..."\n'
                    "  # or run: avala configure"
                ) from exc
            raise click.ClickException(msg) from exc
        except KeyError as exc:
            if exc.args and exc.args[0] == "client":
                raise click.ClickException(
                    "No API key provided.\n\n"
                    "Set your API key with:\n"
                    '  export AVALA_API_KEY="avk_..."\n'
                    "  # or run: avala configure"
                ) from exc
            raise


@click.group(cls=_CliExceptionHandler)
@click.version_option(version=_get_version(), prog_name="avala")
@click.option("--api-key", envvar="AVALA_API_KEY", default=None, help="Avala API key (or set AVALA_API_KEY)")
@click.option("--base-url", envvar="AVALA_BASE_URL", default=None, help="Avala API base URL (or set AVALA_BASE_URL)")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def main(ctx: click.Context, api_key: str | None, base_url: str | None, output: str) -> None:
    """Avala CLI — interact with the Avala API from your terminal."""
    ctx.ensure_object(dict)
    ctx.obj["output_format"] = output

    # Skip client creation for configure, completion, and shell-completion
    if ctx.invoked_subcommand in ("configure", "shell-completion") or ctx.resilient_parsing:
        return

    try:
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        ctx.obj["client"] = Client(**kwargs)
    except ValueError as exc:
        # Re-raise config errors (bad base URL) so _CliExceptionHandler shows a
        # friendly message. But swallow "no API key" errors so --help still works
        # without a key — that case is caught by the KeyError handler later.
        if "api key" not in str(exc).lower():
            raise
    except Exception:
        # Don't exit here — subcommand --help should still work without
        # an API key. Subcommands that need the client will fail with a
        # KeyError on ctx.obj["client"], which is a clearer error anyway.
        pass


# Import and register subcommands
from avala.cli.agents import agents  # noqa: E402
from avala.cli.auto_label import auto_label  # noqa: E402
from avala.cli.configure import configure  # noqa: E402
from avala.cli.consensus import consensus  # noqa: E402
from avala.cli.datasets import datasets  # noqa: E402
from avala.cli.exports import exports  # noqa: E402
from avala.cli.fleet import fleet  # noqa: E402
from avala.cli.inference_providers import inference_providers  # noqa: E402
from avala.cli.projects import projects  # noqa: E402
from avala.cli.quality_targets import quality_targets  # noqa: E402
from avala.cli.shell_completion import shell_completion  # noqa: E402
from avala.cli.status import status  # noqa: E402
from avala.cli.storage_configs import storage_configs  # noqa: E402
from avala.cli.tasks import tasks  # noqa: E402
from avala.cli.webhooks import webhooks  # noqa: E402

main.add_command(agents)
main.add_command(auto_label)
main.add_command(configure)
main.add_command(consensus)
main.add_command(datasets)
main.add_command(exports)
main.add_command(fleet)
main.add_command(inference_providers)
main.add_command(projects)
main.add_command(quality_targets)
main.add_command(shell_completion)
main.add_command(status)
main.add_command(storage_configs)
main.add_command(tasks)
main.add_command(webhooks)
