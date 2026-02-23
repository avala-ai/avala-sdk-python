"""CLI commands for agents."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group("agents")
def agents() -> None:
    """Manage agents."""


@agents.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_agents(ctx: click.Context, limit: int | None) -> None:
    """List agents."""
    client = ctx.obj["client"]
    page = client.agents.list(limit=limit)
    rows = [
        (a.uid, a.name, "Yes" if a.is_active else "No", ", ".join(a.events), str(a.created_at or "—"))
        for a in page.items
    ]
    print_table("Agents", ["UID", "Name", "Active", "Events", "Created"], rows)


@agents.command("get")
@click.argument("uid")
@click.pass_context
def get_agent(ctx: click.Context, uid: str) -> None:
    """Get an agent by UID."""
    client = ctx.obj["client"]
    a = client.agents.get(uid)
    print_detail(
        f"Agent: {a.name}",
        [
            ("UID", a.uid),
            ("Name", a.name),
            ("Description", a.description or "—"),
            ("Events", ", ".join(a.events)),
            ("Callback URL", a.callback_url or "—"),
            ("Active", "Yes" if a.is_active else "No"),
            ("Project", a.project or "—"),
            ("Task Types", ", ".join(a.task_types) if a.task_types else "—"),
            ("Created", str(a.created_at or "—")),
            ("Updated", str(a.updated_at or "—")),
        ],
    )


@agents.command("create")
@click.option("--name", required=True, help="Agent name")
@click.option("--events", default=None, help="Comma-separated list of event types")
@click.option("--callback-url", default=None, help="Webhook callback URL (HTTPS)")
@click.option("--description", default=None, help="Agent description")
@click.option("--project", default=None, help="Project UID to scope the agent to")
@click.option("--task-types", default=None, help="Comma-separated list of task types")
@click.pass_context
def create_agent(
    ctx: click.Context,
    name: str,
    events: str | None,
    callback_url: str | None,
    description: str | None,
    project: str | None,
    task_types: str | None,
) -> None:
    """Create a new agent."""
    client = ctx.obj["client"]
    kwargs: dict = {"name": name}
    if events is not None:
        kwargs["events"] = [e.strip() for e in events.split(",")]
    if callback_url is not None:
        kwargs["callback_url"] = callback_url
    if description is not None:
        kwargs["description"] = description
    if project is not None:
        kwargs["project"] = project
    if task_types is not None:
        kwargs["task_types"] = [t.strip() for t in task_types.split(",")]
    a = client.agents.create(**kwargs)
    click.echo(f"Agent created: {a.uid} ({a.name})")
    if a.secret:
        click.echo(f"Secret: {a.secret}")
        click.echo("Save this secret — it will not be shown again.")


@agents.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this agent?")
@click.pass_context
def delete_agent(ctx: click.Context, uid: str) -> None:
    """Delete an agent."""
    client = ctx.obj["client"]
    client.agents.delete(uid)
    click.echo(f"Agent {uid} deleted.")


@agents.command("executions")
@click.argument("uid")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_executions(ctx: click.Context, uid: str, limit: int | None) -> None:
    """List executions for an agent."""
    client = ctx.obj["client"]
    page = client.agents.list_executions(uid, limit=limit)
    rows = [(e.uid, e.event_type, e.status or "—", e.action or "—", str(e.created_at or "—")) for e in page.items]
    print_table("Agent Executions", ["UID", "Event", "Status", "Action", "Created"], rows)


@agents.command("test")
@click.argument("uid")
@click.pass_context
def test_agent(ctx: click.Context, uid: str) -> None:
    """Test an agent."""
    client = ctx.obj["client"]
    result = client.agents.test(uid)
    if result.get("success"):
        click.echo("Agent test passed.")
    else:
        click.echo(f"Agent test failed: {result.get('message', 'unknown error')}")
