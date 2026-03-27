"""CLI command for organization status dashboard."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from avala.cli._output import _get_output_format


def _safe_call(fn, *args, **kwargs):
    """Call an SDK method, returning None on any error."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show organization overview -- datasets, projects, exports, fleet health."""
    client = ctx.obj["client"]
    output_json = _get_output_format() == "json"

    # 1. Organization info
    orgs_page = _safe_call(client.organizations.list, limit=1)
    org = orgs_page.items[0] if orgs_page and orgs_page.items else None
    org_name = org.name if org else "Unknown Organization"

    # 2. Datasets (fetch 5 recent, report "5+" if page is full)
    datasets_page = _safe_call(client.datasets.list, limit=5)
    datasets_items = datasets_page.items if datasets_page else []

    # 3. Projects
    projects_page = _safe_call(client.projects.list, limit=5)
    projects_items = projects_page.items if projects_page else []

    # 4. Exports
    exports_page = _safe_call(client.exports.list, limit=5)
    exports_items = exports_page.items if exports_page else []
    pending_exports = [e for e in exports_items if e.status in ("pending", "processing")]

    # 5. Fleet (optional -- may not be enabled)
    fleet_page = _safe_call(client.fleet.devices.list, limit=100)
    fleet_devices = fleet_page.items if fleet_page else []
    fleet_available = fleet_page is not None

    if output_json:
        data: dict = {
            "organization": org_name,
            "datasets": {
                "showing": len(datasets_items),
                "hasMore": datasets_page.has_more if datasets_page else False,
                "latest": [{"uid": d.uid, "name": d.name} for d in datasets_items],
            },
            "projects": {
                "showing": len(projects_items),
                "hasMore": projects_page.has_more if projects_page else False,
                "latest": [{"uid": p.uid, "name": p.name, "status": p.status or "unknown"} for p in projects_items],
            },
            "exports": {
                "showing": len(exports_items),
                "pending": [{"uid": e.uid, "status": e.status} for e in pending_exports],
            },
        }
        if fleet_available:
            online = sum(1 for d in fleet_devices if d.status == "online")
            offline = len(fleet_devices) - online
            data["fleet"] = {"total": len(fleet_devices), "online": online, "offline": offline}
        click.echo(json.dumps(data, indent=2, default=str))
        return

    console = Console()

    # Header
    console.print()
    console.print(Panel(Text(org_name, style="bold cyan", justify="center"), title="Avala Status"))
    console.print()

    # Datasets section
    ds_table = Table(title="Datasets", expand=True)
    ds_table.add_column("Name")
    ds_table.add_column("UID", style="dim")
    if datasets_items:
        for d in datasets_items:
            ds_table.add_row(d.name, d.uid)
        if datasets_page and datasets_page.has_more:
            ds_table.add_row("[dim]... more datasets available[/dim]", "")
    else:
        ds_table.add_row("No datasets found", "")
    console.print(ds_table)
    console.print()

    # Projects section
    pj_table = Table(title="Projects", expand=True)
    pj_table.add_column("Name")
    pj_table.add_column("Status")
    pj_table.add_column("UID", style="dim")
    if projects_items:
        for p in projects_items:
            pj_table.add_row(p.name, p.status or "unknown", p.uid)
        if projects_page and projects_page.has_more:
            pj_table.add_row("[dim]... more projects available[/dim]", "", "")
    else:
        pj_table.add_row("No projects found", "", "")
    console.print(pj_table)
    console.print()

    # Exports section
    if pending_exports:
        ex_table = Table(title="Pending Exports", expand=True)
        ex_table.add_column("UID")
        ex_table.add_column("Status")
        for e in pending_exports:
            ex_table.add_row(e.uid, e.status or "unknown")
        console.print(ex_table)
    else:
        console.print("[green]No pending exports[/green]")
    console.print()

    # Fleet section (skip if not enabled)
    if fleet_available:
        online = sum(1 for d in fleet_devices if d.status == "online")
        offline = len(fleet_devices) - online
        fleet_text = f"[green]{online} online[/green]  [red]{offline} offline[/red]  ({len(fleet_devices)} total)"
        console.print(Panel(fleet_text, title="Fleet"))
        console.print()
