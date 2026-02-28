"""CLI commands for fleet management."""

from __future__ import annotations

import json

import click

from avala.cli._output import print_detail, print_table


@click.group("fleet")
def fleet() -> None:
    """Fleet management commands."""


# ── Devices ──────────────────────────────────────────────────────────────────


@fleet.group("devices")
def devices() -> None:
    """Manage fleet devices."""


@devices.command("list")
@click.option("--status", default=None, help="Filter by status (online, offline, maintenance)")
@click.option("--type", "device_type", default=None, help="Filter by device type")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_devices(ctx: click.Context, status: str | None, device_type: str | None, limit: int | None) -> None:
    """List fleet devices."""
    client = ctx.obj["client"]
    page = client.fleet.devices.list(status=status, type=device_type, limit=limit)
    rows = [
        (d.uid, d.name, d.type or "—", d.status or "—", d.firmware_version or "—", str(d.last_seen_at or "—"))
        for d in page.items
    ]
    print_table("Devices", ["UID", "Name", "Type", "Status", "Firmware", "Last Seen"], rows)


@devices.command("get")
@click.argument("uid")
@click.pass_context
def get_device(ctx: click.Context, uid: str) -> None:
    """Get device details."""
    client = ctx.obj["client"]
    d = client.fleet.devices.get(uid)
    print_detail(
        f"Device: {d.name}",
        [
            ("UID", d.uid),
            ("Name", d.name),
            ("Type", d.type or "—"),
            ("Status", d.status or "—"),
            ("Firmware", d.firmware_version or "—"),
            ("Tags", ", ".join(d.tags) if d.tags else "—"),
            ("Last Seen", str(d.last_seen_at or "—")),
            ("Device Token", d.device_token or "—"),
            ("Created", str(d.created_at or "—")),
            ("Updated", str(d.updated_at or "—")),
        ],
    )


@devices.command("register")
@click.option("--name", required=True, help="Device name")
@click.option("--type", "device_type", required=True, help="Device type")
@click.option("--tags", default=None, help="Comma-separated tags")
@click.option("--firmware-version", default=None, help="Firmware version")
@click.pass_context
def register_device(
    ctx: click.Context,
    name: str,
    device_type: str,
    tags: str | None,
    firmware_version: str | None,
) -> None:
    """Register a new device."""
    client = ctx.obj["client"]
    kwargs: dict = {"name": name, "type": device_type}
    if firmware_version is not None:
        kwargs["firmware_version"] = firmware_version
    if tags is not None:
        kwargs["tags"] = [t.strip() for t in tags.split(",")]
    d = client.fleet.devices.register(**kwargs)
    click.echo(f"Device registered: {d.uid} ({d.name})")
    if d.device_token:
        click.echo(f"Device token: {d.device_token}")
        click.echo("Save this token — it may not be shown again.")


@devices.command("update")
@click.argument("uid")
@click.option("--status", default=None, help="Device status")
@click.option("--name", default=None, help="Device name")
@click.option("--tags", default=None, help="Comma-separated tags")
@click.pass_context
def update_device(ctx: click.Context, uid: str, status: str | None, name: str | None, tags: str | None) -> None:
    """Update a device."""
    client = ctx.obj["client"]
    kwargs: dict = {}
    if status is not None:
        kwargs["status"] = status
    if name is not None:
        kwargs["name"] = name
    if tags is not None:
        kwargs["tags"] = [t.strip() for t in tags.split(",")]
    d = client.fleet.devices.update(uid, **kwargs)
    click.echo(f"Device updated: {d.uid} ({d.name})")


@devices.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this device?")
@click.pass_context
def delete_device(ctx: click.Context, uid: str) -> None:
    """Delete a device."""
    client = ctx.obj["client"]
    client.fleet.devices.delete(uid)
    click.echo(f"Device {uid} deleted.")


# ── Recordings ───────────────────────────────────────────────────────────────


@fleet.group("recordings")
def recordings() -> None:
    """Manage fleet recordings."""


@recordings.command("list")
@click.option("--device", default=None, help="Filter by device UID")
@click.option("--status", default=None, help="Filter by status (uploading, processing, ready, error, archived)")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_recordings(ctx: click.Context, device: str | None, status: str | None, limit: int | None) -> None:
    """List fleet recordings."""
    client = ctx.obj["client"]
    page = client.fleet.recordings.list(device_id=device, status=status, limit=limit)
    rows = [
        (
            r.uid,
            r.device or "—",
            r.status or "—",
            str(r.duration_seconds or "—"),
            str(r.topic_count),
            str(r.created_at or "—"),
        )
        for r in page.items
    ]
    print_table("Recordings", ["UID", "Device", "Status", "Duration (s)", "Topics", "Created"], rows)


@recordings.command("get")
@click.argument("uid")
@click.pass_context
def get_recording(ctx: click.Context, uid: str) -> None:
    """Get recording details."""
    client = ctx.obj["client"]
    r = client.fleet.recordings.get(uid)
    print_detail(
        f"Recording: {r.uid}",
        [
            ("UID", r.uid),
            ("Device", r.device or "—"),
            ("Status", r.status or "—"),
            ("Duration (s)", str(r.duration_seconds or "—")),
            ("Size (bytes)", str(r.size_bytes or "—")),
            ("Topics", str(r.topic_count)),
            ("Tags", ", ".join(r.tags) if r.tags else "—"),
            ("Started", str(r.started_at or "—")),
            ("Ended", str(r.ended_at or "—")),
            ("Created", str(r.created_at or "—")),
            ("Updated", str(r.updated_at or "—")),
        ],
    )


# ── Events ───────────────────────────────────────────────────────────────────


@fleet.group("events")
def events() -> None:
    """Manage fleet events."""


@events.command("list")
@click.option("--device", default=None, help="Filter by device UID")
@click.option("--recording", default=None, help="Filter by recording UID")
@click.option("--type", "event_type", default=None, help="Filter by event type")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_events(
    ctx: click.Context, device: str | None, recording: str | None, event_type: str | None, limit: int | None
) -> None:
    """List fleet events."""
    client = ctx.obj["client"]
    page = client.fleet.events.list(device_id=device, recording_id=recording, type=event_type, limit=limit)
    rows = [
        (e.uid, e.type or "—", e.label or "—", e.severity or "—", str(e.timestamp or "—"), str(e.created_at or "—"))
        for e in page.items
    ]
    print_table("Events", ["UID", "Type", "Label", "Severity", "Timestamp", "Created"], rows)


@events.command("get")
@click.argument("uid")
@click.pass_context
def get_event(ctx: click.Context, uid: str) -> None:
    """Get event details."""
    client = ctx.obj["client"]
    e = client.fleet.events.get(uid)
    print_detail(
        f"Event: {e.uid}",
        [
            ("UID", e.uid),
            ("Recording", e.recording or "—"),
            ("Device", e.device or "—"),
            ("Type", e.type or "—"),
            ("Label", e.label or "—"),
            ("Description", e.description or "—"),
            ("Severity", e.severity or "—"),
            ("Timestamp", str(e.timestamp or "—")),
            ("Duration (ms)", str(e.duration_ms or "—")),
            ("Tags", ", ".join(e.tags) if e.tags else "—"),
            ("Created", str(e.created_at or "—")),
            ("Updated", str(e.updated_at or "—")),
        ],
    )


@events.command("create")
@click.option("--recording", required=True, help="Recording UID")
@click.option("--device", required=True, help="Device UID")
@click.option("--type", "event_type", required=True, help="Event type")
@click.option("--label", required=True, help="Event label")
@click.option("--timestamp", required=True, help="Event timestamp (ISO 8601)")
@click.option("--severity", default=None, help="Severity (info, warning, error, critical)")
@click.pass_context
def create_event(
    ctx: click.Context,
    recording: str,
    device: str,
    event_type: str,
    label: str,
    timestamp: str,
    severity: str | None,
) -> None:
    """Create a new event."""
    client = ctx.obj["client"]
    kwargs: dict = {
        "recording": recording,
        "device": device,
        "type": event_type,
        "label": label,
        "timestamp": timestamp,
    }
    if severity is not None:
        kwargs["severity"] = severity
    e = client.fleet.events.create(**kwargs)
    click.echo(f"Event created: {e.uid}")


# ── Rules ────────────────────────────────────────────────────────────────────


@fleet.group("rules")
def rules() -> None:
    """Manage fleet rules."""


@rules.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_rules(ctx: click.Context, limit: int | None) -> None:
    """List fleet rules."""
    client = ctx.obj["client"]
    page = client.fleet.rules.list(limit=limit)
    rows = [
        (r.uid, r.name, "Yes" if r.enabled else "No", str(r.hit_count), str(r.created_at or "—")) for r in page.items
    ]
    print_table("Rules", ["UID", "Name", "Enabled", "Hits", "Created"], rows)


@rules.command("get")
@click.argument("uid")
@click.pass_context
def get_rule(ctx: click.Context, uid: str) -> None:
    """Get rule details."""
    client = ctx.obj["client"]
    r = client.fleet.rules.get(uid)
    print_detail(
        f"Rule: {r.name}",
        [
            ("UID", r.uid),
            ("Name", r.name),
            ("Description", r.description or "—"),
            ("Enabled", "Yes" if r.enabled else "No"),
            ("Condition", json.dumps(r.condition) if r.condition else "—"),
            ("Actions", json.dumps(r.actions) if r.actions else "—"),
            ("Scope", json.dumps(r.scope) if r.scope else "—"),
            ("Hit Count", str(r.hit_count)),
            ("Last Hit", str(r.last_hit_at or "—")),
            ("Created", str(r.created_at or "—")),
            ("Updated", str(r.updated_at or "—")),
        ],
    )


@rules.command("create")
@click.option("--name", required=True, help="Rule name")
@click.option("--condition-file", required=True, type=click.Path(exists=True), help="JSON file with condition")
@click.option("--enabled/--disabled", default=True, help="Enable or disable the rule")
@click.pass_context
def create_rule(ctx: click.Context, name: str, condition_file: str, enabled: bool) -> None:
    """Create a new rule."""
    client = ctx.obj["client"]
    with open(condition_file) as f:
        data = json.load(f)
    condition = data.get("condition", data)
    actions = data.get("actions", [])
    kwargs: dict = {"name": name, "condition": condition, "actions": actions, "enabled": enabled}
    description = data.get("description")
    if description is not None:
        kwargs["description"] = description
    scope = data.get("scope")
    if scope is not None:
        kwargs["scope"] = scope
    r = client.fleet.rules.create(**kwargs)
    click.echo(f"Rule created: {r.uid} ({r.name})")


@rules.command("update")
@click.argument("uid")
@click.option("--name", default=None, help="Rule name")
@click.option("--enabled/--disabled", default=None, help="Enable or disable the rule")
@click.pass_context
def update_rule(ctx: click.Context, uid: str, name: str | None, enabled: bool | None) -> None:
    """Update a rule."""
    client = ctx.obj["client"]
    kwargs: dict = {}
    if name is not None:
        kwargs["name"] = name
    if enabled is not None:
        kwargs["enabled"] = enabled
    r = client.fleet.rules.update(uid, **kwargs)
    click.echo(f"Rule updated: {r.uid} ({r.name})")


@rules.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this rule?")
@click.pass_context
def delete_rule(ctx: click.Context, uid: str) -> None:
    """Delete a rule."""
    client = ctx.obj["client"]
    client.fleet.rules.delete(uid)
    click.echo(f"Rule {uid} deleted.")


# ── Alerts ───────────────────────────────────────────────────────────────────


@fleet.group("alerts")
def alerts() -> None:
    """Manage fleet alerts."""


@alerts.command("list")
@click.option("--status", default=None, help="Filter by status (open, acknowledged, resolved)")
@click.option("--severity", default=None, help="Filter by severity (info, warning, error, critical)")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_alerts(ctx: click.Context, status: str | None, severity: str | None, limit: int | None) -> None:
    """List fleet alerts."""
    client = ctx.obj["client"]
    page = client.fleet.alerts.list(status=status, severity=severity, limit=limit)
    rows = [
        (
            a.uid,
            a.severity or "—",
            a.status or "—",
            a.message or "—",
            str(a.triggered_at or "—"),
        )
        for a in page.items
    ]
    print_table("Alerts", ["UID", "Severity", "Status", "Message", "Triggered"], rows)


@alerts.command("get")
@click.argument("uid")
@click.pass_context
def get_alert(ctx: click.Context, uid: str) -> None:
    """Get alert details."""
    client = ctx.obj["client"]
    a = client.fleet.alerts.get(uid)
    print_detail(
        f"Alert: {a.uid}",
        [
            ("UID", a.uid),
            ("Rule", a.rule or "—"),
            ("Device", a.device or "—"),
            ("Recording", a.recording or "—"),
            ("Severity", a.severity or "—"),
            ("Status", a.status or "—"),
            ("Message", a.message or "—"),
            ("Triggered", str(a.triggered_at or "—")),
            ("Acknowledged", str(a.acknowledged_at or "—")),
            ("Acknowledged By", a.acknowledged_by or "—"),
            ("Resolved", str(a.resolved_at or "—")),
            ("Resolution Note", a.resolution_note or "—"),
            ("Created", str(a.created_at or "—")),
            ("Updated", str(a.updated_at or "—")),
        ],
    )


@alerts.command("acknowledge")
@click.argument("uid")
@click.pass_context
def acknowledge_alert(ctx: click.Context, uid: str) -> None:
    """Acknowledge an alert."""
    client = ctx.obj["client"]
    a = client.fleet.alerts.acknowledge(uid)
    click.echo(f"Alert {a.uid} acknowledged.")


@alerts.command("resolve")
@click.argument("uid")
@click.option("--note", default=None, help="Resolution note")
@click.pass_context
def resolve_alert(ctx: click.Context, uid: str, note: str | None) -> None:
    """Resolve an alert."""
    client = ctx.obj["client"]
    kwargs: dict = {}
    if note is not None:
        kwargs["resolution_note"] = note
    a = client.fleet.alerts.resolve(uid, **kwargs)
    click.echo(f"Alert {a.uid} resolved.")
