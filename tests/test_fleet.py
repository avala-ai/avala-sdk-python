"""Tests for the Fleet resource (devices, recordings, events, rules, alerts, uploads)."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client
from avala.resources.fleet import uploads as uploads_mod

BASE_URL = "https://api.avala.ai/api/v1"


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


@respx.mock
def test_list_devices_with_filters():
    """Devices.list() forwards status/type filters and returns a CursorPage of Device."""
    route = respx.get(f"{BASE_URL}/fleet/devices/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "device-001",
                        "name": "Robot Arm 1",
                        "type": "robot_arm",
                        "status": "online",
                        "tags": ["lab", "arm"],
                        "firmware_version": "1.2.3",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.devices.list(status="online", type="robot_arm")
    assert len(page.items) == 1
    assert page.items[0].uid == "device-001"
    assert page.items[0].name == "Robot Arm 1"
    assert page.items[0].status == "online"
    assert page.has_more is False
    request = route.calls.last.request
    assert request.url.params["status"] == "online"
    assert request.url.params["type"] == "robot_arm"
    client.close()


@respx.mock
def test_register_device():
    """Devices.register() sends correct payload and returns a Device."""
    route = respx.post(f"{BASE_URL}/fleet/devices/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "device-new-001",
                "name": "Robot Arm 2",
                "type": "robot_arm",
                "status": "provisioning",
                "firmware_version": "2.0.0",
                "tags": ["lab"],
                "device_token": "tok-secret",
            },
        )
    )
    client = Client(api_key="test-key")
    device = client.fleet.devices.register(
        name="Robot Arm 2",
        type="robot_arm",
        firmware_version="2.0.0",
        tags=["lab"],
        metadata={"location": "bay-3"},
    )
    assert device.uid == "device-new-001"
    assert device.device_token == "tok-secret"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "Robot Arm 2"
    assert request_body["type"] == "robot_arm"
    assert request_body["firmware_version"] == "2.0.0"
    assert request_body["tags"] == ["lab"]
    assert request_body["metadata"] == {"location": "bay-3"}
    client.close()


@respx.mock
def test_update_device():
    """Devices.update() sends a PATCH with only the provided fields."""
    uid = "device-001"
    route = respx.patch(f"{BASE_URL}/fleet/devices/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Renamed Arm",
                "type": "robot_arm",
                "status": "maintenance",
                "tags": ["lab", "retired"],
            },
        )
    )
    client = Client(api_key="test-key")
    device = client.fleet.devices.update(
        uid,
        name="Renamed Arm",
        status="maintenance",
        tags=["lab", "retired"],
        metadata={"note": "service"},
    )
    assert device.uid == uid
    assert device.status == "maintenance"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "Renamed Arm",
        "status": "maintenance",
        "tags": ["lab", "retired"],
        "metadata": {"note": "service"},
    }
    client.close()


@respx.mock
def test_delete_device():
    """Devices.delete() issues a DELETE and returns None."""
    uid = "device-001"
    route = respx.delete(f"{BASE_URL}/fleet/devices/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    assert client.fleet.devices.delete(uid) is None
    assert route.called
    assert route.calls.last.request.method == "DELETE"
    client.close()


@respx.mock
def test_get_device():
    """Devices.get() returns a single Device by uid."""
    uid = "device-001"
    respx.get(f"{BASE_URL}/fleet/devices/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Robot Arm 1",
                "type": "robot_arm",
                "status": "online",
            },
        )
    )
    client = Client(api_key="test-key")
    device = client.fleet.devices.get(uid)
    assert device.uid == uid
    assert device.name == "Robot Arm 1"
    client.close()


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


@respx.mock
def test_list_recordings_with_filters():
    """Recordings.list() maps device_id -> device and forwards status filter."""
    route = respx.get(f"{BASE_URL}/fleet/recordings/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "rec-001",
                        "device": "device-001",
                        "status": "uploaded",
                        "duration_seconds": 42.5,
                        "size_bytes": 1024,
                        "topic_count": 3,
                        "tags": ["session-a"],
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.recordings.list(device_id="device-001", status="uploaded")
    assert len(page.items) == 1
    assert page.items[0].uid == "rec-001"
    assert page.items[0].device == "device-001"
    assert page.items[0].topic_count == 3
    assert page.has_more is False
    request = route.calls.last.request
    assert request.url.params["device"] == "device-001"
    assert request.url.params["status"] == "uploaded"
    client.close()


@respx.mock
def test_get_recording():
    """Recordings.get() returns a single Recording by uid."""
    uid = "rec-001"
    respx.get(f"{BASE_URL}/fleet/recordings/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "device": "device-001",
                "status": "uploaded",
                "duration_seconds": 42.5,
            },
        )
    )
    client = Client(api_key="test-key")
    recording = client.fleet.recordings.get(uid)
    assert recording.uid == uid
    assert recording.duration_seconds == 42.5
    client.close()


@respx.mock
def test_update_recording():
    """Recordings.update() sends a PATCH with only the provided fields."""
    uid = "rec-001"
    route = respx.patch(f"{BASE_URL}/fleet/recordings/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "device": "device-001",
                "status": "archived",
                "tags": ["session-a", "reviewed"],
            },
        )
    )
    client = Client(api_key="test-key")
    recording = client.fleet.recordings.update(uid, status="archived", tags=["session-a", "reviewed"])
    assert recording.uid == uid
    assert recording.status == "archived"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"status": "archived", "tags": ["session-a", "reviewed"]}
    client.close()


@respx.mock
def test_delete_recording():
    """Recordings.delete() issues a DELETE and returns None."""
    uid = "rec-001"
    route = respx.delete(f"{BASE_URL}/fleet/recordings/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    assert client.fleet.recordings.delete(uid) is None
    assert route.called
    assert route.calls.last.request.method == "DELETE"
    client.close()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@respx.mock
def test_create_event():
    """Events.create() sends correct payload and returns a FleetEvent."""
    route = respx.post(f"{BASE_URL}/fleet/events/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "event-001",
                "recording": "rec-001",
                "device": "device-001",
                "type": "anomaly",
                "label": "collision",
                "severity": "high",
                "timestamp": "2026-01-01T00:00:00Z",
                "duration_ms": 500,
            },
        )
    )
    client = Client(api_key="test-key")
    event = client.fleet.events.create(
        recording="rec-001",
        device="device-001",
        label="collision",
        type="anomaly",
        timestamp="2026-01-01T00:00:00Z",
        severity="high",
        duration_ms=500,
        tags=["safety"],
    )
    assert event.uid == "event-001"
    assert event.label == "collision"
    assert event.severity == "high"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["recording"] == "rec-001"
    assert request_body["device"] == "device-001"
    assert request_body["label"] == "collision"
    assert request_body["type"] == "anomaly"
    assert request_body["timestamp"] == "2026-01-01T00:00:00Z"
    assert request_body["severity"] == "high"
    assert request_body["duration_ms"] == 500
    assert request_body["tags"] == ["safety"]
    client.close()


@respx.mock
def test_list_events_with_filters():
    """Events.list() maps recording_id/device_id and forwards type/severity filters."""
    route = respx.get(f"{BASE_URL}/fleet/events/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "event-001",
                        "recording": "rec-001",
                        "device": "device-001",
                        "type": "anomaly",
                        "label": "collision",
                        "severity": "high",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.events.list(
        recording_id="rec-001",
        device_id="device-001",
        type="anomaly",
        severity="high",
    )
    assert len(page.items) == 1
    assert page.items[0].uid == "event-001"
    assert page.items[0].label == "collision"
    assert page.has_more is False
    params = route.calls.last.request.url.params
    assert params["recording"] == "rec-001"
    assert params["device"] == "device-001"
    assert params["type"] == "anomaly"
    assert params["severity"] == "high"
    client.close()


@respx.mock
def test_get_event():
    """Events.get() returns a single FleetEvent by uid."""
    uid = "event-001"
    respx.get(f"{BASE_URL}/fleet/events/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "recording": "rec-001",
                "device": "device-001",
                "type": "anomaly",
                "label": "collision",
            },
        )
    )
    client = Client(api_key="test-key")
    event = client.fleet.events.get(uid)
    assert event.uid == uid
    assert event.label == "collision"
    client.close()


@respx.mock
def test_delete_event():
    """Events.delete() issues a DELETE and returns None."""
    uid = "event-001"
    route = respx.delete(f"{BASE_URL}/fleet/events/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    assert client.fleet.events.delete(uid) is None
    assert route.called
    assert route.calls.last.request.method == "DELETE"
    client.close()


@respx.mock
def test_create_batch_events():
    """Events.create_batch() POSTs {events: [...]} and returns the created count."""
    route = respx.post(f"{BASE_URL}/fleet/events/batch/").mock(return_value=httpx.Response(201, json={"created": 2}))
    client = Client(api_key="test-key")
    events = [
        {
            "recording": "rec-001",
            "device": "device-001",
            "label": "collision",
            "type": "anomaly",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "recording": "rec-001",
            "device": "device-001",
            "label": "drift",
            "type": "anomaly",
            "timestamp": "2026-01-01T00:01:00Z",
            "severity": "low",
        },
    ]
    created = client.fleet.events.create_batch(events=events)
    assert created == 2
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"events": events}
    client.close()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@respx.mock
def test_list_rules():
    """Rules.list() forwards the enabled filter and returns a CursorPage of Rule."""
    route = respx.get(f"{BASE_URL}/fleet/rules/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "rule-001",
                        "name": "High temp alert",
                        "description": "Fires when temperature exceeds threshold",
                        "enabled": True,
                        "condition": {"field": "temp", "op": "gt", "value": 80},
                        "actions": [{"type": "alert"}],
                        "hit_count": 5,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.rules.list(enabled=True)
    assert len(page.items) == 1
    assert page.items[0].uid == "rule-001"
    assert page.items[0].name == "High temp alert"
    assert page.items[0].enabled is True
    assert page.items[0].hit_count == 5
    assert page.has_more is False
    assert route.calls.last.request.url.params["enabled"] == "true"
    client.close()


@respx.mock
def test_create_rule():
    """Rules.create() sends the required + optional fields and returns a Rule."""
    route = respx.post(f"{BASE_URL}/fleet/rules/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "rule-001",
                "name": "High temp alert",
                "description": "Fires when temperature exceeds threshold",
                "enabled": True,
                "condition": {"field": "temp", "op": "gt", "value": 80},
                "actions": [{"type": "alert"}],
            },
        )
    )
    client = Client(api_key="test-key")
    rule = client.fleet.rules.create(
        name="High temp alert",
        condition={"field": "temp", "op": "gt", "value": 80},
        actions=[{"type": "alert"}],
        description="Fires when temperature exceeds threshold",
        enabled=True,
        scope={"device_type": "robot_arm"},
    )
    assert rule.uid == "rule-001"
    assert rule.name == "High temp alert"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "High temp alert",
        "condition": {"field": "temp", "op": "gt", "value": 80},
        "actions": [{"type": "alert"}],
        "description": "Fires when temperature exceeds threshold",
        "enabled": True,
        "scope": {"device_type": "robot_arm"},
    }
    client.close()


@respx.mock
def test_get_rule():
    """Rules.get() returns a single Rule by uid."""
    uid = "rule-001"
    respx.get(f"{BASE_URL}/fleet/rules/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={"uid": uid, "name": "High temp alert", "enabled": True},
        )
    )
    client = Client(api_key="test-key")
    rule = client.fleet.rules.get(uid)
    assert rule.uid == uid
    assert rule.name == "High temp alert"
    client.close()


@respx.mock
def test_update_rule():
    """Rules.update() sends a PATCH with only the provided fields."""
    uid = "rule-001"
    route = respx.patch(f"{BASE_URL}/fleet/rules/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={"uid": uid, "name": "High temp alert", "enabled": False},
        )
    )
    client = Client(api_key="test-key")
    rule = client.fleet.rules.update(uid, enabled=False, condition={"field": "temp", "op": "gt", "value": 90})
    assert rule.uid == uid
    assert rule.enabled is False
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"enabled": False, "condition": {"field": "temp", "op": "gt", "value": 90}}
    client.close()


@respx.mock
def test_delete_rule():
    """Rules.delete() issues a DELETE and returns None."""
    uid = "rule-001"
    route = respx.delete(f"{BASE_URL}/fleet/rules/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    assert client.fleet.rules.delete(uid) is None
    assert route.called
    assert route.calls.last.request.method == "DELETE"
    client.close()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@respx.mock
def test_list_alerts_with_filters():
    """Alerts.list() forwards status/severity filters and returns a CursorPage of Alert."""
    route = respx.get(f"{BASE_URL}/fleet/alerts/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "alert-001",
                        "rule": "rule-001",
                        "device": "device-001",
                        "severity": "critical",
                        "status": "open",
                        "message": "Temperature exceeded 80C",
                        "triggered_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.alerts.list(status="open", severity="critical")
    assert len(page.items) == 1
    assert page.items[0].uid == "alert-001"
    assert page.items[0].severity == "critical"
    assert page.items[0].status == "open"
    assert page.has_more is False
    request = route.calls.last.request
    assert request.url.params["status"] == "open"
    assert request.url.params["severity"] == "critical"
    client.close()


@respx.mock
def test_get_alert():
    """Alerts.get() returns a single Alert by uid."""
    uid = "alert-001"
    respx.get(f"{BASE_URL}/fleet/alerts/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "rule": "rule-001",
                "device": "device-001",
                "severity": "critical",
                "status": "open",
                "message": "Temperature exceeded 80C",
            },
        )
    )
    client = Client(api_key="test-key")
    alert = client.fleet.alerts.get(uid)
    assert alert.uid == uid
    assert alert.status == "open"
    client.close()


@respx.mock
def test_acknowledge_alert():
    """Alerts.acknowledge() POSTs to /acknowledge/ and returns the updated Alert."""
    uid = "alert-001"
    route = respx.post(f"{BASE_URL}/fleet/alerts/{uid}/acknowledge/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "status": "acknowledged",
                "acknowledged_by": "user-1",
                "acknowledged_at": "2026-01-01T01:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    alert = client.fleet.alerts.acknowledge(uid)
    assert alert.uid == uid
    assert alert.status == "acknowledged"
    assert route.called
    assert route.calls.last.request.method == "POST"
    client.close()


@respx.mock
def test_resolve_alert():
    """Alerts.resolve() POSTs to /resolve/ with the resolution_note body."""
    uid = "alert-001"
    route = respx.post(f"{BASE_URL}/fleet/alerts/{uid}/resolve/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "status": "resolved",
                "resolution_note": "Cooled down",
                "resolved_at": "2026-01-01T02:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    alert = client.fleet.alerts.resolve(uid, resolution_note="Cooled down")
    assert alert.uid == uid
    assert alert.status == "resolved"
    assert alert.resolution_note == "Cooled down"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"resolution_note": "Cooled down"}
    client.close()


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------


@respx.mock
def test_list_alert_channels():
    """AlertChannels.list() returns a CursorPage of AlertChannel."""
    route = respx.get(f"{BASE_URL}/fleet/alert-channels/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "chan-001",
                        "name": "Ops Slack",
                        "type": "slack",
                        "config": {"webhook_url": "https://hooks.slack.com/x"},
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.fleet.alert_channels.list(limit=10)
    assert len(page.items) == 1
    assert page.items[0].uid == "chan-001"
    assert page.items[0].type == "slack"
    assert page.has_more is False
    assert route.calls.last.request.url.params["limit"] == "10"
    client.close()


@respx.mock
def test_create_alert_channel():
    """AlertChannels.create() sends name/type/config and returns an AlertChannel."""
    route = respx.post(f"{BASE_URL}/fleet/alert-channels/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "chan-001",
                "name": "Ops Slack",
                "type": "slack",
                "config": {"webhook_url": "https://hooks.slack.com/x"},
            },
        )
    )
    client = Client(api_key="test-key")
    channel = client.fleet.alert_channels.create(
        name="Ops Slack",
        type="slack",
        config={"webhook_url": "https://hooks.slack.com/x"},
    )
    assert channel.uid == "chan-001"
    assert channel.type == "slack"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "Ops Slack",
        "type": "slack",
        "config": {"webhook_url": "https://hooks.slack.com/x"},
    }
    client.close()


@respx.mock
def test_delete_alert_channel():
    """AlertChannels.delete() issues a DELETE and returns None."""
    uid = "chan-001"
    route = respx.delete(f"{BASE_URL}/fleet/alert-channels/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    assert client.fleet.alert_channels.delete(uid) is None
    assert route.called
    assert route.calls.last.request.method == "DELETE"
    client.close()


@respx.mock
def test_test_alert_channel():
    """AlertChannels.test() POSTs to /test/ and returns the raw response dict."""
    uid = "chan-001"
    route = respx.post(f"{BASE_URL}/fleet/alert-channels/{uid}/test/").mock(
        return_value=httpx.Response(200, json={"ok": True, "delivered": True})
    )
    client = Client(api_key="test-key")
    result = client.fleet.alert_channels.test(uid)
    assert result == {"ok": True, "delivered": True}
    assert route.called
    assert route.calls.last.request.method == "POST"
    client.close()


# ---------------------------------------------------------------------------
# Uploads (FleetUploadManager)
# ---------------------------------------------------------------------------


@respx.mock
def test_init_upload():
    """init_upload() POSTs the file manifest and returns an UploadSession."""
    rec = "rec-001"
    route = respx.post(f"{BASE_URL}/fleet/recordings/{rec}/upload/init/").mock(
        return_value=httpx.Response(
            201,
            json={"uid": "session-1", "total_files": 2, "total_bytes": 32, "status": "active"},
        )
    )
    client = Client(api_key="test-key")
    files = [{"path": "a.txt", "size_bytes": 16}, {"path": "b.txt", "size_bytes": 16}]
    session = client.fleet.uploads.init_upload(rec, files, storage_config_uid="sc-1")
    assert session.uid == "session-1"
    assert session.total_files == 2
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"files": files, "storage_config_uid": "sc-1"}
    client.close()


@respx.mock
def test_get_upload_urls():
    """get_upload_urls() POSTs session/paths/ttl and returns presigned URL entries."""
    rec = "rec-001"
    route = respx.post(f"{BASE_URL}/fleet/recordings/{rec}/upload/urls/").mock(
        return_value=httpx.Response(
            200,
            json={
                "urls": [
                    {
                        "path": "a.txt",
                        "put_url": "https://s3.us-west-2.amazonaws.com/bucket/a.txt",
                        "s3_key": "bucket/a.txt",
                        "headers": {"Content-Type": "application/octet-stream"},
                    }
                ]
            },
        )
    )
    client = Client(api_key="test-key")
    resp = client.fleet.uploads.get_upload_urls(rec, "session-1", ["a.txt"], ttl_seconds=600)
    assert len(resp.urls) == 1
    assert resp.urls[0].path == "a.txt"
    assert resp.urls[0].put_url.endswith("/a.txt")
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"session_uid": "session-1", "file_paths": ["a.txt"], "ttl_seconds": 600}
    client.close()


@respx.mock
def test_confirm_upload():
    """confirm_upload() POSTs the confirmed files and returns the raw response."""
    rec = "rec-001"
    route = respx.post(f"{BASE_URL}/fleet/recordings/{rec}/upload/confirm/").mock(
        return_value=httpx.Response(200, json={"confirmed": 1, "total_confirmed": 1, "total_files": 2})
    )
    client = Client(api_key="test-key")
    files = [{"path": "a.txt", "etag": "abc", "size_bytes": 16}]
    result = client.fleet.uploads.confirm_upload(rec, "session-1", files)
    assert result == {"confirmed": 1, "total_confirmed": 1, "total_files": 2}
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"session_uid": "session-1", "files": files}
    client.close()


@respx.mock
def test_finalize_upload():
    """finalize_upload() POSTs the session_uid and returns the raw response."""
    rec = "rec-001"
    route = respx.post(f"{BASE_URL}/fleet/recordings/{rec}/upload/finalize/").mock(
        return_value=httpx.Response(200, json={"status": "finalized"})
    )
    client = Client(api_key="test-key")
    result = client.fleet.uploads.finalize_upload(rec, "session-1")
    assert result == {"status": "finalized"}
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"session_uid": "session-1"}
    client.close()


@respx.mock
def test_get_upload_status():
    """get_upload_status() GETs the status endpoint and returns an UploadStatusResponse."""
    rec = "rec-001"
    respx.get(f"{BASE_URL}/fleet/recordings/{rec}/upload/status/").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_uid": "session-1",
                "status": "active",
                "total_files": 2,
                "confirmed_files": 1,
                "pending_paths": ["b.txt"],
            },
        )
    )
    client = Client(api_key="test-key")
    status = client.fleet.uploads.get_upload_status(rec)
    assert status.session_uid == "session-1"
    assert status.confirmed_files == 1
    assert status.pending_paths == ["b.txt"]
    client.close()


@respx.mock
def test_upload_recording_end_to_end(tmp_path, monkeypatch):
    """upload_recording() walks a dir, presigns, PUTs to S3, confirms, finalizes."""
    # Keep crash-recovery state out of the real home directory.
    monkeypatch.setattr(uploads_mod, "_STATE_DIR", tmp_path / "state")

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_bytes(b"x" * 16)
    (source / "b.txt").write_bytes(b"y" * 16)

    rec = "rec-upload-1"
    base = f"{BASE_URL}/fleet/recordings/{rec}"

    respx.post(f"{base}/upload/init/").mock(
        return_value=httpx.Response(
            201,
            json={"uid": "session-1", "total_files": 2, "total_bytes": 32, "status": "active"},
        )
    )
    respx.post(f"{base}/upload/urls/").mock(
        return_value=httpx.Response(
            200,
            json={
                "urls": [
                    {
                        "path": "a.txt",
                        "put_url": "https://s3.us-west-2.amazonaws.com/bucket/a.txt",
                        "s3_key": "bucket/a.txt",
                        "headers": {},
                    },
                    {
                        "path": "b.txt",
                        "put_url": "https://s3.us-west-2.amazonaws.com/bucket/b.txt",
                        "s3_key": "bucket/b.txt",
                        "headers": {},
                    },
                ]
            },
        )
    )
    respx.put("https://s3.us-west-2.amazonaws.com/bucket/a.txt").mock(
        return_value=httpx.Response(200, headers={"ETag": '"etag-a"'})
    )
    respx.put("https://s3.us-west-2.amazonaws.com/bucket/b.txt").mock(
        return_value=httpx.Response(200, headers={"ETag": '"etag-b"'})
    )
    confirm_route = respx.post(f"{base}/upload/confirm/").mock(
        return_value=httpx.Response(200, json={"confirmed": 2, "total_confirmed": 2, "total_files": 2})
    )
    finalize_route = respx.post(f"{base}/upload/finalize/").mock(
        return_value=httpx.Response(200, json={"status": "finalized"})
    )
    respx.get(f"{base}/upload/status/").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_uid": "session-1",
                "status": "completed",
                "total_files": 2,
                "confirmed_files": 2,
                "total_bytes": 32,
                "confirmed_bytes": 32,
                "pending_paths": [],
            },
        )
    )

    client = Client(api_key="test-key")
    progress_events = []
    session = client.fleet.uploads.upload_recording(
        rec,
        source,
        storage_config_uid="sc-1",
        on_progress=progress_events.append,
    )
    assert session.uid == "session-1"
    assert session.status == "completed"
    assert session.confirmed_files == 2
    assert confirm_route.called
    assert finalize_route.called
    # All files confirmed -> local state file cleaned up.
    assert not (tmp_path / "state" / f"{rec}.json").exists()
    assert progress_events and progress_events[-1].uploaded_files == 2
    client.close()
