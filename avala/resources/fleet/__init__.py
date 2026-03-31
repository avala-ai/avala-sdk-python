"""Fleet management resources."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avala._async_http import AsyncHTTPTransport
    from avala._http import SyncHTTPTransport

from avala.resources.fleet.alerts import AsyncFleetAlertChannels, AsyncFleetAlerts, FleetAlertChannels, FleetAlerts
from avala.resources.fleet.devices import AsyncFleetDevices, FleetDevices
from avala.resources.fleet.events import AsyncFleetEvents, FleetEvents
from avala.resources.fleet.recordings import AsyncFleetRecordings, FleetRecordings
from avala.resources.fleet.rules import AsyncFleetRules, FleetRules
from avala.resources.fleet.uploads import FleetUploadManager

__all__ = [
    "AsyncFleet",
    "Fleet",
]


class Fleet:
    def __init__(self, transport: SyncHTTPTransport) -> None:
        self.devices = FleetDevices(transport)
        self.recordings = FleetRecordings(transport)
        self.events = FleetEvents(transport)
        self.rules = FleetRules(transport)
        self.alerts = FleetAlerts(transport)
        self.alert_channels = FleetAlertChannels(transport)
        self.uploads = FleetUploadManager(transport)


class AsyncFleet:
    def __init__(self, transport: AsyncHTTPTransport) -> None:
        self.devices = AsyncFleetDevices(transport)
        self.recordings = AsyncFleetRecordings(transport)
        self.events = AsyncFleetEvents(transport)
        self.rules = AsyncFleetRules(transport)
        self.alerts = AsyncFleetAlerts(transport)
        self.alert_channels = AsyncFleetAlertChannels(transport)
