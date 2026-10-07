"""Torch-free LeRobot v3.0 reader and writer.

``LeRobotV3Dataset`` reads, ``LeRobotV3Writer`` writes, and :mod:`avala.converters.lerobot_v3.mcap`
bridges to Avala MCAP episodes (``lerobot_to_mcap``). Nothing here imports torch or the
``lerobot`` library; the dependencies are pyarrow, numpy, pillow and the mcap packages
(``avala[lerobot-core]``), plus PyAV for video features (``avala[lerobot-core-video]``).

The on-disk layout, and where each piece of it was verified, is documented in
:mod:`avala.converters.lerobot_v3._layout`.

Names are resolved lazily so ``import avala.converters.lerobot_v3`` stays free.
"""

from __future__ import annotations

from typing import Any

from avala.converters.lerobot_v3._layout import CODEBASE_VERSION

__all__ = [
    "CODEBASE_VERSION",
    "CONTROL_SOURCE_TOPIC",
    "CONTROL_SOURCE_VALUES",
    "DEFAULT_CONTROL_SOURCE_COLUMN",
    "LeRobotV3Dataset",
    "LeRobotV3Writer",
    "UnsupportedLeRobotVersion",
    "lerobot_to_mcap",
]

_LAZY = {
    "LeRobotV3Dataset": "avala.converters.lerobot_v3.reader",
    "UnsupportedLeRobotVersion": "avala.converters.lerobot_v3.reader",
    "LeRobotV3Writer": "avala.converters.lerobot_v3.writer",
    "lerobot_to_mcap": "avala.converters.lerobot_v3.mcap",
    "CONTROL_SOURCE_TOPIC": "avala.converters.lerobot_v3.mcap",
    "CONTROL_SOURCE_VALUES": "avala.converters.lerobot_v3.mcap",
    "DEFAULT_CONTROL_SOURCE_COLUMN": "avala.converters.lerobot_v3.mcap",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
