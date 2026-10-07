"""Format converters that run without the heavy upstream libraries.

Each sub-package reads and/or writes one external on-disk format with a minimal,
optional dependency set, so the same code can run in the SDK, in a CLI, or inside the
Avala server (no torch, no GPU stack). Importing this package is free; each converter
imports its own dependencies lazily.
"""
