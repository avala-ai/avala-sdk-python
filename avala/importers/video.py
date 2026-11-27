"""Select native video originals for named vendor contributions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from avala.importers import _INDEXED_SUFFIXES


_VIDEO_SUFFIXES = tuple(suffix for suffix in _INDEXED_SUFFIXES["video"] if suffix != ".episode")


def select_video_files(files: Sequence[tuple[str, str]], *, single_file: bool) -> list[tuple[str, str]]:
    selected = []
    for local, relative in files:
        supported = relative.lower().endswith(_VIDEO_SUFFIXES)
        if not supported or (not single_file and Path(relative).name.startswith("._")):
            if single_file:
                raise ValueError(f"unsupported video input {relative!r}; expected {', '.join(_VIDEO_SUFFIXES)}")
            continue
        path = Path(local)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"video input {relative!r} must be a regular file, not a symbolic link")
        if path.stat().st_size == 0:
            raise ValueError(f"video input {relative!r} is empty")
        selected.append((local, relative))
    if not selected:
        raise ValueError(f"no supported videos found; expected {', '.join(_VIDEO_SUFFIXES)}")
    return selected
