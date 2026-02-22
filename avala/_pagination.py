"""Cursor-based pagination support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class CursorPage(Generic[T]):
    """A single page of results from a paginated API endpoint."""

    items: list[T] = field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
