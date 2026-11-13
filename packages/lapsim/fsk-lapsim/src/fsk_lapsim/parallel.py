"""Process-pool map with a shared read-only context (extension E5).

Each worker receives the context (event setup, solver options …) once through the pool
initializer instead of with every task, and keeps its own GGV cache. Results are identical to
the serial path (the simulation is deterministic).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_CTX: dict[str, Any] = {}


def _init(ctx: dict[str, Any]) -> None:
    _CTX.clear()
    _CTX.update(ctx)


def _apply(args: tuple[Callable[[Any, dict[str, Any]], Any], Any]) -> Any:
    fn, item = args
    return fn(item, _CTX)


def default_workers() -> int:
    """Number of worker processes to use by default (CPU count, at most 8)."""
    return max(1, min(8, os.cpu_count() or 1))


def pmap(
    fn: Callable[[T, dict[str, Any]], R], items: Sequence[T], ctx: dict[str, Any], workers: int = 1
) -> list[R]:
    """``[fn(item, ctx) for item in items]``, in parallel when ``workers > 1``.

    ``fn`` must be a module-level function (picklable by reference).
    """
    if workers <= 1 or len(items) < 2:
        return [fn(i, ctx) for i in items]
    n = min(workers, len(items))
    chunk = max(1, len(items) // (4 * n))
    with ProcessPoolExecutor(n, initializer=_init, initargs=(ctx,)) as ex:
        return list(ex.map(_apply, [(fn, i) for i in items], chunksize=chunk))
