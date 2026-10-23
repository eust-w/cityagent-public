from __future__ import annotations

from typing import Any


def evaluate_year(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Import the heavyweight evaluator lazily.

    Label parsing is also reused by the chronological history index.  Keeping
    the evaluator import lazy avoids a cycle from history -> labels -> metrics
    -> orchestrator -> history.
    """

    from .metrics import evaluate_year as _evaluate_year

    return _evaluate_year(*args, **kwargs)


__all__ = ["evaluate_year"]
