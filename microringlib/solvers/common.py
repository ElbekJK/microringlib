from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class SolverAvailability:
    name: str
    available: bool
    import_error: str | None = None


def check_import(module_name: str, install_hint: str = "") -> SolverAvailability:
    try:
        import_module(module_name)
        return SolverAvailability(name=module_name, available=True)
    except Exception as exc:
        hint = f" Install hint: {install_hint}" if install_hint else ""
        return SolverAvailability(
            name=module_name,
            available=False,
            import_error=f"Could not import {module_name!r}.{hint} Original error: {exc}",
        )


def require_module(module_name: str, install_hint: str = "") -> Any:
    status = check_import(module_name, install_hint=install_hint)
    if not status.available:
        raise ImportError(status.import_error)
    return import_module(module_name)
