"""
Optional full-wave and mode-solver backends for microringlib.

These modules are intentionally optional:

- modesolverpy: FDE/eigenmode solving
- meep: FDTD simulation

The main microringlib package does not require either dependency.
"""

from .common import SolverAvailability, check_import, require_module

__all__ = [
    "SolverAvailability",
    "check_import",
    "require_module",
]
