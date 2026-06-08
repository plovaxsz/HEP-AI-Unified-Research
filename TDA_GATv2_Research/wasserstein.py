"""Lightweight compatibility shim for EnergyFlow imports.

The upstream EnergyFlow package imports a native ``wasserstein`` extension at
module import time. That compiled dependency is not available in this execution
environment, but the quark/gluon benchmark loader only needs the module to exist
so the package can initialize.

This shim keeps the public import surface alive. It is intentionally minimal and
is not a drop-in replacement for the real optimal-transport solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _BaseEMD:
    """Minimal stand-in for the native EMD solver classes."""

    R: float = 1.0
    beta: float = 1.0
    norm: bool = False
    dtype: str = "float64"

    def set_network_simplex_params(
        self,
        n_iter_max: int,
        epsilon_large_factor: float,
        epsilon_small_factor: float,
    ) -> None:
        self.n_iter_max = n_iter_max
        self.epsilon_large_factor = epsilon_large_factor
        self.epsilon_small_factor = epsilon_small_factor

    def __call__(self, *args, **kwargs) -> float:
        return 0.0

    def flows(self):
        return []


class EMD(_BaseEMD):
    """Placeholder for the standard EMD solver."""


class EMDYPhi(_BaseEMD):
    """Placeholder for the periodic-phi EMD solver."""


__all__ = ["EMD", "EMDYPhi"]
