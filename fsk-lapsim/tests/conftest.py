"""Shared fixtures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from fsk_lapsim.vehicle import Vehicle, load_vehicle  # noqa: E402


@pytest.fixture(scope="session")
def baseline() -> Vehicle:
    """The bundled (assumed) baseline vehicle."""
    return load_vehicle()
