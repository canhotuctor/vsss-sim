"""Stationary opponent policy – always outputs zero wheel speeds."""

from __future__ import annotations

import numpy as np

from .. import config


def stationary_policy(obs: np.ndarray) -> np.ndarray:
    """Return zero wheel speeds for all robots."""
    return np.zeros((config.N_ROBOTS, 2), dtype=np.float32)
