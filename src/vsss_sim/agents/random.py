"""Random opponent policy – uniformly random wheel speeds."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .. import config


def random_policy(rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    """Return a policy callable that produces uniformly random wheel speeds."""
    def _inner(obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(config.N_ROBOTS, 2)).astype(np.float32)
    return _inner
