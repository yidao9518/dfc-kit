"""Internal NumPy array ownership helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def readonly_copy(values: ArrayLike) -> NDArray:
    """Return an independent NumPy array with mutation disabled."""
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output
