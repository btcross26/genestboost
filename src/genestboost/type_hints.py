"""
Custom type hints for use in various places of module
"""

# Author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-11-15


from __future__ import annotations

from typing import Any, Dict

import numpy as np

# import Protocol and self return type type hints depending on Python version
try:  # pre-Python 3.8
    from typing_extensions import Protocol
except ImportError:  # Python 3.8+
    from typing import Protocol as Protocol  # type: ignore


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class LinearModel(Model, Protocol):
    coef_: np.ndarray
    intercept_: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearModel:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class ModelCallback(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Model:
        ...


class ActivationCallback(Protocol):
    def __call__(self, yp: np.ndarray) -> np.ndarray:
        ...


class WeightsCallback(Protocol):
    def __call__(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        ...
