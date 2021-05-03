"""
Custom type hints for use in various places of module
"""

# Author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-11-15


from __future__ import annotations

# import sys
# PYTHON_VERSION = float("{:d}.{:d}".format(_sys.version_info[:2]))
# import Protocol and self return type type hints depending on Python version
# if PYTHON_VERSION < 3.8:
#     from typing_extensions import Protocol
# else:  # Python 3.8+
from typing import Any, Dict
from typing import Protocol as Protocol  # type: ignore

import numpy as np


class Predictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class Model(Predictor, Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        ...


class LinearModel(Model, Protocol):
    coef_: np.ndarray
    intercept_: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearModel:
        ...


class ModelCallback(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Model:
        ...


class WeightsCallback(Protocol):
    def __call__(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        ...
