"""
Complementary log-log link function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class CLogLogLink(BaseLink):
    """
    Complementary log-log link function
    """

    def __init__(self, eps: float = 1e-24):
        super().__init__()
        self._eps = eps

    def _link(self, y: np.ndarray) -> np.ndarray:
        return np.log(-np.log(1.0 - y + self._eps))

    def _inverse_link(self, nu: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-np.exp(nu))

    def dydnu(self, y: np.ndarray) -> np.ndarray:
        return -(1.0 - y) * np.log(1.0 - y + self._eps)

    def d2ydnu2(self, y: np.ndarray) -> np.ndarray:
        return self.dydnu(y) * (1.0 + np.log(1.0 - y + self._eps))
