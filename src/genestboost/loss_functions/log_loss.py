"""
Bernoulli loss function implementation (log-loss)
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLoss


class LogLoss(BaseLoss):
    """
    Log loss function class
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self._eps = eps

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -yt * np.log(yp + self._eps) - (1.0 - yt) * np.log(1.0 - yp + self._eps)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -(yt / (yp + self._eps)) + (1.0 - yt) / (1.0 - yp + self._eps)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return (yt / (yp ** 2 + self._eps)) + (1.0 - yt) / ((1.0 - yp) ** 2 + self._eps)
