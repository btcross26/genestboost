"""Bernoulli loss function implementation (log-loss)."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLoss


class LogLoss(BaseLoss):
    """Log loss function class."""

    def __init__(self, eps: float = 1e-12):
        """
        Class initializer.

        Extends the BaseLoss class intializer.

        Parameter
        ---------
        eps: float (default = 1e-10)
            A small constant float to prevent log from returning negative infinity.
        """
        super().__init__()
        self._eps = eps

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return -yt * np.log(yp + self._eps) - (1.0 - yt) * np.log(1.0 - yp + self._eps)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return -(yt / (yp + self._eps)) + (1.0 - yt) / (1.0 - yp + self._eps)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        return (yt / (yp ** 2 + self._eps)) + (1.0 - yt) / ((1.0 - yp) ** 2 + self._eps)
