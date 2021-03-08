"""Log cosh loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import numpy as np

from .base_class import BaseLoss


class LogCoshLoss(BaseLoss):
    """Log cosh loss function class."""

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return np.log(np.cosh(yt - yp))

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return -np.tanh(yt - yp)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        return 1.0 / np.cosh(yt - yp) ** 2
