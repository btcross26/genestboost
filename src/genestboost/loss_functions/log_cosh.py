"""
Log cosh loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import numpy as np

from .base_class import BaseLoss


class LogCoshLoss(BaseLoss):
    """
    Log cosh loss function class
    """

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return np.log(np.cosh(yt - yp))

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -np.tanh(yt - yp)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return 1.0 / np.cosh(yt - yp) ** 2
