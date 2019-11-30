"""
Quantile regression loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-27

import warnings

import numpy as np

from .base_class import BaseLoss


class QuantileLoss(BaseLoss):
    """
    Quantile regression loss function class
    """

    def __init__(self, p: float):
        super().__init__()
        self.p_ = p

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        multiplier = np.where(yt - yp < 0, 1.0 - self.p_, self.p_)
        return np.abs(yt - yp) * multiplier

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return np.where(yt - yp < 0, 1.0 - self.p_, -self.p_)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        warnings.warn(
            "second derivative of quantile value loss with respect to yp is zero"
        )
        return 0.0
