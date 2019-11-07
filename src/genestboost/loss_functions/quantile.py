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

    def __init__(self, p):
        super().__init__()
        self.p_ = p

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        multiplier = np.where(yt - yp < 0, 1.0 - self.p_, self.p_)
        return np.abs(yt - yp) * multiplier

    def dldyp(self, yt, yp):
        return np.where(yt - yp < 0, 1.0 - self.p_, -self.p_)

    def d2ldyp2(self, yt, yp):
        warnings.warn(
            "second derivative of quantile value loss with respect to yp is zero"
        )
        return 0.0
