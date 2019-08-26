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

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        return -yt * np.log(yp) - (1.0 - yt) * np.log(1.0 - yp)

    def dldyp(self, yt, yp):
        return -(yt / yp) + (1.0 - yt) / (1.0 - yp)

    def d2ldyp2(self, yt, yp):
        return (yt / yp ** 2) + (1.0 - yt) / (1.0 - yp) ** 2
