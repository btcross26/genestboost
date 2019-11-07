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

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        return np.log(np.cosh(yt - yp))

    def dldyp(self, yt, yp):
        return -np.tanh(yt - yp)

    def d2ldyp2(self, yt, yp):
        return 1.0 / np.cosh(yt - yp) ** 2
