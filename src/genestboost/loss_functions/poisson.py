"""
Poisson loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import numpy as np

from .base_class import BaseLoss


class PoissonLoss(BaseLoss):
    """
    Poisson loss function class
    """

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        return yp - yt * np.log(yp)

    def dldyp(self, yt, yp):
        return 1.0 - yt / yp

    def d2ldyp2(self, yt, yp):
        return yt / yp ** 2
