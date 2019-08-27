"""
Least squares loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

from .base_class import BaseLoss


class LeastSquaresLoss(BaseLoss):
    """
    Least squares loss function class
    """

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        return 0.5 * (yt - yp) * (yt - yp)

    def dldyp(self, yt, yp):
        return yp - yt

    def d2ldyp2(self, yt, yp):
        return 1.0
