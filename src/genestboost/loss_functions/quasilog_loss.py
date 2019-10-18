"""
QuasiLogLoss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-10-16

import numpy as np
from scipy.special import beta as beta_function
from .base_class import BaseLoss


class QuasiLogLoss(BaseLoss):
    """
    QuasiLogLoss loss function class
    """
    def __init__(self, vt_callback, d0_n=100, d2_eps=1e-12):
        super().__init__()
        self._vt_callback = vt_callback
        self._d0_n = d0_n + (d0_n % 2)
        self._d2_eps = d2_eps

    def _loss(self, yt, yp):
        # use Simpson's rule to numerically integrate
        dims = tuple([1] * yp.ndim + [-1])
        x = np.linspace(0.0, 1.0, self._d0_n + 1).reshape(dims)
        iwts = (np
                .hstack([[1.0],
                         np.tile([4.0, 2.0], (self._d0_n - 2) // 2),
                         [4.0, 1.0]])
                .reshape(dims))
        ipts = np.expand_dims(yp, axis=-1) + np.expand_dims(yt - yp, axis=-1) * x
        values = np.sum(self.dldyp(np.expand_dims(yt, axis=-1), ipts) * iwts, axis=-1)
        values *= (yp - yt) / (3.0 * self._d0_n)
        return values

    def dldyp(self, yt, yp):
        return -(yt - yp) / self._vt_callback(yp)

    def d2ldyp2(self, yt, yp):
        v1 = self.dldyp(yt, yp - self._d2_eps)
        v2 = self.dldyp(yt, yp + self._d2_eps)
        return (v2 - v1) / (2.0 * self._d2_eps)
