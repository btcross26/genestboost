"""
QuasiLogLoss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-10-16

from typing import Callable, Optional

import numpy as np
from scipy.special import beta as beta_function

from .base_class import BaseLoss


class QuasiLogLoss(BaseLoss):
    """
    QuasiLogLoss loss function class
    """

    def __init__(
        self,
        vt_callback: Callable[[np.ndarray], np.ndarray],
        d0_n: int = 100,
        d2_eps: float = 1e-12,
    ):
        super().__init__()
        self._vt_callback = vt_callback
        self._d0_n = d0_n + (d0_n % 2)
        self._d2_eps = d2_eps

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        # use Simpson's rule to numerically integrate
        dims = tuple([1] * yp.ndim + [-1])
        x = np.linspace(0.0, 1.0, self._d0_n + 1).reshape(dims)
        iwts = np.hstack(
            [[1.0], np.tile([4.0, 2.0], (self._d0_n - 2) // 2), [4.0, 1.0]]
        ).reshape(dims)
        ipts = np.expand_dims(yp, axis=-1) + np.expand_dims(yt - yp, axis=-1) * x
        values = np.sum(self.dldyp(np.expand_dims(yt, axis=-1), ipts) * iwts, axis=-1)
        values = values * (yp - yt) / (3.0 * self._d0_n)
        return values

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -(yt - yp) / self._vt_callback(yp)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        v1 = self.dldyp(yt, yp - self._d2_eps)
        v2 = self.dldyp(yt, yp + self._d2_eps)
        return (v2 - v1) / (2.0 * self._d2_eps)


class BetaLoss(QuasiLogLoss):
    def __init__(
        self,
        alpha: float,
        beta: float,
        d0_n: int = 100,
        d2_eps: Optional[float] = None,
        log_eps: float = 1e-8,
    ):
        self.alpha = alpha
        self.beta = beta
        self.log_eps = log_eps

        # define callback function
        def vt_callback(yp: np.ndarray) -> np.ndarray:
            return np.exp(
                np.log(beta_function(alpha, beta))
                + (1.0 - alpha) * np.log(yp + log_eps)
                + (1.0 - beta) * np.log(1.0 - yp + log_eps)
            )

        d2_eps = log_eps / 10.0 if d2_eps is None else d2_eps
        super().__init__(vt_callback, d0_n=d0_n, d2_eps=d2_eps)


class LeakyBetaLoss(BetaLoss):
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float = 1.0,
        d0_n: int = 100,
        d2_eps: Optional[float] = None,
        log_eps: float = 1e-8,
    ):
        self.gamma = gamma
        super().__init__(alpha, beta, d0_n, d2_eps, log_eps)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        # calculate loss function values from regular beta
        values = super().dldyp(yt, yp)

        # find leaky point values
        rL = self.alpha / (self.alpha + self.beta)
        vL = self.gamma * super().dldyp(0.0, rL)
        rR = 1.0 - rL
        vR = self.gamma * super().dldyp(1.0, 1.0 - rR)

        # edit values outside of ratio bounds
        values = np.where(
            (yt - yp < -rL) & (values < vL),
            vL,
            np.where((yt - yp > rR) & (values > vR), vR, values),
        )

        return values
