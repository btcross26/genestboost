"""
Least squares loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-11-03


from typing import Callable

import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betainc

from . import BaseLoss


class BetaLoss(BaseLoss):
    def __init__(self, alpha: float, beta: float, eps: float = 1e-10):
        """
        Class initializer

        Parameters
        ----------
        alpha: float
            Value of alpha in beta loss denominator, mu ** (1 - alpha)

        beta: float
            Value of beta in beta loss denominator, (1 - mu) ** (1 - beta)

        eps: float
            Small value to use in log calculations to avoid numerical error
        """
        self.alpha = alpha
        self.beta = beta
        self.scale = 1.0 / beta_function(alpha, beta)
        self._vt_callback = self.beta_callback(alpha, beta, eps)

    @staticmethod
    def beta_callback(
        alpha: float, beta: float, eps: float = 1e-10
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Static method to return beta callback function for quasi-deviance

        Parameters
        ----------
        alpha: float
            Value of alpha in beta loss denominator, mu ** (1 - alpha)

        beta: float
            Value of beta in beta loss denominator, (1 - mu) ** (1 - beta)

        eps: float
            Small value to use in log calculations to avoid numerical error

        Returns
        -------
        callable
            A callable that takes an np.ndarray and returns an np.ndarray after
            calculating the beta quasi-deviance denominator
        """
        scale = 1.0 / beta_function(alpha, beta)

        def vt_callback(yp: np.ndarray) -> np.ndarray:
            return np.exp(
                -np.log(scale)
                + (1.0 - alpha) * np.log(yp + eps)
                + (1.0 - beta) * np.log(1.0 - yp + eps)
            )

        return vt_callback

    def _loss(self, yt, yp):
        c1 = betainc(self.alpha, self.beta, yt) - betainc(self.alpha, self.beta, yp)
        c2 = betainc(self.alpha + 1.0, self.beta, yt) - betainc(
            self.alpha + 1.0, self.beta, yp
        )
        b1 = beta_function(self.alpha, self.beta)
        b2 = beta_function(self.alpha + 1, self.beta)
        return (yt * c1 * b1 - c2 * b2) * self.scale

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -(yt - yp) / self._vt_callback(yp)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        v1 = self.dldyp(yt, yp - 5e-13)
        v2 = self.dlyp(yt, yp + 5e-13)
        return (v2 - v1) / 1e-12
