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

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return yp - yt * np.log(yp)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return 1.0 - yt / yp

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return yt / yp ** 2
