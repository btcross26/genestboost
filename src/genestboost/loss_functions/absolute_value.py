"""
Absolute value loss function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import warnings

import numpy as np

from .base_class import BaseLoss


class AbsoluteLoss(BaseLoss):
    """
    Absolute loss function class
    """

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return np.abs(yt - yp)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return np.where(yt - yp < 0, 1, -1)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        warnings.warn(
            "second derivative of absolute value loss with respect to yp is zero"
        )
        return 0.0
