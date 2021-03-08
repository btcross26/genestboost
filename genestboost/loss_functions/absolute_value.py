"""Absolute value loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import warnings

import numpy as np

from .base_class import BaseLoss


class AbsoluteLoss(BaseLoss):
    """Absolute loss function class."""

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return np.abs(yt - yp)

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return np.where(yt - yp < 0, 1, -1)

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        warnings.warn(
            "second derivative of absolute value loss with respect to yp is zero"
        )
        return np.zeros(yp.shape)
