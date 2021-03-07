"""Logit link function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class LogitLink(BaseLink):
    """Implementation of the logit link function."""

    def _link(self, y: np.ndarray) -> np.ndarray:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return np.log(y / (1.0 - y))

    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return 1.0 / (1.0 + np.exp(-eta))

    def dydeta(self, y: np.ndarray) -> np.ndarray:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return y * (1.0 - y)

    def d2ydeta2(self, y: np.ndarray) -> np.ndarray:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return self.dydeta(y) * (1.0 - 2.0 * y)
