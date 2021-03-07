"""Link functions related to log."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class LogLink(BaseLink):
    """Implementation of the log link function."""

    def __init__(self, summand: float = 0.0):
        """
        Class initializer.

        Extends the BaseLink class intializer.

        Parameters
        ----------
        summand: float
            Summand of the log in the link implementation - i.e.,
            link = log(y + summand).
        """
        super().__init__()
        self.summand_ = summand

    def _link(self, y: np.ndarray) -> np.ndarray:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return np.log(y + self.summand_)

    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return np.exp(eta) - self.summand_

    def dydeta(self, y: np.ndarray) -> np.ndarray:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return y + self.summand_

    def d2ydeta2(self, y: np.ndarray) -> np.ndarray:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return y + self.summand_


class Logp1Link(LogLink):
    """Log plus 1 link function implementation."""

    def __init__(self) -> None:
        """Class initializer extends LogLink initializer, setting summand equal to 1.0."""
        super().__init__(summand=1.0)
