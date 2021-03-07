"""Link function abstract base class."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from abc import ABC, abstractmethod

import numpy as np


class BaseLink(ABC):
    """Abstract base class for link functions."""

    def __call__(self, y: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Call the instance object to calculate the link function.

        If `inverse` is True (default), get the link, eta, as a function of `y`.
        If `inverse` is False, then get y as a function of the link. In this case,
        one should pass the link function eta as the argument `y`.
        """
        if inverse:
            return self._inverse_link(y)
        return self._link(y)

    @abstractmethod
    def _link(self, y: np.ndarray) -> np.ndarray:
        """Get the link, eta, as a function of y."""
        ...

    @abstractmethod
    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Get the target, y, as a function of the link, `eta`."""
        ...

    @abstractmethod
    def dydeta(self, y: np.ndarray) -> np.ndarray:
        """Get the derivative of `y` with respect to the link as a function of y."""
        ...

    @abstractmethod
    def d2ydeta2(self, y: np.ndarray) -> np.ndarray:
        """Get the second derivative of `y` with respect to the link as a function of y."""
        ...
