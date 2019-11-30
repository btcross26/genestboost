"""
Link function abstract base class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

from abc import ABC, abstractmethod

import numpy as np


class BaseLink(ABC):
    """
    Base class for link functions
    """

    def __call__(self, y: np.ndarray, inverse: bool = False) -> np.ndarray:
        if inverse:
            return self._inverse_link(y)
        return self._link(y)

    @abstractmethod
    def _link(self, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _inverse_link(self, nu: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dydnu(self, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def d2ydnu2(self, y: np.ndarray) -> np.ndarray:
        pass
