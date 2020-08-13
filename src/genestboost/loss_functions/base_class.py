"""
Loss function abstract base class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

from abc import ABC, abstractmethod

import numpy as np


class BaseLoss(ABC):
    """
    Base class for loss functions
    """

    def __call__(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss(yt, yp)

    @abstractmethod
    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        pass
