"""Loss function abstract base class."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

from abc import ABC, abstractmethod

import numpy as np


class BaseLoss(ABC):
    """Base class for loss functions."""

    def __call__(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Call the instance object to calculate the loss function."""
        return self._loss(yt, yp)

    @abstractmethod
    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Calculate the per-observation loss as a function of `yt` and `yp`."""
        ...

    @abstractmethod
    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Calculate the first derivative of the loss with respect to `yp`."""
        pass

    @abstractmethod
    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Calculate the second derivative of the loss with respect to `yp`."""
        pass
