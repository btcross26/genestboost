"""
Student's t function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

import numpy as np

from .base_class import BaseLoss


class StudentTLoss(BaseLoss):
    """
    Student's t loss function class

    The math is likely wrong here...need to check before finalizing
    """

    def __init__(self, scale: float = 1.0, dof: int = 2):
        super().__init__()
        self.scale_ = scale
        self.nu_ = dof + 1.0

    def __call__(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss(yt, yp)

    def _loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return 0.5 * np.log(1.0 + (yt - yp) ** 2 / (self.scale_ * self.nu_))

    def dldyp(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return -1.0 / (self.scale_ * self.nu_) * (yt - yp) / (2.0 * self._loss(yt, yp))

    def d2ldyp2(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        devisor = self.scale_ * self.nu_
        residual = yt - yp
        loss = self._loss(yt, yp)
        return 2.0 * residual ** 2 / (devisor * loss) ** 2 + 2.0 / (devisor * loss)
