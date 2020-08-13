"""
Link functions related to power function
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from typing import Union

import numpy as np

from .base_class import BaseLink


class PowerLink(BaseLink):
    def __init__(self, power: Union[float, int], summand: float = 0.0):
        super().__init__()
        self.power_ = power
        self.summand_ = summand

    def _link(self, y: np.ndarray) -> np.ndarray:
        return (y + self.summand_) ** self.power_

    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return eta ** (1.0 / self.power_) - self.summand_

    def dydeta(self, y: np.ndarray) -> np.ndarray:
        return (1.0 / self.power_) * (y + self.summand_) ** (1.0 - self.power_)

    def d2ydeta2(self, y: np.ndarray) -> np.ndarray:
        return self.dydeta(y) * (1.0 / self.power_ - 1.0) / self._link(y)


class SqrtLink(PowerLink):
    def __init__(self) -> None:
        super().__init__(power=0.5)


class CubeRootLink(PowerLink):
    def __init__(self) -> None:
        super().__init__(power=(1.0 / 3.0))


class ReciprocalLink(PowerLink):
    def __init__(self, summand: float = 0.0):
        super().__init__(power=-1.0, summand=summand)
