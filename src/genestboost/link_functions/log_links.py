"""
Link functions related to log
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class LogLink(BaseLink):
    def __init__(self, summand: float = 0.0):
        super().__init__()
        self.summand_ = summand

    def _link(self, y: np.ndarray) -> np.ndarray:
        return np.log(y + self.summand_)

    def _inverse_link(self, nu: np.ndarray) -> np.ndarray:
        return np.exp(nu) - self.summand_

    def dydnu(self, y: np.ndarray) -> np.ndarray:
        return y + self.summand_

    def d2ydnu2(self, y: np.ndarray) -> np.ndarray:
        return y + self.summand_


class Logp1Link(LogLink):
    def __init__(self) -> None:
        super().__init__(summand=1.0)
