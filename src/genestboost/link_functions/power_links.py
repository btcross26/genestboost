"""
Link functions related to power function
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class PowerLink(BaseLink):
    def __init__(self, power, summand=0.0):
        super().__init__(self)
        if power == 0.0:
            raise ValueError("for power=0.0, use LogLink")
        self.power_ = power
        self.summand_ = summand

    def _link(self, y):
        return (y + self.summand_) ** self.power_

    def _inverse_link(self, nu):
        return np.exp(nu) - self.summand_

    def dydnu(self, y):
        return (1.0 / self.power_) * (y + self.summand_) ** (1.0 - self.power_)

    def d2ydnu2(self, y):
        return (
            self.dydnu(y)
            * (1.0 / self.power_ - 1.0)
            / (y + self.summand_) ** self.power_
        )


class SqrtLink(PowerLink):
    def __init__(self):
        super().__init__(self, power=0.5)


class CubeRootLink(PowerLink):
    def __init__(self):
        super().__init__(self, power=(1.0 / 3.0))


class ReciprocalLink(PowerLink):
    def __init__(self, summand=0.0):
        super().__init__(self, power=-1.0, summand=summand)
