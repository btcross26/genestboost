"""
Link functions related to log
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class LogpLink(BaseLink):
    def __init__(self, summand=0.0):
        super().__init__()
        self.summand_ = summand

    def _link(self, y):
        return np.log(y + self.summand_)

    def _inverse_link(self, nu):
        return np.exp(nu) - self.summand_

    def dydnu(self, y):
        return y + self.summand_

    def d2ydnu2(self, y):
        return y + self.summand_


class LogLink(LogpLink):
    def __init__(self):
        super().__init__(self)


class Logp1Link(LogpLink):
    def __init__(self):
        super().__init__(self, summand=1.0)
