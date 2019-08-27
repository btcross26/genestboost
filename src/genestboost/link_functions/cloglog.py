"""
Complementary log-log link function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import numpy as np

from .base_class import BaseLink


class CLogLogLink(BaseLink):
    """
    Complementary log-log link function
    """

    def _link(self, y):
        return np.log(-np.log(1.0 - y))

    def _inverse_link(self, nu):
        return 1.0 - np.exp(-np.exp(nu))

    def dydnu(self, y):
        return -(1.0 - y) * np.log(1.0 - y)

    def d2ydnu2(self, y):
        return self.dydnu(y) * (1.0 - np.log(1.0 - y))
