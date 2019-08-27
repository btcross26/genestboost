"""
Identity link function implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from .base_class import BaseLink


class IdentityLink(BaseLink):
    def _link(self, y):
        return 1.0 * y

    def _inverse_link(self, nu):
        return 1.0 * nu

    def dydnu(self, y):
        return 1.0

    def d2ydnu2(self, y):
        return 0.0
