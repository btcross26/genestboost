"""
Link function abstract base class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


class BaseLink:
    """
    Base class for link functions
    """

    def __call__(self, y, inverse=False):
        if inverse:
            return self._inverse_link(y)
        return self._link(y)

    def _link(self, y):
        raise NotImplementedError("class method:<_link> not implemented")

    def _inverse_link(self, nu):
        raise NotImplementedError("class method:<_inverse_link> not implemented")

    def dydnu(self, y):
        raise NotImplementedError("class method:<dydnu> not implemented")

    def d2ydnu2(self, y):
        raise NotImplementedError("class method:<d2ydnu2> not implemented")
