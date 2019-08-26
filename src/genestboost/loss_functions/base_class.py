"""
Loss function abstract base class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


class BaseLoss:
    """
    Base class for loss functions
    """

    def __call__(self, yt, yp):
        return self._loss(yt, yp)

    def _loss(self, yt, yp):
        raise NotImplementedError("class method:<_loss> not implemented")

    def dldyp(self, yt, yp):
        raise NotImplementedError("class method:<dydnu> not implemented")

    def d2ldyp2(self, yt, yp):
        raise NotImplementedError("class method:<d2ydnu2> not implemented")
