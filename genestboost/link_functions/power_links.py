"""Link functions related to power function."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from typing import Union

import numpy as np

from .base_class import BaseLink


class PowerLink(BaseLink):
    """Implementation of power link functions."""

    def __init__(self, power: Union[float, int], summand: float = 0.0):
        """
        Initialize a power link instance - i.e., link = (y + summand) ** power.

        Extends BaseLink class intializer.

        Parameters
        ----------
        power: float or int
            Exponent of the link function.

        summand: float or int
            Summand of the link function.
        """
        super().__init__()
        self.power_ = power
        self.summand_ = summand

    def _link(self, y: np.ndarray) -> np.ndarray:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return (y + self.summand_) ** self.power_

    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return eta ** (1.0 / self.power_) - self.summand_

    def dydeta(self, y: np.ndarray) -> np.ndarray:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return (1.0 / self.power_) * (y + self.summand_) ** (1.0 - self.power_)

    def d2ydeta2(self, y: np.ndarray) -> np.ndarray:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return self.dydeta(y) * (1.0 / self.power_ - 1.0) / self._link(y)


class SqrtLink(PowerLink):
    """Square root link function implementation."""

    def __init__(self) -> None:
        """
        Class initializer.

        Extends PowerLink class intializer by specifying power=0.5 and summand=0.0.
        """
        super().__init__(power=0.5)


class CubeRootLink(PowerLink):
    """Cube root link function implementation."""

    def __init__(self) -> None:
        """
        Class initializer.

        Extends PowerLink class initializer by specifying power=1/3 and summand=0.0.
        """
        super().__init__(power=(1.0 / 3.0))


class ReciprocalLink(PowerLink):
    """Inverse link function implementation."""

    def __init__(self, summand: float = 0.0):
        """
        Class initializer.

        Extends PowerLink class initializer by specifying power=-1 and summand=0.0.
        """
        super().__init__(power=-1.0, summand=summand)
