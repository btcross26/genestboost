"""
Helper functions to approximate gradients.

Uses central differences for use in testing and numerical derivatives
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-16


from typing import Callable

import numpy as np


def d1_central_difference(
    func: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float = 1e-8
) -> np.ndarray:
    """
    Approximate the first derivative of func at y.

    Uses a first-order central difference with spacing `h` (error term O(h^2))

    Parameters
    ----------
    func: callable
        Function that takes y as an argument and returns a value

    y: np.ndarray
        Locations to calculate the value of func

    h: float
        Spacing to use in calculating central difference. This value should be
        greater than zero.

    Returns
    -------
    np.ndarray
        The approximated values of the first derivative
    """
    value = (func(y + h) - func(y - h)) / (2.0 * h)
    return value


def d2_central_difference(
    func: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float = 1e-8
) -> np.ndarray:
    """
    Approximate the first derivative of func at y.

    Uses a first-order central difference with spacing `h` (error term O(h^2))

    Parameters
    ----------
    func: callable
        Function that takes y as an argument and returns a value

    y: np.ndarray
        Locations to calculate the value of func

    h: float
        Spacing to use in calculating central difference. This value should be
        greater than zero.

    Returns
    -------
    np.ndarray
        The approximated values of the first derivative
    """
    value = (func(y + h) - 2.0 * func(y) + func(y - h)) / (h ** 2)
    return value
