"""
List of test params to import into loss_function_tests.py
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-17


import numpy as np

from genestboost.loss_functions import (
    AbsoluteLoss,
    LeastSquaresLoss,
    LogCoshLoss,
    LogLoss,
    PoissonLoss,
    QuantileLoss,
    StudentTLoss,
)

# setup link test params to loop through tests
LOSS_TESTS = list()

# least squares test params
LOSS_TESTS.append(
    (
        "least_squares",
        LeastSquaresLoss(),
        np.array([[2.0, 2.0, 0.0], [5.0, 4.0, 1.0 / 2.0], [10.0, 15.0, 25.0 / 2.0]]),
        (0.002, 0.0001),
    )
)

# absolute loss test params
LOSS_TESTS.append(
    (
        "absolute_loss",
        AbsoluteLoss(),
        np.array([[1.5, 2.0, 0.5], [5.0, 4.0, 1.0], [10.0, 15.0, 5.0]]),
        (0.001, 0.0001),
    )
)

# logcosh loss test params
LOSS_TESTS.append(
    (
        "logcosh_loss",
        LogCoshLoss(),
        np.array([[1.5, 2.0, 0.120114507],
                  [5.0, 4.0, 0.433780830],
                  [10.0, 15.0, 4.30689822]]),
        (0.001, 0.0001),
    )
)

# log loss test params
LOSS_TESTS.append(
    (
        "log_loss",
        LogLoss(),
        np.array([[1.0, 0.5, 0.693147180],
                  [1.0, np.exp(-1), 1.0],
                  [1.0, np.exp(-2), 2.0],
                  [0.0, 0.5, 0.693147180],
                  [0.0, 1.0 - np.exp(-1), 1.0],
                  [0.0, 1.0 - np.exp(-2), 2.0]]),
        (0.001, 0.0001),
    )
)

# poisson loss test params
LOSS_TESTS.append(
    (
        "poisson_loss",
        PoissonLoss(),
        np.array([[0.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, np.exp(-1), np.exp(-1) + 1.0],
                  [1.0, np.exp(2), np.exp(2) - 2.0],
                  [5.0, np.exp(1), np.exp(1) - 5.0]]),
        (0.001, 0.0001),
    )
)
