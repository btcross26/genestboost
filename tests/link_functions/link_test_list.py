"""
List of test params to import into link_function_tests.py
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-16

import numpy as np

from genestboost.link_functions import (
    CLogLogLink,
    CubeRootLink,
    IdentityLink,
    LogitLink,
    LogLink,
    Logp1Link,
    PowerLink,
    ReciprocalLink,
    SqrtLink,
)

# setup link test params to loop through tests
LINK_TESTS = list()

# identity test params
LINK_TESTS.append(
    (
        "identity_test",
        IdentityLink(),
        np.array([[-5.0, -5.0], [0.0, 0.0], [10.0, 10.0]]),
        np.linspace(-50, 50, 101),
        (0.005, 0.0001),
    )
)

# logit test params
LINK_TESTS.append(
    (
        "logit_test",
        LogitLink(),
        np.array([[0.5, 0.0], [0.25, -np.log(3)], [0.9, np.log(9)]]),
        np.linspace(0.01, 0.99, 99),
        (0.001, 0.0001),
    )
)

# cloglog test params
LINK_TESTS.append(
    (
        "cloglog_test",
        CLogLogLink(),
        np.array(
            [
                [0.5, np.log(-np.log(0.5))],
                [0.25, np.log(-np.log(0.75))],
                [0.9, np.log(-np.log(0.1))],
            ]
        ),
        np.linspace(0.01, 0.99, 99),
        (0.001, 0.0001),
    )
)

# log link test params
LINK_TESTS.append(
    (
        "log_test",
        LogLink(),
        np.array([[1.0, 0.0], [np.exp(1), 1.0], [np.exp(10), 10.0]]),
        np.logspace(0.0, 10.0, 111, base=np.exp(1)),
        (0.001, 0.001),
    )
)

# logp1 link test params
LINK_TESTS.append(
    (
        "logp1_test",
        Logp1Link(),
        np.array([[0.0, 0.0], [np.exp(1) - 1.0, 1.0], [np.exp(10) - 1.0, 10.0]]),
        np.logspace(0.0, 10.0, 111, base=np.exp(1)),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "power_base_test",
        PowerLink(power=2, summand=1.0),
        np.array([[0.0, 1.0], [2.0, 9.0], [10.0, 121.0]]),
        np.linspace(0.0, 100.0, 101),
        (0.005, 0.001),
    )
)

# sqrt link test params
LINK_TESTS.append(
    (
        "sqrt_link_test",
        SqrtLink(),
        np.array([[0.01, 0.1], [4.0, 2.0], [100.0, 10.0]]),
        np.linspace(0.01, 10.0, 101),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "cuberoot_link_test",
        CubeRootLink(),
        np.array([[0.001, 0.1], [8.0, 2.0], [1000.0, 10.0]]),
        np.linspace(0.01, 10.0, 101),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "reciprocal_link_test",
        ReciprocalLink(),
        np.array([[1.0, 1.0], [2.0, 0.5], [10.0, 0.1]]),
        np.linspace(0.01, 10.0, 101),
        (0.001, 0.001),
    )
)
