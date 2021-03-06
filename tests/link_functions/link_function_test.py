"""
Unit tests for link function implementations
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-16

import logging

import numpy as np
import pytest

from genestboost.utils import d1_central_difference, d2_central_difference

from .link_test_list import LINK_TESTS

LOGGER = logging.getLogger(__name__)


# constants
TEST_NAMES = list(map(lambda x: x[0], LINK_TESTS))


# generic class to test link functions
@pytest.mark.parametrize(
    "test_name,link,link_test_values,d_test_values,tol", LINK_TESTS, ids=TEST_NAMES
)
class TestLinkFunction:
    """
    Generic unit tests for link functions
    """

    def test_link_function(self, test_name, link, link_test_values, d_test_values, tol):
        """
        Test link calculations
        """
        # GIVEN some true (y, nu) pairs of link function values
        y = link_test_values[:, 0]
        eta = link_test_values[:, 1]

        # WHEN the link function is called on x
        calculated_values = link(y)

        # THEN the correct values of nu should be returned
        np.testing.assert_allclose(calculated_values, eta)

    def test_inverse_link_function(
        self, test_name, link, link_test_values, d_test_values, tol
    ):
        """
        Test inverse link calculations
        """
        # GIVEN some true (y, nu) pairs of link function values
        y = link_test_values[:, 0]
        eta = link_test_values[:, 1]

        # WHEN the inverse link function is called on nu
        calculated_values = link(eta, inverse=True)

        # THEN the correct values of x should be returned
        np.testing.assert_allclose(calculated_values, y)

    def test_d1_link(self, test_name, link, link_test_values, d_test_values, tol):
        """
        Test link first derivative calculations
        """
        # GIVEN some y locations to calculate link derivatives
        y = d_test_values
        eta = link(d_test_values)

        # WHEN the first derivative is approximated using central differences
        def func(eta):
            return link(eta, inverse=True)

        calculated_values = d1_central_difference(func, eta, h=1e-8)

        # THEN the calculated values of the derivative should be close to the
        # true implemented computed values
        expected_values = link.dydeta(y)
        np.testing.assert_allclose(
            calculated_values, expected_values, atol=tol[0], rtol=tol[1]
        )

    def test_d2_link(self, test_name, link, link_test_values, d_test_values, tol):
        """
        Test link second derivative calculations
        """
        # GIVEN some y locations to calculate link derivatives
        y = d_test_values
        eta = link(d_test_values)

        # WHEN the second derivative is approximated using central differences
        def func(eta):
            return link(eta, inverse=True)

        calculated_values = d2_central_difference(func, eta, h=1e-6)

        # THEN the calculated values of the derivative should be close to the
        # true implemented computed values
        expected_values = link.d2ydeta2(y)
        np.testing.assert_allclose(
            calculated_values, expected_values, atol=tol[0], rtol=tol[1]
        )
