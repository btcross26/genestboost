"""
Unit tests for single-parameter loss function implementations
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-17


import logging
import warnings

import numpy as np
import pytest

from genestboost.utils import d1_central_difference, d2_central_difference

from .loss_test_list import LOSS_TESTS

LOGGER = logging.getLogger(__name__)


# constants
TEST_NAMES = list(map(lambda x: x[0], LOSS_TESTS))


# generic class to test loss functions
@pytest.mark.parametrize(
    "test_name,loss,loss_test_values,tol", LOSS_TESTS, ids=TEST_NAMES
)
class TestLossFunction:
    """
    Generic unit tests for loss functions
    """

    def test_loss_function(self, test_name, loss, loss_test_values, tol):
        """
        Test loss calculations
        """
        # GIVEN some true (yt, yp, error) tuples of loss function values
        yt = loss_test_values[:, 0]
        yp = loss_test_values[:, 1]
        error = loss_test_values[:, 2]

        # WHEN the loss function is called on yt, yp
        calculated_values = loss(yt, yp)

        # THEN the correct values of error should be returned
        np.testing.assert_allclose(calculated_values, error, atol=1e-4, rtol=1e-5)

    def test_d1_loss(self, test_name, loss, loss_test_values, tol):
        """
        Test loss first derivative calculations
        """
        # GIVEN some yt, yp locations to calculate link derivatives
        yt = loss_test_values[:, 0]
        yp = loss_test_values[:, 1]

        # WHEN the first derivative is approximated using central differences
        def func(y):
            return loss(yt, y)

        calculated_values = d1_central_difference(func, yp, h=1e-8)

        # THEN the calculated values of the derivative should be close to the
        # true implemented computed values
        expected_values = loss.dldyp(yt, yp)
        np.testing.assert_allclose(
            calculated_values, expected_values, atol=tol[0], rtol=tol[1]
        )

    def test_d2_loss(self, test_name, loss, loss_test_values, tol):
        """
        Test loss second derivative calculations
        """
        # GIVEN some yt, yp locations to calculate link derivatives
        yt = loss_test_values[:, 0]
        yp = loss_test_values[:, 1]

        # WHEN the second derivative is approximated using central differences
        def func(y):
            return loss(yt, y)

        calculated_values = d2_central_difference(func, yp, h=1e-6)

        # THEN the calculated values of the derivative should be close to the
        # true implemented computed values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected_values = loss.d2ldyp2(yt, yp)
        np.testing.assert_allclose(
            calculated_values, expected_values, atol=tol[0], rtol=tol[1]
        )
