"""
Tests for functions in genestboost.utils gradients.py src file
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-16


import numpy as np

from genestboost.utils import d1_central_difference, d2_central_difference


# tests for central difference utils functions
class TestCentralDifferences:
    @staticmethod
    def f(y):
        """
        Arbitrary polynomial function
        """
        return 3.0 * y ** 3 - 4.0 * y ** 2 + 7.0 * y - 4.0

    @staticmethod
    def fp1(y):
        """
        First derivative of f
        """
        return 9.0 * y ** 2 - 8.0 * y + 7.0

    @staticmethod
    def fp2(y):
        """
        Second derivative of f
        """
        return 18.0 * y - 8.0

    def test_d1_central_difference(self):
        """
        Test first-order first derivative utils function
        """
        # GIVEN a simple polynomial function, derivatives, and evaluation locations
        y = np.linspace(2, 12, 101)

        # WHEN the first order central difference is used to approximate derivatives
        expected_values_d1 = self.fp1(y)
        expected_values_d2 = self.fp2(y)
        calculated_values_d1 = d1_central_difference(self.f, y, 1e-8)
        calculated_values_d2 = d1_central_difference(self.fp1, y, 1e-8)

        # THEN calculate values should be reasonable close to exact values
        np.testing.assert_allclose(
            calculated_values_d1, expected_values_d1, atol=0.0, rtol=1e-3
        )
        np.testing.assert_allclose(
            calculated_values_d2, expected_values_d2, atol=0.0, rtol=1e-3
        )

    def test_d2_central_difference(self):
        """
        Test first-order second derivative utils function
        """
        # GIVEN a simple polynomial function, derivatives, and evaluation locations
        y = np.linspace(2, 12, 101)

        # WHEN the first order central difference is used to approximate 2nd derivatives
        expected_values_d2 = self.fp2(y)
        calculated_values_d2 = d2_central_difference(self.f, y, 1e-6)

        # THEN calculate values should be reasonable close to exact values
        np.testing.assert_allclose(
            calculated_values_d2, expected_values_d2, atol=0.0, rtol=1e-2
        )
