"""
Unit tests for SimplePLS weak learner class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-20


import numpy as np
import pytest

from genestboost.weak_learners import SimplePLS

from .simple_pls_data import X, y


@pytest.mark.parametrize("threshold", [None, 0.1, 0.5, 1.0])
def test_fit_max_vars_1(threshold):
    # GIVEN a SimplePLS instance with max_vars=1 and arbitrary filter_threshold
    model = SimplePLS(max_vars=1, filter_threshold=None)

    # WHEN the model is fit using provided dummy X, y data
    model.fit(X, y)

    # THEN only column 7 should have a coefficient (most correlated)
    assert model.coef_[7] != 0

    # THEN all other coefficients should be zero
    assert np.all(model.coef_[:7] == 0.0)

    # THEN the actual fitted column 7 coefficient and intercpet should be equal to the
    # least squares with one-var estimate using column 7
    Xs = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X[:, [7]]])
    ls_coef = np.linalg.lstsq(Xs, y, rcond=None)[0]
    np.testing.assert_allclose(
        np.array([model.intercept_, model.coef_[7]]), ls_coef, atol=0.0, rtol=1e-6
    )


@pytest.mark.parametrize("num_vars", [1, 2, 3, 4, 5, 6, 7, 8])
def test_fit_multiple_vars(num_vars):
    # GIVEN a SimplePLS instance with max_vars=num_vars and no filter_threshold
    model = SimplePLS(max_vars=num_vars, filter_threshold=None)

    # WHEN the model is fit using provided dummy X, y data
    model.fit(X, y)

    # THEN the number of non-zero coefs should equal num_vars
    assert (model.coef_ != 0.0).sum() == num_vars
