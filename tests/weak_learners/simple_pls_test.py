"""
Unit tests for SimplePLS weak learner class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-20


from unittest.mock import MagicMock

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


def test_predict_method():
    # GIVEN a mocked SimplePLS object with predict method and known coefficients and
    # intercept attributes
    obj = MagicMock()
    obj.coef_ = np.array([0.0, 1.0, 2.0, 0.0, 3.0, -4.0, 0.0, -5.0])
    obj.intercept_ = 2.0
    obj.predict = SimplePLS.predict.__get__(obj)

    # WHEN the predict method is called on the mocked object with dummy data X
    calculated_values = obj.predict(X)

    # THEN the calculated_values should equal the values expected from a linear model
    expected_values = (
        2.0
        + 0.0 * X[:, 0]
        + 1.0 * X[:, 1]
        + 2.0 * X[:, 2]
        + 0.0 * X[:, 3]
        + 3.0 * X[:, 4]
        - 4.0 * X[:, 5]
        + 0.0 * X[:, 6]
        - 5.0 * X[:, 7]
    )
    np.testing.assert_allclose(calculated_values, expected_values, atol=0.0, rtol=1e-8)


@pytest.mark.parametrize("num_vars", [1, 2, 3])
def test_mask_coefs_max_vars(num_vars):
    # GIVEN a SimplePLS object fit on the X, y dummy data and num_vars equal to 1, 2,
    # or 3 (with no filter_threshold)
    model = SimplePLS(max_vars=num_vars, filter_threshold=None)

    # WHEN the model is fit on X, y
    model.fit(X, y)

    # THEN a 1-var model should only have non-zero coefficient column 7
    if num_vars == 1:
        assert model.coef_[7] != 0.0
        assert np.all(model.coef_[:7] == 0.0)

    # THEN a 2-var model should only have non-zero coefficient columns 7 and 0
    if num_vars == 2:
        assert np.all(model.coef_[[7, 0]] != 0.0)
        assert np.all(model.coef_[[1, 2, 3, 4, 5, 6]] == 0.0)

    # THEN a 3-var model should only have non-zero coefficient columns 7, 0, and 3
    if num_vars == 3:
        assert np.all(model.coef_[[7, 0, 3]] != 0.0)
        assert np.all(model.coef_[[1, 2, 4, 5, 6]] == 0.0)


@pytest.mark.parametrize("threshold", [1.0, 0.95, 0.60, 0.0])
def test_mask_coefs_filter_threshold(threshold):
    # GIVEN a SimplePLS object fit on the X, y dummy data with num_vars unspecified
    # and varying filter thresholds
    model = SimplePLS(max_vars=None, filter_threshold=threshold)

    # WHEN the model is fit on X, y
    model.fit(X, y)

    # THEN with 1.0 threshold, only column 7 should be selected
    if threshold == 1.0:
        assert model.coef_[7] != 0.0
        assert np.all(model.coef_[:7] == 0.0)

    # THEN with 0.95 threshold, only 7 and 0 should be selected
    if threshold == 0.95:
        assert np.all(model.coef_[[7, 0]] != 0.0)
        assert np.all(model.coef_[[1, 2, 3, 4, 5, 6]] == 0.0)

    # THEN with 0.60 threshold, only 7, 0, and 3 should be selected
    if threshold == 0.60:
        assert np.all(model.coef_[[7, 0, 3]] != 0.0)
        assert np.all(model.coef_[[1, 2, 4, 5, 6]] == 0.0)

    # THEN with 0.0 threshold, all variables should be non-zero
    if threshold <= 0.0:
        assert np.all(model.coef_ != 0.0)


def test_init_args_none():
    # GIVEN a SimplePLS object with None for both init args
    model = SimplePLS(max_vars=None, filter_threshold=None)

    # WHEN the model is fit on dummy data X, y
    model.fit(X, y)

    # THEN all variables should be used and non-zero
    assert np.all(model.coef_ != 0.0)
