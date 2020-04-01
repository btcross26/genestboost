"""
Unit tests for boosted_model class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2020-03-31

import logging
from unittest import mock

import numpy as np
import pytest

from genestboost import BoostedModel
from genestboost.link_functions import IdentityLink
from genestboost.loss_functions import LeastSquaresLoss
from genestboost.weak_learners import SimplePLS

LOGGER = logging.getLogger(__name__)


# module pytest fixtures

# fixture for generic boosted model object
@pytest.fixture(scope="function")
def boosted_model_instance(request):
    # setup
    model = BoostedModel(
        link=IdentityLink(),
        loss=LeastSquaresLoss(),
        model_callback=SimplePLS,
        model_callback_kwargs={"max_vars": 3},
    )
    # yield objects
    yield model

    # teardown
    del model


# test init and that model is not fit after initialization
def test_initialize_and_bool(boosted_model_instance):
    """
    Test for basic initialization and __bool__ magic method, which returns whether
    the model has been fit or not
    """
    # GIVEN a BoostedModel instance with internal link, loss, and model_callbacks
    model = boosted_model_instance

    # WHEN checking model fit, False should be returned
    assert not model


# test compute link
def test_compute_link(boosted_model_instance):
    """
    Test that compute link is calling _link object attribute
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN the instance link is mocked and compute_link called
    # THEN the link object attribute should have been called once
    with mock.patch.object(model, "_link") as mock_link:
        y = np.linspace(-100, 200, 1001)
        model.compute_link(y)

        # assertions
        mock_link.assert_called_once()


# test compute loss
def test_compute_loss(boosted_model_instance):
    """
    Test that compute_loss is calling _loss object attribute
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN the instance link is mocked and compute_link called
    # THEN the link object attribute should have been called once
    with mock.patch.object(model, "_loss") as mock_loss:
        yp = np.linspace(-100, 200, 1001)
        yt = yp + np.random.randn(*yp.shape)
        model.compute_loss(yt, yp)

        # assertions
        mock_loss.assert_called_once()


# test compute gradients
def test_compute_gradients(boosted_model_instance):
    """
    Test that compute_loss is calling _loss object attribute
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN the instance link is mocked and compute_link called
    # THEN the link object attribute should have been called once
    with mock.patch.object(model, "_loss") as mock_loss, mock.patch.object(
        model, "_link"
    ) as mock_link:
        yp = np.linspace(-100, 200, 1001)
        yt = yp + np.random.randn(*yp.shape)
        model.compute_gradients(yt, yp)

        # assertions
        mock_loss.dldyp.assert_called_once()
        mock_link.dydeta.assert_called_once()


# test compute newton weights 1
def test_compute_newton_weights_1(boosted_model_instance):
    """
    Test that compute_newton_weights makes the right calls
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN the instance link is mocked and compute_newton_weights called
    # THEN the right link/loss calls should be made
    with mock.patch.object(model, "_loss") as mock_loss, mock.patch.object(
        model, "_link"
    ) as mock_link:
        yp = np.linspace(-100, 200, 1001)
        yt = yp + np.random.randn(*yp.shape)
        model.compute_newton_weights(yt, yp)

        # assertions
        mock_loss.d2ldyp2.assert_called_once()
        mock_link.dydeta.assert_called_once()
        mock_loss.dldyp.assert_called_once()
        mock_link.d2ydeta2.assert_called_once()


# test compute newton weights 2
def test_compute_newton_weights_2(boosted_model_instance):
    """
    Test that compute_newton_weights makes the right calls
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN the Identity/Least Squares link combo is called
    yp = np.linspace(-100, 200, 1001)
    yt = yp + np.random.randn(*yp.shape)
    weights = model.compute_newton_weights(yt, yp)

    # THEN all newton weights should be equal to 1.0
    np.all(weights == 1.0)


# test compute weights
@pytest.mark.parametrize(
    "weights_value",
    ["none", "newton", mock.MagicMock()],
    ids=["none", "newton", "callable"],
)
def test_compute_weights(boosted_model_instance, weights_value):
    """
    Test that compute_newton_weights makes the right calls
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN relevant attributes to compute_weights are mocked
    # THEN the right link/loss calls should be made
    with mock.patch.object(model, "weights", weights_value), mock.patch.object(
        model, "compute_newton_weights"
    ) as mock_cnw:
        yp = np.linspace(-100, 200, 1001)
        yt = yp + np.random.randn(*yp.shape)
        weights = model.compute_weights(yt, yp)

        # assertions
        if weights_value == "none":
            assert weights == 1.0
        elif weights_value == "newton":
            mock_cnw.assert_called_once()
        elif callable(weights_value):
            weights_value.assert_called_once()


# test compute weights exception
@pytest.mark.parametrize(
    "weights_value", [1, 2, None, "None", "Newton", "error"],
)
def test_compute_weights_exception(boosted_model_instance, weights_value):
    """
    Test that compute_weights throws exception at the right times
    """
    # GIVEN a BoostedModel instance with identity link
    model = boosted_model_instance

    # WHEN relevant attributes to compute_weights are mocked
    # THEN exception should be properly thrown with errant args
    with mock.patch.object(model, "weights", weights_value), mock.patch.object(
        model, "compute_newton_weights"
    ) as mock_cnw:
        yp = np.linspace(-100, 200, 1001)
        yt = yp + np.random.randn(*yp.shape)
        with pytest.raises(AttributeError) as excinfo:
            weights = model.compute_weights(yt, yp)
    assert (
        excinfo.value.args[0] == "attribute:<weights> should be 'none', 'newton', "
        "or a callable"
    )

    #     if weights_value not in ["none", "newton"] and not callable(weights_value):
    #         with pytest.raises(AttributeError) as exc_info:
    #             weights = model.compute_weights(yt, yp)
    #             assert exc_info.value.args[0] == "attribute:<weights> should be 'none', 'newton', or a callable"
    #     else:
    #
    # # THEN all newton weights should be equal to 1.0
    # np.all(weights == 1.0)


# # test compute loss
# # test init and that model is not fit after initialization
# def test_leaky_beta_loss_transition():
#     """
#     Test for basic initialization and __bool__ magic method, which returns whether
#     the model has been fit or not
#     """
#     # GIVEN a BoostedModel instance with internal link, loss, and model_callbacks
#     model = BoostedModel(IdentityLink(),
#                          LeastSquaresLoss(),
#                          SimplePLS)
#
#     # WHEN checking model fit, False should be returned
#     assert not model


# def test_d2_link(self, test_name, link, link_test_values, d_test_values, tol):
#     """
#     Test link second derivative calculations
#     """
#     # GIVEN some y locations to calculate link derivatives
#     y = d_test_values
#     eta = link(d_test_values)
#
#     # WHEN the second derivative is approximated using central differences
#     def func(eta):
#         return link(eta, inverse=True)
#
#     calculated_values = d2_central_difference(func, eta, h=1e-6)
#
#     # THEN the calculated values of the derivative should be close to the
#     # true implemented computed values
#     expected_values = link.d2ydeta2(y)
#     np.testing.assert_allclose(
#         calculated_values, expected_values, atol=tol[0], rtol=tol[1]
#     )
