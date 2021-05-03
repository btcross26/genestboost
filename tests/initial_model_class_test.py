"""
Unit tests for BoostedModel.InitialModel class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2020-04-06


import logging

import numpy as np
import pytest

from genestboost import BoostedModel
from genestboost.link_functions import IdentityLink

from .weak_learners.simple_pls_data import X, y

LOGGER = logging.getLogger(__name__)


# test init types
@pytest.mark.parametrize(
    "init_type", ["zero", "residuals", "mean"], ids=["zero", "residuals", "mean"],
)
def test_init_types(init_type):
    """
    Quick obvious tests on initial values in InitialModel class
    """
    # GIVEN a BoostedModel instance with internal link, loss, and model_callbacks
    model = BoostedModel.InitialModel(link=IdentityLink(), init_type=init_type)

    # WHEN fitting the model
    model.fit(X, y)

    # THEN appropriate attributes should be stored based on init_type
    # when weights are not specified
    if init_type in ["zero", "residuals"]:
        assert model._value == 0.0
    elif init_type == "mean":
        assert model._value == model._link(np.mean(y))


# test initial model exception
@pytest.mark.parametrize("init_type", [None, "other", "error", 5])
def test_initial_model_exception(init_type):
    """
    Test exception for BoostedModel.InitialModel when init_type is mis-specified
    """
    # GIVEN a BoostedModel instance with internal link, loss, and model_callbacks
    # and errant specification of init_type
    model = BoostedModel.InitialModel(link=IdentityLink(), init_type=init_type)

    # WHEN fitting the model
    # THEN an exception should be thrown
    with pytest.raises(AttributeError) as excinfo:
        model.fit(X, y)
    assert excinfo.value.args[0] == "init arg:<init_type> is mis-specified"


# test predict method of InitialModel
@pytest.mark.parametrize("init_type", [None, "other", "error", 5])
def test_initial_model_exception(init_type):
    """
    Test exception for BoostedModel.InitialModel when init_type is mis-specified
    """
    # GIVEN a BoostedModel instance with internal link, loss, and model_callbacks
    # and errant specification of init_type
    model = BoostedModel.InitialModel(link=IdentityLink(), init_type=init_type)

    # WHEN fitting the model
    # THEN an exception should be thrown
    with pytest.raises(AttributeError) as excinfo:
        model.fit(X, y)
    assert excinfo.value.args[0] == "init arg:<init_type> is mis-specified"
