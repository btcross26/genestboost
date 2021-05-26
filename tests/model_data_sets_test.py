"""
Unit tests for ModelDataSets class.

This ModelDataSets class basically does nothing more than take a data set and split it
if given a `validation_fraction` argument. There is a stratify option for
classification sets as well - the logic there is not really tested below
though other than insuring 100% coverage when providing the argument.
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2021-05-25

import logging

import numpy as np
import pytest

from genestboost import ModelDataSets

LOGGER = logging.getLogger(__name__)

# global test vars
X = np.random.randn(1000, 3)
yt = np.random.binomial(1, 0.5, X.shape[0])

# test init and that model is not fit after initialization
@pytest.mark.parametrize(
    "validation_fraction,validation_stratify", [(0.0, False), (0.1, False), (0.2, True)]
)
def test_initialize(validation_fraction, validation_stratify):
    """
    Test for basic initialization
    """
    # GIVEN an X, yt, and validation_fraction
    # WHEN a ModelDataSets instance is initialized with these values
    dataset = ModelDataSets(
        X,
        yt,
        validation_fraction=validation_fraction,
        validation_stratify=validation_stratify,
    )

    # THEN a training set should be created internally
    assert hasattr(dataset, "_tindex")
    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "yt_train")

    # THEN a validation set should exist if validation fraction is not 0.0
    if validation_fraction != 0.0:
        assert hasattr(dataset, "_vindex")
        assert hasattr(dataset, "X_val")
        assert hasattr(dataset, "yt_val")
        assert dataset.has_validation_set()
    else:
        assert not hasattr(dataset, "_vindex")
        assert not dataset.has_validation_set()
