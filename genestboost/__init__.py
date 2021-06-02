"""genestboost top-level module."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-28

from .boosted_linear_model import BoostedLinearModel
from .boosted_model import BoostedModel
from .model_data_sets import ModelDataSets

__version__ = "0.3.1"

__all__ = ["BoostedLinearModel", "BoostedModel", "ModelDataSets", "__version__"]
