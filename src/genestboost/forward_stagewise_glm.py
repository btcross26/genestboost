"""
Forward stagewise GLM class implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from typing import Optional

import numpy as np

from .boosted_model import BoostedModel
from .weak_learners import SimpleOLS


class ForwardStagewiseGLM(BoostedModel):
    """
    Forward Stagewise GLM class implementation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        model = SimpleOLS()
        super().fit(X, yt, model, iterations, weights)

    def get_coefficient_order(self):
        if self._is_fit:
            in_set = set()
            var_order = list()
            for model in self._model_list:
                coef_index = model.coef_index_
                if coef_index not in in_set:
                    var_order.append(coef_index)
                    in_set.add(coef_index)
            return var_order

    def get_coefficient_history(self, standardize=True):
        if self._is_fit:
            coef_array = (
                np.zeros(self._model_list[0]._X_means.shape[1])
            )
            coef_history = list()
            for model in self._model_list:
                index = model.coef_index_
                coef_array[index] += model.coef_ * self.alpha
                coef_history.append(1.0 * coef_array)
        return np.vstack(coef_history)
