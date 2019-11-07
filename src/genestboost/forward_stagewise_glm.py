"""
Forward stagewise GLM class implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from typing import Optional

import numpy as np
from collections import OrderedDict

from .boosted_model import BoostedModel
from .weak_learners import SimplePLS


class ForwardStagewiseGLM(BoostedModel):
    """
    Forward Stagewise GLM class implementation
    """

    def __init__(self,
                 link,
                 loss,
                 model_callback=SimplePLS,
                 model_callback_kwargs=None,
                 max_vars=None,
                 filter_threshold=None,
                 weights: Optional[str] = None,
                 alpha: float = 1.0,
                 step_type: str = "default",
                 betas=None,
                 init_type=None,
                 random_state=None,
                 validation_fraction=0.0,
                 validation_stratify=False,
                 validation_iter_stop=10,
                 tol=1e-8):

        super().__init__(link,
                         loss,
                         model_callback,
                         model_callback_kwargs,
                         weights,
                         alpha,
                         step_type,
                         betas,
                         init_type,
                         random_state,
                         validation_fraction,
                         validation_stratify,
                         validation_iter_stop,
                         tol)

        self.coef_ = None
        self.intercept_ = None

    def initialize_model(self, X: np.ndarray, yt: np.ndarray, weights=None):
        yp, eta_p = super().initialize_model(X, yt, weights)
        self.coef_ = np.zeros(X[:, :self._msi].shape[1])
        self.intercept_ = self._model_init._value
        return yp, eta_p

    def boost(self, X, yt, yp, eta_p, model_callback, model_callback_kwargs=None, weights=None):
        yp_next, eta_p_next = super().boost(X, yt, yp, eta_p, model_callback, model_callback_kwargs, weights)
        model, lr = self._model_list[-1]
        self.coef_ += lr * model.coef_
        self.intercept_ += lr * model.intercept_
        return yp_next, eta_p_next

    def get_coefficient_order(self, scale=None):
        scale = 1.0 if scale is None else scale
        coef_order_dict = OrderedDict()
        for model, _ in self._model_list:
            coefs = model.coef_ * scale
            nc = (coefs != 0.0).sum()
            order = np.argsort(np.abs(coefs))[::-1].tolist()
            coef_order_dict.update(OrderedDict.fromkeys(order[:nc]))
        return list(coef_order_dict.keys())

    def get_coefficient_history(self, scale=None):
        scale = 1.0 if scale is None else scale.reshape((1, -1))
        if self._is_fit:
            coef_history = list()
            for i, (model, lr) in enumerate(self._model_list):
                coef = model.coef_ * lr
                if i == 0:
                    coef_history.append(coef)
                    continue
                coef = coef + coef_history[i - 1]
                coef_history.append(coef)
            coef_matrix = np.vstack([np.zeros_like(coef), coef_history])
            coef_matrix *= scale
            return coef_matrix
        else:
            raise AttributeError("model has not yet been fit")

    def get_prediction_var_history(self, X, groups=None):
        coef_history = self.get_coefficient_history()
        pred_vars = np.zeros_like(coef_history)
        Xc = X[:, :self._msi] - X[:, :self._msi].mean(axis=0).reshape((1, -1))

        for i, coef in enumerate(coef_history):
            preds = Xc * coef.reshape((1, -1))
            pred_vars[i, :] = preds.var(axis=0)

        if groups is not None:
            groups = np.array(groups)
            max_ind = np.max(groups)
            group_vars = np.zeros((pred_vars.shape[0], max_ind + 1))
            for i in range(max_ind + 1):
                col_index = np.nonzero(groups == i)[0].tolist()
                group_vars[:, i] = (
                    np.sum((pred_vars[:, col_index] * coef_history[:, col_index]) ** 2, axis=1)
                )
            pred_vars = group_vars

        return pred_vars

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.get_iterations() == 0:
            eta_p = self._model_init.predict(X)
        else:
            eta_p = self.intercept_ + X[:, :self._msi].dot(self.coef_)
        return eta_p

