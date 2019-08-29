"""
General boosting model class implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from typing import Optional

import numpy as np

from .link_functions.base_class import BaseLink
from .loss_functions.base_class import BaseLoss


class BoostedModel:
    """
    General boosting model class implementation
    """

    def __init__(
        self,
        link: BaseLink,
        loss: BaseLoss,
        weights: Optional[str] = None,
        alpha: float = 0.1,
        step_type: str = "default",
        track_loss: bool = True,
        warm_start: bool = False,
    ):
        self._link = link
        self._loss = loss
        self.weights = weights
        self.alpha = alpha
        self.step_type = step_type
        self.yp0_ = None,
        self._loss_list = list() if track_loss else None
        self.warm_start = warm_start
        self._model_list = list()
        self._model_weights = list()
        self._is_fit = False

    def compute_gradients(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss.dldyp(yt, yp) * self._link.dydnu(yp)

    def compute_newton_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        term_1 = self._loss.d2ldyp2(yt, yp) * self._link.dydnu(yp) ** 2
        term_2 = self._loss.dldyp(yt, yp) * self._link.d2ydnu2(yp)
        return 1.0 / (term_1 + term_2)

    def compute_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return 1
        elif self.weights == "newton":
            return self.compute_newton_weights(yt, yp)
        else:
            raise AttributeError("attribute:<weights> should be None or 'newton'")

    def compute_p_residuals(self, yt, yp):
        numerator = -self.compute_gradients(yt, yp)
        denominator = self.compute_weights(yt, yp)
        return numerator / denominator

    def _compute_alpha(self):
        return self.alpha

    def _initialize_predictions(self, yt: np.ndarray) -> None:
        if not self._is_fit or not self.warm_start:
            # initialize predictions
            self.yp0_ = yt.mean()
            self._loss_list = None if self._loss_list is None else list()
            self._model_list = list()
            self._model_weights = list()
            self._is_fit = True

    def _fit_next_model(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        yp: np.ndarray,
        model,
        weights: Optional[np.ndarray] = None,
    ):
        weights = 1.0 if weights is None else weights
        p_residuals = self.compute_p_residuals(yt, yp) * weights
        model.fit(X, p_residuals)
        return model

    def _track_loss(self, yt: np.ndarray, yp: np.ndarray) -> float:
        if self._loss_list is not None:
            loss = self._loss(yt, yp).mean()
            self._loss_list.append(loss)

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        model,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self._initialize_predictions(yt)
        yp = self.yp0_ * np.ones_like(yt)
        self._track_loss(yt, yp)
        eta_p = self.decision_function(X)
        model_ = model.clone()
        for _ in range(iterations):
            next_model = self._fit_next_model(X, yt, yp, model, weights)
            eta_p += self._compute_alpha() * next_model.predict(X)
            yp = self._link(eta_p, inverse=True)
            self._track_loss(yt, yp)
            self._model_list.append(next_model)
            model = model_.clone()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        eta_p = self._link(self.yp0_) * np.ones(X.shape[0])
        for model in self._model_list:
            eta_p += self._compute_alpha() * model.predict(X)
        return eta_p

    def predict(self, X: np.ndarray) -> np.ndarray:
        eta_p = self.decision_function(X)
        return self._link(eta_p, inverse=True)

    def prediction_history(self, X: np.ndarray) -> np.ndarray:
        pred_history = np.zeros((X.shape[0], len(self._model_list) + 1),
                                dtype=np.float64)
        eta_p = self._link(self.yp0_) * np.ones(X.shape[0])
        pred_history[:, 0] = self._link(eta_p, inverse=True)
        for i, model in enumerate(self._model_list):
            eta_p += self._compute_alpha() * model.predict(X)
            pred_history[:, i + 1] = self._link(eta_p, inverse=True)
        return pred_history

    def loss_history(self, X: np.ndarray, yt: np.ndarray) -> np.ndarray:
        loss_history = self.prediction_history(X)   # initialize with predictions
        loss_history = self._loss(yt.reshape((-1, 1)), loss_history)
        loss_history = loss_history.mean(axis=0)
        return loss_history
