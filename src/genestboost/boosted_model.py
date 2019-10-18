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
        validation_fraction: Optional[float] =None,
        validation_index=None,
        validation_iter_stop=10,
        random_state=None,
        tol=1e-8
    ):
        self._link = link
        self._loss = loss
        self.weights = weights
        self.alpha = alpha
        self.step_type = step_type
        self.yp0_ = None,
        self._loss_list = list() if track_loss else None
        self._model_list = list()
        self._is_fit = False
        self._betas = (np.tile([1, 2, 5], 4) * np.repeat([0.001, 0.01, 0.1, 1.0], 3))[:-2]
        self._beta_index = None
        self._validation_fraction = validation_fraction
        self._vindex = validation_index
        self._validation_iter_stop = validation_iter_stop
        self._random_state = random_state
        self._tol = tol

    def compute_gradients(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss.dldyp(yt, yp) * self._link.dydnu(yp)

    def compute_newton_weights(self, yt: np.ndarray, yp: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        term_1 = self._loss.d2ldyp2(yt, yp) * self._link.dydnu(yp) ** 2
        term_2 = self._loss.dldyp(yt, yp) * self._link.d2ydnu2(yp)
        denominator = term_1 + term_2
        denominator = np.where(denominator == 0,
                               np.sign(denominator) * eps,
                               denominator)
        return 1.0 / denominator

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

    def _initialize_model(self, yt: np.ndarray) -> None:
        if not self._is_fit:
            # get training/validation index
            self._tindex = [i for i in range(yt.shape[0])]
            if self._vindex is None and self._validation_fraction is not None:
                self._vindex = np.random.choice(yt.shape[0],
                                                int(self._validation_fraction * yt.shape[0]),
                                                replace=False)
            if self._vindex is not None:
                self._tindex = set(self._tindex).difference(set(self._vindex))
                self._tindex = sorted(list(self._tindex))

            # initialize predictions
            self.yp0_ = yt[self._tindex].mean()
            self._loss_list = None if self._loss_list is None else list()
            self._model_list = list()
            self._model_weights = list()
            self._is_fit = True

    def _track_loss(self, yt: np.ndarray, yp: np.ndarray) -> None:
        if self._loss_list is not None:
            loss = self._loss(yt, yp)
            if self._vindex is not None:
                tloss = loss[self._tindex].mean()
                vloss = loss[self._vindex].mean()
            else:
                tloss = loss.mean()
                vloss = None
            self._loss_list.append((tloss, vloss))

    def _line_search_best(self, yt, eta_p, next_model_preds):
        etas = eta_p.reshape((-1, 1)) + next_model_preds.reshape((-1, 1)) * self._betas
        preds = self._link(etas, inverse=True)
        loss_vector = self._loss(yt.reshape((-1, 1)), preds).mean(axis=0)
        argmin = np.argmin(loss_vector)
        return self._betas[argmin]

    def _line_search(self, yt, eta_p, next_model_preds):
        if self._beta_index == 0:
            return self._betas[0]
        if len(self._model_list) == 0:
            beta = self._line_search_best(yt, eta_p, next_model_preds)
            self._beta_index = np.argwhere(self._betas == beta)[0, 0]

        etas = eta_p + next_model_preds * self._betas[self._beta_index]
        preds = self._link(etas, inverse=True)
        loss0 = self._loss(yt, preds).mean()
        while self._beta_index > 0:
            index = self._beta_index - 1
            etas = eta_p + next_model_preds * self._beta_index[index]
            preds = self._link(etas, inverse=True)
            loss = self._loss(yt, preds).mean()
            if loss < loss0:
                self._beta_index = index
                loss0 = loss
            else:
                break
        return self._betas[self._beta_index]

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        model,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self._initialize_model(yt)
        yp = self.yp0_ * np.ones_like(yt)   ### this needs thought/correction for warm starts
        self._track_loss(yt, yp)
        eta_p = self.decision_function(X)
        model_ = model.clone()
        n = self._validation_iter_stop
        for _ in range(iterations):
            next_model = self._fit_next_model(X, yt, yp, model, weights)
            preds = next_model.predict(X)
            beta = self._line_search(yt[self._tindex], eta_p[self._tindex], preds[self._tindex])
            learning_rate = self.alpha * beta
            eta_p += learning_rate * preds
            yp = self._link(eta_p, inverse=True)
            self._track_loss(yt, yp)
            self._model_list.append((next_model, learning_rate))
            model = model_.clone()
            tloss = self._loss_list[-1][0] - self._loss_list
            if np.abs(tloss) < self._tol: ### needs though for signs here
                break
            if self._vindex is not None and self.get_iterations() >= 2 * n:
                loss_array = np.array(self._loss_list[-2 * n])[:, 1]
                if loss_array[:n].mean() < loss_array[n:].mean():
                    break

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        eta_p = self._link(self.yp0_) * np.ones(X.shape[0])
        for model, lr in self._model_list:
            eta_p += lr * model.predict(X)
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

    def get_loss_history(self):
        return np.array(self._loss_list).astype(np.float64)

    def get_iterations(self):
        return len(self._model_list)
