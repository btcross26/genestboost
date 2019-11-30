"""
General boosting model class implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import logging
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .link_functions import BaseLink
from .loss_functions import BaseLoss
from .type_hints import *

LOGGER = logging.getLogger(__name__)


# BoostedModel implementation
class BoostedModel:
    """
    General boosting model class implementation
    """

    def __init__(
        self,
        link: BaseLink,
        loss: BaseLoss,
        model_callback: Callable[..., Model],
        model_callback_kwargs: Optional[Dict[str, Any]] = None,
        weights: Union[str, WeightsCallback] = None,
        alpha: float = 1.0,
        step_type: str = "default",
        betas: Optional[List[float]] = None,
        init_type: Optional[str] = None,
        random_state: Optional[int] = None,
        validation_fraction: float = 0.0,
        validation_stratify: bool = False,
        validation_iter_stop: int = 10,
        tol: float = 1e-8,
        activation_callback: Optional[ActivationCallback] = None,
    ):
        # set state based on initializer arguments
        self._link = link
        self._loss = loss
        self.model_callback = model_callback
        self.model_callback_kwargs = (
            dict() if model_callback_kwargs is None else model_callback_kwargs
        )
        self.weights = weights
        self.alpha = alpha
        self.step_type = step_type
        self.init_type = init_type
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.validation_stratify = validation_stratify
        self.validation_iter_stop = validation_iter_stop
        self.tol = tol
        self.betas = list(np.logspace(-6, 0, 19)) if betas is None else betas
        self.activation_callback = (
            (lambda x: x) if activation_callback is None else activation_callback
        )

        # additional vars used during the fitting process
        self._msi = -1 if init_type in ["offset", "residuals"] else None
        self._loss_list: Optional[Iterable[Tuple[float, float]]] = None
        self._model_list: Optional[List[Model]] = None
        self._is_fit: bool = False
        self._beta_index: Optional[int] = None
        self._tindex: Optional[Iterable[int]] = None
        self._vindex: Optional[Iterable[int]] = None

    def compute_link(self, yp: np.ndarray, inverse: bool = False) -> np.ndarray:
        return self._link(yp, inverse)

    def compute_loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss(yt, yp)

    def __bool__(self) -> bool:
        return self._is_fit

    def compute_gradients(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss.dldyp(yt, yp) * self._link.dydnu(yp)

    def compute_newton_weights(
        self, yt: np.ndarray, yp: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        term_1 = self._loss.d2ldyp2(yt, yp) * self._link.dydnu(yp) ** 2
        term_2 = self._loss.dldyp(yt, yp) * self._link.d2ydnu2(yp)
        denominator = term_1 + term_2
        denominator = np.where(
            denominator == 0, np.sign(denominator) * eps, denominator
        )
        denominator = denominator * yt.shape[0] / np.sum(denominator)
        return 1.0 / denominator

    def compute_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return 1
        elif self.weights == "newton":
            return self.compute_newton_weights(yt, yp)
        elif callable(self.weights):
            return self.weights(yt, yp)
        else:
            raise AttributeError(
                "attribute:<weights> should be None, 'newton', or a callable"
            )

    def compute_p_residuals(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        numerator = -self.compute_gradients(yt, yp)
        denominator = self.compute_weights(yt, yp)
        return numerator / denominator

    def _split_data(self, X: np.ndarray, yt: np.ndarray) -> None:
        # get training/validation index
        self._tindex = [i for i in range(yt.shape[0])]
        np.random.seed(self.random_state)
        if self.validation_stratify == True:
            index_list = list()
            for group in np.unique(yt):
                mask = yt == group
                n = np.sum(mask)
                index = np.random.choice(
                    np.arange(yt.shape[0])[mask],
                    int(self.validation_fraction * n),
                    replace=False,
                )
                index_list.append(index)
            self._vindex = np.hstack(index_list)
        else:
            self._vindex = np.random.choice(
                yt.shape[0], int(self.validation_fraction * yt.shape[0]), replace=False
            )
        self._vindex = sorted(self._vindex)
        self._tindex = set(self._tindex).difference(set(self._vindex))
        self._tindex = sorted(list(self._tindex))

    def initialize_model(
        self, X: np.ndarray, yt: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            # initialize model lists/attributes
            self._loss_list = list()
            self._model_list = list()
            self._model_init = self.InitialModel(self._link, self.init_type)
            self._model_init.fit(X, yt, weights)
            self._is_fit = True

        # calculate and return current eta_p and yp
        eta_p = self._model_init.predict(X)
        yp = self._link(eta_p, inverse=True)
        return yp, eta_p

    def reset_model(self) -> None:
        self._is_fit = False
        self._loss_list = None
        self._model_list = None

    def boost(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        yp: np.ndarray,
        eta_p: np.ndarray,
        model_callback: Callable[..., Model],
        model_callback_kwargs: Optional[Dict] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if model_callback_kwargs is None:
            model_ = model_callback()
        else:
            model_ = model_callback(**model_callback_kwargs)
        weights = 1.0 if weights is None else weights
        p_residuals = self.compute_p_residuals(yt, yp) * weights
        model_ = model_.fit(X[:, : self._msi], p_residuals)
        preds = self.activation_callback(model_.predict(X[:, : self._msi]))
        beta = self._compute_beta(yt, eta_p, preds)
        learning_rate = self.alpha * beta
        eta_p_next = eta_p + learning_rate * preds
        yp_next = self._link(eta_p_next, inverse=True)
        self._model_list.append((model_, learning_rate))
        return yp_next, eta_p_next

    def _compute_beta(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        if self.step_type in ["default", "shrinking"]:
            return self._line_search_shrinking(yt, eta_p, next_model_preds)
        elif self.step_type == "decaying":
            return self._line_search_decaying(yt, eta_p, next_model_preds)
        elif self.step_type == "best":
            return self._line_search_best
        elif self.step_type == "constant":
            return self._line_search_constant(yt, eta_p, next_model_preds)

    def _track_loss(
        self,
        yt_train: np.ndarray,
        yp_train: np.ndarray,
        weights_train: np.ndarray,
        yt_val: Optional[np.ndarray],
        yp_val: Optional[np.ndarray],
        weights_val: Optional[np.ndarray],
    ) -> None:
        if self._loss_list is not None:
            tloss = np.sum(
                self._loss(yt_train, yp_train) * weights_train / np.sum(weights_train)
            )
            if yt_val is None or yp_val is None:
                vloss = None
            else:
                vloss = np.sum(self._loss(yt_val, yp_val) * weights_val) / np.sum(
                    weights_val
                )
            self._loss_list.append((tloss, vloss))

    def _line_search_constant(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        return 1.0

    def _line_search_best(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        etas = eta_p.reshape((-1, 1)) + next_model_preds.reshape((-1, 1)) * self.betas
        preds = self._link(etas, inverse=True)
        loss_vector = self._loss(yt.reshape((-1, 1)), preds).mean(axis=0)
        argmin = np.argmin(loss_vector)
        return self.betas[argmin]

    def _line_search_shrinking(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        if self._beta_index == 0:
            return self.betas[0]
        if len(self._model_list) == 0:
            beta = self._line_search_best(yt, eta_p, next_model_preds)
            self._beta_index = np.argwhere(self.betas == beta)[0, 0]

        etas = eta_p + next_model_preds * self.betas[self._beta_index]
        preds = self._link(etas, inverse=True)
        loss0 = self._loss(yt, preds).mean()
        while self._beta_index > 0:
            index = self._beta_index - 1
            etas = eta_p + next_model_preds * self.betas[index]
            preds = self._link(etas, inverse=True)
            loss = self._loss(yt, preds).mean()
            if loss < loss0:
                self._beta_index = index
                loss0 = loss
            else:
                break
        return self.betas[self._beta_index]

    def _line_search_decaying(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        if self.get_iterations() == 0:
            beta0 = self.alpha
            self._beta_index = self.alpha
        else:
            beta0 = self._beta_index

        etas = eta_p + next_model_preds * beta0
        preds = self._link(etas, inverse=True)
        loss0 = self._loss(yt, preds).mean()
        while True:
            beta = beta0 * self.betas
            etas = eta_p + next_model_preds * beta
            preds = self._link(etas, inverse=True)
            loss = self._loss(yt, preds).mean()
            if loss < loss0:
                loss0 = loss
                beta0 = beta
            else:
                self._beta_index = beta0
                break
        return beta0

    def _stop_model(self) -> None:
        # training loss condition
        if self.tol is not None:
            tloss = self._loss_list[-1][0] - self._loss_list[-2][0]
            if np.abs(tloss) < self.tol:
                return True

        # validation loss condition
        if self._vindex is not None:
            n = self.validation_iter_stop
            if self.get_iterations() >= 2 * n:
                loss_array = np.array(self._loss_list[-2 * n :])[:, 1]
                if loss_array[:n].mean() < loss_array[n:].mean():
                    return True

        return False

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
    ) -> Model:
        # compute weights if null
        weights = np.ones_like(yt) if weights is None else weights

        # split data if specified
        if self._vindex is None and self.validation_fraction != 0.0:
            self._split_data(X, yt)

        if self._vindex is not None:
            X_train = X[self._tindex]
            yt_train = yt[self._tindex]
            weights_train = weights[self._tindex]
            X_val = X[self._vindex]
            yt_val = yt[self._vindex]
            weights_val = weights[self._vindex]
        else:
            X_train = X
            yt_train = yt
            weights_train = weights
            X_val = None
            yt_val = None
            weights_val = None

        # initialize model and get initial predictions
        yp_train, eta_p_train = self.initialize_model(X_train, yt_train, weights_train)

        if X_val is not None:
            eta_p_val = self._model_init.predict(X_val)
            yp_val = self._link(eta_p_val, inverse=True)
        else:
            yp_val = None
        self._track_loss(yt_train, yp_train, weights_train, yt_val, yp_val, weights_val)

        # perform boosting iterations
        for _ in range(iterations):
            yp_train, eta_p_train = self.boost(
                X_train,
                yt_train,
                yp_train,
                eta_p_train,
                self.model_callback,
                self.model_callback_kwargs,
                weights_train,
            )
            if X_val is not None:
                eta_p_val += self.decision_function_last_model(X_val)
                yp_val = self._link(eta_p_val, inverse=True)
            self._track_loss(
                yt_train, yp_train, weights_train, yt_val, yp_val, weights_val
            )
            if self._stop_model():
                break

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        eta_p = self._model_init.predict(X)
        for model, lr in self._model_list:
            eta_p += lr * self.activation_callback(model.predict(X[:, : self._msi]))
        return eta_p

    def decision_function_last_model(self, X: np.ndarray) -> np.ndarray:
        model, lr = self._model_list[-1]
        eta_p = self.activation_callback(model.predict(X[:, : self._msi])) * lr
        return eta_p

    def predict(self, X: np.ndarray) -> np.ndarray:
        eta_p = self.decision_function(X)
        return self._link(eta_p, inverse=True)

    def prediction_history(self, X: np.ndarray, links: bool = False) -> np.ndarray:
        model_links = self.get_model_links(X)
        lr_array = np.hstack([[1], [tup[1] for tup in self._model_list]]).reshape(
            (1, -1)
        )
        model_links *= lr_array
        link_history = np.cumsum(model_links, axis=1)
        if links == True:
            return link_history
        else:
            return self._link(link_history, inverse=True)

    def get_model_links(self, X: np.ndarray) -> np.ndarray:
        model_links = np.zeros(
            (X.shape[0], len(self._model_list) + 1), dtype=np.float64
        )
        eta_p = self._model_init.predict(X)
        model_links[:, 0] = eta_p
        for i, (model, lr) in enumerate(self._model_list):
            eta_p = self.activation_callback(model.predict(X[:, : self._msi]))
            model_links[:, i + 1] = eta_p
        return model_links

    def get_loss_history(self) -> np.ndarray:
        return np.array(self._loss_list).astype(np.float)

    def get_iterations(self) -> int:
        return 0 if self._model_list is None else len(self._model_list)

    class InitialModel:
        def __init__(self, link: BaseLink, init_type: Optional[str] = None):
            self._link = link  # type: BaseLink
            self._init_type = init_type  # type: str
            self._value = 0.0  # type: float

        def fit(
            self, X: np.ndarray, yt: np.ndarray, weights: Optional[np.ndarray] = None
        ) -> Model:
            weights = np.ones_like(yt) if weights is None else weights
            if self._init_type in ["zero", "residuals"]:
                self._value = 0.0
            elif self._init_type == "offset":
                values = self._link(yt) - X[:, -1]
                self._value = np.sum(weights * values) / np.sum(weights)
            elif self._init_type is None:
                value = np.sum(yt * weights) / np.sum(weights)
                self._value = self._link(value)

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._init_type == "residuals":
                return self._link(X[:, -1])
            elif self._init_type == "offset":
                return self._value + X[:, -1]
            else:
                return np.ones(X.shape[0]) * self._value
