"""
General boosting model class implementation
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .link_functions import BaseLink
from .loss_functions import BaseLoss
from .model_data_sets import ModelDataSets
from .type_hints import ActivationCallback, Model, ModelCallback, WeightsCallback

LOGGER = logging.getLogger(__name__)


# BoostedModel implementation
class BoostedModel:
    """
    General boosting model class implementation
    """

    def __init__(
        self,
        link,
        loss,
        model_callback,
        model_callback_kwargs=None,
        weights="none",
        alpha=1.0,
        step_type="decaying",
        step_decay_factor=0.48,
        init_type="mean",
        random_state=None,
        validation_fraction=0.0,
        validation_stratify=False,
        validation_iter_stop=10,
        tol=1e-8,
        activation_callback=lambda yp: yp,
    ):
        # set state based on initializer arguments
        self._link = link
        self._loss = loss
        self.model_callback = model_callback
        self.model_callback_kwargs = (
            dict() if model_callback_kwargs is None else model_callback_kwargs
        )  # type: Dict[str, Any]
        self.weights = weights
        self.alpha = alpha
        self.step_type = step_type
        self.step_decay_factor = step_decay_factor
        self.init_type = init_type
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.validation_stratify = validation_stratify
        self.validation_iter_stop = validation_iter_stop
        self.tol = tol
        self.activation_callback = activation_callback

        # additional vars used during the fitting process
        self._beta: float = 1.0 if step_type == "constant" else alpha
        self._msi = -1 if init_type in ["offset", "residuals"] else None
        self._loss_list: List[Tuple[float, float]] = list()
        self._model_list: List[Tuple[Model, float]] = list()
        self._is_fit: bool = False
        self._beta_index: int = -1
        self._tindex: Optional[Iterable[int]] = None
        self._vindex: Optional[Iterable[int]] = None
        self._model_init: BoostedModel.InitialModel

    def __bool__(self):
        return 1.0  # self._is_fit

    def boost(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        yp: np.ndarray,
        eta_p: np.ndarray,
        model_callback: ModelCallback,
        model_callback_kwargs: Dict[str, Any],
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def compute_link(self, yp: np.ndarray, inverse: bool = False) -> np.ndarray:
        return self._link(yp, inverse)

    def compute_loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss(yt, yp)

    def compute_gradients(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return self._loss.dldyp(yt, yp) * self._link.dydeta(yp)

    def compute_newton_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        term_1 = self._loss.d2ldyp2(yt, yp) * self._link.dydeta(yp) ** 2
        term_2 = self._loss.dldyp(yt, yp) * self._link.d2ydeta2(yp)
        denominator = term_1 + term_2
        denominator = denominator * yt.shape[0] / np.sum(denominator)
        return 1.0 / denominator

    def compute_newton_tr_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        pass

    def compute_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        if self.weights == "none":
            return 1.0

        if self.weights == "newton":
            return self.compute_newton_weights(yt, yp)

        if self.weights == "newton-tr":
            return self.compute_newton_tr_weights(yt, yp)

        if callable(self.weights):
            return self.weights(yt, yp)

        raise AttributeError(
            "attribute:<weights> should be 'none', 'newton', or a callable"
        )

    def compute_p_residuals(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        numerator = -self.compute_gradients(yt, yp)
        denominator = self.compute_weights(yt, yp)
        return numerator / denominator

    def decision_function(
        self, X: np.ndarray, model_index: Optional[int] = None
    ) -> np.ndarray:
        eta_p = self._model_init.predict(X)
        for model, lr in self._model_list[:model_index]:
            eta_p += lr * self.activation_callback(model.predict(X[:, : self._msi]))
        return eta_p

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
        min_iterations: Optional[int] = None,
    ) -> Model:
        # compute weights if null and create modeling data sets
        weights = np.ones_like(yt) if weights is None else weights
        model_data = ModelDataSets(
            X,
            yt,
            weights,
            self.validation_fraction,
            self.validation_stratify,
            self.random_state,
        )

        # initialize model and get initial predictions
        yp_train, eta_p_train = self.initialize_model(
            model_data.X_train, model_data.yt_train, model_data.weights_train
        )

        # get validation set predictions
        if model_data.has_validation_set():
            eta_p_val = self.decision_function(model_data.X_val)
            yp_val = self._link(eta_p_val, inverse=True)
        else:
            yp_val = None

        # calculate initial loss if model has not started
        if self.get_iterations() == 0:
            self._track_loss(yp_train, yp_val, model_data)

        # perform boosting iterations
        min_iterations = 0 if min_iterations is None else min_iterations
        for i in range(iterations):
            # perform boosting step
            yp_train, eta_p_train = self.boost(
                model_data.X_train,
                model_data.yt_train,
                yp_train,
                eta_p_train,
                self.model_callback,
                self.model_callback_kwargs,
                model_data.weights_train,
            )

            # track loss
            if model_data.has_validation_set():
                last_model, lr = self._model_list[-1]
                eta_p_val = eta_p_val + lr * last_model.predict(
                    model_data.X_val[:, : self._msi]
                )
                yp_val = self._link(eta_p_val, inverse=True)
            self._track_loss(yp_train, yp_val, model_data)

            # check stopping criteria
            if i + 1 > min_iterations and self._stop_model(model_data):
                break

        return self

    def get_model_links(self, X: np.ndarray) -> np.ndarray:
        """
        Does not apply lr to link columns

        Parameters
        ----------
        X

        Returns
        -------

        """
        model_links = np.zeros(
            (X.shape[0], len(self._model_list) + 1), dtype=np.float64
        )
        eta_p = self._model_init.predict(X)
        model_links[:, 0] = eta_p
        for i, (model, _) in enumerate(self._model_list):
            eta_p = self.activation_callback(model.predict(X[:, : self._msi]))
            model_links[:, i + 1] = eta_p
        return model_links

    def get_loss_history(self) -> np.ndarray:
        return np.array(self._loss_list).astype(np.float)

    def get_iterations(self) -> int:
        return len(self._model_list)

    def initialize_model(
        self, X: np.ndarray, yt: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            # initialize model lists/attributes
            self._loss_list = list()
            self._model_list = list()
            self._model_init = self.InitialModel(self._link, self.init_type)
            self._model_init.fit(X, yt, weights)

            # set _is_fit to True
            self._is_fit = True

        # calculate and return current eta_p and yp
        eta_p = self.decision_function(X)
        yp = self._link(eta_p, inverse=True)

        return yp, eta_p

    def predict(self, X: np.ndarray, model_index: Optional[int] = None) -> np.ndarray:
        eta_p = self.decision_function(X, model_index)
        return self._link(eta_p, inverse=True)

    def prediction_history(self, X: np.ndarray, links: bool = False) -> np.ndarray:
        model_links = self.get_model_links(X)
        lr_array = np.hstack([[1], [tup[1] for tup in self._model_list]]).reshape(
            (1, -1)
        )
        model_links *= lr_array
        link_history = np.cumsum(model_links, axis=1)

        if links is True:
            return link_history

        return self._link(link_history, inverse=True)

    def reset_model(self) -> None:
        self._is_fit = False

    def _compute_beta(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        if self.step_type == "decaying":
            return self._line_search_decaying(yt, eta_p, next_model_preds)

        if self.step_type == "best":
            return self._line_search_best(yt, eta_p, next_model_preds)

        if self.step_type == "constant":
            return 1.0

        raise AttributeError("init arg:<step_type> is mis-specified")

    def _line_search_best(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        self._beta = self.alpha
        beta = self._line_search_decaying(yt, eta_p, next_model_preds)
        return beta

    def _line_search_decaying(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        beta0 = self._beta
        etas = eta_p + next_model_preds * beta0
        preds = self._link(etas, inverse=True)
        loss0 = self._loss(yt, preds).mean()
        while True:
            beta = beta0 * self.step_decay_factor
            etas = eta_p + next_model_preds * beta
            preds = self._link(etas, inverse=True)
            loss = self._loss(yt, preds).mean()
            if loss < loss0:
                loss0 = loss
                beta0 = beta
            else:
                self._beta = beta0
                break
        return beta0

    def _track_loss(
        self,
        yp_train: np.ndarray,
        yp_val: Optional[np.ndarray],
        model_data: ModelDataSets,
    ) -> None:
        tloss = np.sum(
            self._loss(model_data.yt_train, yp_train) * model_data.weights_train
        ) / np.sum(model_data.weights_train)

        if model_data.has_validation_set():
            vloss = np.sum(
                self._loss(model_data.yt_val, yp_val)
                * model_data.weights_val
                / np.sum(model_data.weights_val)
            )
        else:
            vloss = np.nan
        self._loss_list.append((tloss, vloss))

    def _stop_model(self, model_data: ModelDataSets) -> bool:
        # training loss condition
        if self.tol is not None:
            tloss = self._loss_list[-1][0] - self._loss_list[-2][0]
            if np.abs(tloss) < self.tol:
                return True

        # validation loss condition
        if model_data.has_validation_set():
            n = self.validation_iter_stop
            if self.get_iterations() >= 2 * n:
                loss_array = np.array(self._loss_list[-2 * n :])[:, 1]
                if loss_array[:n].mean() < loss_array[n:].mean():
                    return True

        return False

    class InitialModel:
        def __init__(self, link: BaseLink, init_type: str = "mean"):
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
            elif self._init_type == "mean":
                value = np.sum(yt * weights) / np.sum(weights)
                self._value = self._link(value)
            else:
                raise AttributeError("init arg:<init_type> is mis-specified")

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._init_type == "residuals":
                return self._link(X[:, -1])

            if self._init_type == "offset":
                return self._value + X[:, -1]

            return np.ones(X.shape[0]) * self._value
