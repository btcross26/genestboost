"""General boosting model class implementation for any regressor."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .link_functions import BaseLink
from .loss_functions import BaseLoss
from .model_data_sets import ModelDataSets
from .type_hints import Model, ModelCallback, WeightsCallback

LOGGER = logging.getLogger(__name__)


# BoostedModel implementation
class BoostedModel:
    """General boosting model class implementation for any regression model."""

    def __init__(
        self,
        link: BaseLink,
        loss: BaseLoss,
        model_callback: ModelCallback,
        model_callback_kwargs: Optional[Dict[str, Any]] = None,
        weights: Union[str, WeightsCallback] = "none",
        alpha: float = 1.0,
        step_type: str = "decaying",
        step_decay_factor: float = 0.48,
        init_type: str = "mean",
        random_state: Optional[int] = None,
        validation_fraction: float = 0.0,
        validation_stratify: bool = False,
        validation_iter_stop: int = 10,
        tol: Optional[float] = None,
    ):
        """
        Class initializer.

        Parameters
        ----------
        link: BaseLink
            Link function to use in boosting iterations.

        loss: BaseLoss
            Loss function to use in boosting iterations.

        model_callback: Callable
            A callable that returns a model object that implements fit and predict methods.

        model_callback_kwargs: dict, optional (default=None)
            A dictionary of keyword arguments to pass to `model_callback`.

        weights: Union[str, WeightsCallback]: str or Callable
            A string specificying the type of weights (one of "none" or "newton"), or a
            callable of the form 'lambda yt, yp: weights`, where yt and yp are the
            actual target and predicted target arrays. These weights are multiplied
            element-wise by the model gradients to produce the pseudo-residuals that are
            to be predicted at each model iteration.

        alpha: float = 1.0
            A parameter representing the intial trial learning rate. The learning rate that
            actually gets used at each iteration is dependent upon the value of `step_type`.

        step_type: str (default="decaying")
            One of "decaying", "constant", or "best". For "decaying", the initial model
            iteration will start with `alpha` as the learning parameter. If `alpha` times
            `step_decay_factor` results in a greater reduction in loss than `alpha`, then
            use the former. This is repeated until performance does not improve, and the
            final chosen rate will serve as the 'initial' rate for the next boosting
            iteration. For "constant", `alpha` will be used at every boosting iteration.
            Using "step_type" best implements the same process as "decaying", except the
            next boosting iteration will reset the learning rate back to `alpha` as the
            initial trial learning rate.

        step_decay_factor: float (default=0.48)
            The decaying factor to use with `step_type` "decaying" or "best".

        init_type: str (default="mean")
            The type of intial prediction to use. If "mean", then the initial link
            prediction (prior to boosting iterations) will be taken as the link of the mean
            of the non-transformed target. If "residuals" or "zero", the initial link
            prediction will be set to 0.0.

        random_state: int, optional (default=None)
            Random seed to be used for reproducability when random methods are used
            internally.

        validation_fraction: float (default=0.0)
            If 0.0, then no validation set will be used and training will be performed
            using the full training set. Otherwise, the fraction of observations to use
            as a holdout set for early stopping.

        validation_stratify: bool (default=False)
            If true, than stratify the validation holdout set based on the target. Useful
            when the target is binary.

        validation_iter_stop: int (default=10)
            Number of iterations to use for early stopping on the validation set. If the
            holdout loss is greater at the current iteration than `validation_iter_stop`
            iterations prior, then stop model fitting.

        tol: float, optional (default=None)
            Early stopping criteria based on training loss. If training loss fails to
            improve by at least `tol`, then stop training. If None, then training loss
            criteria is not checked to determine early stopping.
        """
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

        # additional vars used during the fitting process
        self._beta: float = alpha
        self._msi = -1 if init_type in ["offset", "residuals"] else None
        self._loss_list: List[Tuple[float, float]] = list()
        self._model_list: List[Tuple[Model, float]] = list()
        self._is_fit: bool = False
        self._model_init: BoostedModel.InitialModel

    def __bool__(self) -> bool:
        """Get the model fit status.

        Returns
        =======
        bool
            True if the model has been fit, false otherwise.
        """
        return self._is_fit

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
        """
        Boost by one model iteration.

        Creates a model using the `model_callback` callable with `model_callback_kwargs`,
        then fits this model to the pseudo-residuals. The learning rate is determined
        according to the chosen method, and current predictions are updated and returned
        such that stopping criteria can be evaluated and boosting continued. The
        fitted model resulting from the iteration is appended to the underlying model
        ensemble.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        eta_p: np.ndarray
            The current link predictions corresponding to the model iteration. This can
            be found by transforming `yp`, but passing this as an argument avoids
            duplicate computations and improves performance.

        model_callback: Callable
            A callable that returns a model object that implements fit and predict methods.

        model_callback_kwargs: dict, optional (default=None)
            A dictionary of keyword arguments to pass to `model_callback`.

        weights: np.ndarray, optional (default=None)
            Sample weights (by observation) to use for fitting. Should be positive.
            Observations with higher weights will affect the model fit more. If 'None',
            then all weights will be equal (1.0).

        Returns
        -------
        yp_next, eta_p_next: tuple(np.ndarray, np.ndarray)
            A tuple of the updated target predictions and target prediction links.
        """
        model_ = model_callback(**model_callback_kwargs)
        weights = 1.0 if weights is None else weights
        p_residuals = self.compute_p_residuals(yt, yp) * weights
        model_ = model_.fit(X[:, : self._msi], p_residuals)
        preds = model_.predict(X[:, : self._msi])
        learning_rate = self._compute_beta(yt, eta_p, preds)
        eta_p_next = eta_p + learning_rate * preds
        yp_next = self._link(eta_p_next, inverse=True)
        self._model_list.append((model_, learning_rate))
        return yp_next, eta_p_next

    def compute_link(self, yp: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Compute the element-wise link function or inverse link function.

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        Returns
        -------
        np.ndarray
        """
        return self._link(yp, inverse)

    def compute_loss(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Compute element-wise loss.

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        Returns
        -------
        np.ndarray
        """
        return self._loss(yt, yp)

    def compute_gradients(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Compute element-wise gradients.

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        Returns
        -------
        np.ndarray
        """
        return self._loss.dldyp(yt, yp) * self._link.dydeta(yp)

    def compute_newton_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Compute newton weights.

        This is a mixture of loss function derivatives and link function derivatives
        by application of chain rule. Using newton weights requires that the second
        derivatives of the loss and link function be defined. This method uses the
        computed second derivatives as-is - there are no adjustments to prevent
        the effects of ill-conditioning (very small second derivatives) or non-positive
        definiteness (negative second derivatives) on computed pseudo residuals.

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        Returns
        -------
        np.ndarray
            The element-wise reciprocal of the second-derivative of the loss function
            with respect to the link function.
        """
        term_1 = self._loss.d2ldyp2(yt, yp) * self._link.dydeta(yp) ** 2
        term_2 = self._loss.dldyp(yt, yp) * self._link.d2ydeta2(yp)
        denominator = term_1 + term_2
        return 1.0 / denominator

    def compute_weights(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Compute model weights that will be multiplied by observation gradients.

        The final result is the pseudo-residuals to be fit at the next boosting
        iteration. This essentially serves as a case/switch statement to redirect to
        the underlying calculation method.

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.
        """
        if self.weights == "none":
            return 1.0

        if self.weights == "newton":
            return self.compute_newton_weights(yt, yp)

        if callable(self.weights):
            return self.weights(yt, yp)

        raise AttributeError(
            "attribute:<weights> should be 'none', 'newton', or a callable"
        )

    def compute_p_residuals(self, yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Calculate pseudo-residuals.

        The psuedo-residuals are taken as the observation gradients times weights that
        are computed as per the selected weighting scheme ("none", "newton", callable).

        Parameters
        ----------
        yt: np.ndarray
            Observed target values.

        yp: np.ndarray
            Predicted target values.

        Returns
        -------
        np.ndarray
        """
        numerator = -self.compute_gradients(yt, yp)
        denominator = self.compute_weights(yt, yp)
        return numerator / denominator

    def decision_function(
        self, X: np.ndarray, model_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the link of computed model predictions.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        model_index: int, optional (default=None)
            If None, then return the full model prediction as the sum of all models in
            the ensemble plus the initial model prediction. If an int, then return only
            the predictions from models up to `model_index` (i.e., [:model_index]).

        Returns
        -------
        np.ndarray
            The link of the computed model predictions.
        """
        eta_p = self._model_init.predict(X)
        for model, lr in self._model_list[:model_index]:
            eta_p += lr * model.predict(X[:, : self._msi])
        return eta_p

    def decision_function_single(
        self, X: np.ndarray, model_index: int = -1, apply_learning_rate: bool = True,
    ) -> np.ndarray:
        """
        Compute the link for a specific ensemble model by index.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        model_index: int (default=-1)
            The model iteration for which to compute the decision function. By default,
            it is -1. This corresponds to the model from the most recent boosting
            iteration.

        apply_learning_rate: bool (default=True)
            If True, then the predictions from the selected model on `X` will be
            multiplied by the corresponding learning rate. Otherwise if False, the
            predictions of the selected model will be returned as if the learning rate
            was equal to 1.0.

        Returns
        -------
        np.ndarray
            The computed link values for the selected model index.
        """
        model, lr = self._model_list[model_index]
        lr = lr if apply_learning_rate else 1.0
        eta_p = lr * model.predict(X[:, : self._msi])
        return eta_p

    def fit(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        iterations: int = 100,
        weights: Optional[np.ndarray] = None,
        min_iterations: Optional[int] = None,
    ) -> Model:
        """
        Fit the boosted model.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        iterations: int (default=100)
            The maximum number of boosting iterations to perform.

        weights: np.ndarray, optional (default=None)
            Sample weights (by observation) to use for fitting. Should be positive.
            Observations with higher weights will affect the model fit more. If 'None',
            then all weights will be equal (1.0).

        min_iterations: int, optional (default=None)
            The minimum number of boosting iterations to perform. If None (the default),
            then there is no minimum.

        Returns
        -------
        self
        """
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
        Compute a matrix of model links (without applying learning rates).

        Returns a matrix of model links, where each column index corresponds to the
        same index in the model ensemble.

        Parameters
        ----------
        X: np.ndarray [n_samples, n_features]
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        Returns
        -------
        link_matrix: np.ndarray [n_samples, n_boosting_iterations]
        """
        model_links = np.zeros(
            (X.shape[0], len(self._model_list) + 1), dtype=np.float64
        )
        eta_p = self._model_init.predict(X)
        model_links[:, 0] = eta_p
        for i, (model, _) in enumerate(self._model_list):
            eta_p = model.predict(X[:, : self._msi])
            model_links[:, i + 1] = eta_p
        return model_links

    def get_loss_history(self) -> np.ndarray:
        """
        Get the loss history for the fitted model (training and validation loss).

        Returns
        -------
        np.ndarray
            A two-column array with with training and holdout loss in each column,
            respectively.
        """
        return np.array(self._loss_list).astype(np.float)

    def get_iterations(self) -> int:
        """Get the current number of model boosting iterations."""
        return len(self._model_list)

    def initialize_model(
        self, X: np.ndarray, yt: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the boosted model.

        This is called internally within the fit function. If manual boosting is being
        performed using the 'boost' method, then this method should be called before
        beginning the manual boosting procedure.

        Parameters
        ----------
        X: np.ndarray [n_samples, n_features]
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        yt: np.ndarray
            Observed target values.

        weights: np.ndarray, optional (default=None)
            Sample weights (by observation) to use for fitting. Should be positive.
            Observations with higher weights will affect the model fit more. If 'None',
            then all weights will be equal (1.0).

        Returns
        -------
        yp, eta_p: tuple(np.ndarray, np.ndarray)
            The initial target predictions and link of target prediction arrays.
        """
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
        """
        Compute model predictions in the original target space.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        model_index: int, optional (default=None)
            If None, then return the full model prediction as the sum of all models in
            the ensemble plus the initial model prediction. If an int, then return only
            the predictions from models up to `model_index` (i.e., [:model_index]).

        Returns
        -------
        predictions: np.ndarray
        """
        eta_p = self.decision_function(X, model_index)
        return self._link(eta_p, inverse=True)

    def prediction_history(self, X: np.ndarray, links: bool = False) -> np.ndarray:
        """
        Compute a prediction history.

        This will compute a matrix of predictions with each column corresponding to the
        predictions up to the underlying ensemble at that column index.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        links: bool (default=False)
            If true, then return the links of the prediction history. Otherwise, return
            the non-transformed predictions.
        """
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
        """
        Reset the model fit status to False.

        This will cause the model to reinitialize if the fit method is called after
        the reset_model method.
        """
        self._is_fit = False

    def _compute_beta(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        """
        Private method - learning rate switch statement.

        Redirect learning rate calcs as per the user-specified learning rate method.
        """
        if self.step_type == "decaying":
            return self._line_search_decaying(yt, eta_p, next_model_preds)

        if self.step_type == "best":
            return self._line_search_best(yt, eta_p, next_model_preds)

        if self.step_type == "constant":
            return self._beta

        raise AttributeError("init arg:<step_type> is mis-specified")

    def _line_search_best(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        """Private method - determine the learning rate for the "best" `step_type`."""
        self._beta = self.alpha
        beta = self._line_search_decaying(yt, eta_p, next_model_preds)
        return beta

    def _line_search_decaying(
        self, yt: np.ndarray, eta_p: np.ndarray, next_model_preds: np.ndarray
    ) -> float:
        """Private method - determine the learning rate for the "decaying" `step_type`."""
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
        """
        Private method - track model loss history.

        Implements the logic to track training and validation loss during the model
        fitting process.
        """
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
        """Private method that contains the logic for the stopping criteria."""
        # training loss condition
        if self.tol is not None:
            tloss = self._loss_list[-2][0] - self._loss_list[-1][0]
            if tloss < self.tol:
                return True

        # validation loss condition
        if model_data.has_validation_set():
            if (
                self.get_iterations() > self.validation_iter_stop
                and self._loss_list[-self.validation_iter_stop - 1][1]
                < self._loss_list[-1][1]
            ):
                return True

        return False

    class InitialModel:
        """InitialModel class implementation - internal to BoostedModel."""

        def __init__(self, link: BaseLink, init_type: str = "mean") -> None:
            """
            Class initializer.

            Parameters
            ----------
            link: BaseLink
                Link function to use for initialization.

            init_type: str (default="mean")
                One of "mean", "residuals", or "zero".
            """
            self._link = link  # type: BaseLink
            self._init_type = init_type  # type: str
            self._value = 0.0  # type: float

        def fit(
            self, X: np.ndarray, yt: np.ndarray, weights: Optional[np.ndarray] = None
        ) -> Model:
            """
            Fit the InitialModel object.

            Parameters
            ----------
            X: np.ndarray
                The model matrix. If `init_type` was set as "residuals", then the model
                scores for which to calculate residuals should form the last column of the
                input matrix.

            yt: np.ndarray
                Observed target values.

            weights: np.ndarray, optional (default=None)
                Sample weights (by observation) to use for fitting. Should be positive.
                Observations with higher weights will affect the model fit more. If 'None',
                then all weights will be equal (1.0).

            Returns
            -------
            self
            """
            weights = np.ones_like(yt) if weights is None else weights
            if self._init_type in ["zero", "residuals"]:
                self._value = 0.0
            elif self._init_type == "mean":
                value = np.sum(yt * weights) / np.sum(weights)
                self._value = self._link(value)
            else:
                raise AttributeError("init arg:<init_type> is mis-specified")

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Compute InitialModel predictions.

            Parameters
            ----------
            X: np.ndarray
                The model matrix. If `init_type` was set as "residuals", then the model
                scores for which to calculate residuals should form the last column of the
                input matrix.

            Returns
            -------
            predictions: np.ndarray
            """
            if self._init_type == "residuals":
                return self._link(X[:, -1])

            if self._init_type == "offset":
                return self._value + X[:, -1]

            return np.ones(X.shape[0]) * self._value
