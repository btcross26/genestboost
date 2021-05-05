"""BoostedLinearModel implementation for boosting models with linear coefficients."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .boosted_model import BoostedModel
from .link_functions import BaseLink
from .loss_functions import BaseLoss
from .type_hints import LinearModel, ModelCallback, WeightsCallback


class BoostedLinearModel(BoostedModel):
    """BoostedLinearModel class implementation."""

    def __init__(
        self,
        link: BaseLink,
        loss: BaseLoss,
        model_callback: Callable[..., LinearModel],
        model_callback_kwargs: Optional[Dict[str, Any]] = None,
        weights: Union[str, WeightsCallback] = "none",
        alpha: float = 1.0,
        step_type: str = "default",
        step_decay_factor: float = 0.6,
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
            A callable that returns a model object that implements fit and predict
            methods. The model object that is returned must be a linear model that
            has coef_ and intercept_ attributes.

        model_callback_kwargs: dict, optional (default=None)
            A dictionary of keyword arguments to pass to `model_callback`.

        weights: Union[str, WeightsCallback]: str or Callable
            A string specificying the type of weights (one of "none" or "newton"), or a
            callable of the form 'lambda yt, yp: weights`, where yt and yp are the
            actual target and predicted target arrays. These weights are multiplied
            element-wise by the model gradients to produce the pseudo-residuals that are
            to be predicted at each model iteration.

        alpha: float (default=1.0)
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
        super().__init__(
            link,
            loss,
            model_callback,
            model_callback_kwargs,
            weights,
            alpha,
            step_type,
            step_decay_factor,
            init_type,
            random_state,
            validation_fraction,
            validation_stratify,
            validation_iter_stop,
            tol,
        )

        self.coef_: np.ndarray
        self.intercept_: float

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
        yp_next, eta_p_next = super().boost(
            X, yt, yp, eta_p, model_callback, model_callback_kwargs, weights
        )
        model, lr = self._model_list[-1]
        self.coef_ += lr * model.coef_  # type: ignore
        self.intercept_ += lr * model.intercept_  # type: ignore
        return yp_next, eta_p_next

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
        if model_index is not None:
            warnings.warn(
                "arg:model_index is ignored for BoostedLinearModel.decision_function"
            )
        if self.get_iterations() == 0:
            eta_p = self._model_init.predict(X)
        else:
            eta_p = self.intercept_ + X[:, : self._msi].dot(self.coef_)
        return eta_p

    def get_coefficient_order(self, scale: Optional[np.ndarray] = None) -> List[int]:
        """
        Get the order that coefficients were selected for the model.

        In the case that multiple coefficients were selected for the first time at the
        same model iteration, the "larger" coefficient will be considered to have been
        selected first. The `scale` argument can be used to standardize coefficients
        if models were fitted in a manner such that coefficients were not standardized.

        Parameters
        ----------
        scale: np.ndarray, optional (default=None)
            Vector to scale the coefficients in the ordering calculation. If None, then
            coefficients are not scaled (or alternatively, all coefficients are scaled
            by a factor of 1.0).

        Returns
        -------
        coefficient_order: List[int]
            A list of zero-based indexes specifying the order that coefficients entered
            the boosted model.
        """
        scale = 1.0 if scale is None else scale
        coef_order_dict = OrderedDict()  # type: Dict[int, None]
        for model, _ in self._model_list:
            coefs = model.coef_ * scale  # type: ignore
            nc = (coefs != 0.0).sum()
            order = np.argsort(np.abs(coefs))[::-1].tolist()
            coef_order_dict.update(OrderedDict.fromkeys(order[:nc]))
        return list(coef_order_dict.keys())

    def get_coefficient_history(self, scale: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the history of coefficient values.

        Returns a matrix of coefficients, where each row is representative of the
        model coefficients at a specific boosting iteration (i.e., row 1 is after the
        first round of boosting, etc.)

        Parameters
        ----------
        scale: np.ndarray, optional (default=None)
            Vector to scale the coefficients in the history calculation. If None, then
            coefficients are not scaled (or alternatively, all coefficients are scaled
            by a factor of 1.0).

        Returns
        -------
        coefficient_history: np.ndarray [n_boosting_iterations + 1, n_coefs]
        """
        scale = 1.0 if scale is None else scale.reshape((1, -1))
        if self._is_fit:
            coef_history = list()
            for i, (model, lr) in enumerate(self._model_list):
                coef = model.coef_ * lr  # type: ignore
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

    def get_prediction_var_history(
        self, X: np.ndarray, groups: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get the history of prediction variance on the model matrix X.

        Returns a matrix of prediction variances at each round of boosting for each
        specified group of model coefficients. If `groups` is None, then each
        coefficient is considered separately as its own group.

        Parameters
        ----------
        X: np.ndarray
            The model matrix. If `init_type` was set as "residuals", then the model
            scores for which to calculate residuals should form the last column of the
            input matrix.

        groups: List[int], optional (default=None)
            A list of indices representing coefficient groups. Indices for groups should
            start at zero and be sequenced in order to the number of groups minus one.
            If None, then each feature is its own group.

        Returns
        -------
        pred_var: np.ndarray [n_boosting_iterations, n_groups]
        """
        coef_history = self.get_coefficient_history()
        pred_vars = np.zeros_like(coef_history)
        Xc = X[:, : self._msi] - X[:, : self._msi].mean(axis=0).reshape((1, -1))

        for i, coef in enumerate(coef_history):
            preds = Xc * coef.reshape((1, -1))
            pred_vars[i, :] = preds.var(axis=0)

        if groups is not None:
            groups = np.array(groups)
            max_ind = np.max(groups)
            group_vars = np.zeros((pred_vars.shape[0], max_ind + 1))
            for i in range(max_ind + 1):
                col_index = np.nonzero(groups == i)[0].tolist()
                group_vars[:, i] = np.sum(
                    (pred_vars[:, col_index] * coef_history[:, col_index]) ** 2, axis=1
                )
            pred_vars = group_vars

        return pred_vars

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
        yp, eta_p = super().initialize_model(X, yt, weights)

        if self.get_iterations() == 0:
            self.coef_ = np.zeros(X[:, : self._msi].shape[1])
            self.intercept_ = self._model_init._value

        return yp, eta_p
