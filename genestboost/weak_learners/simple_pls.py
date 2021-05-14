"""Implementation for SimplePLS, a partial-PLS component regression model."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-28


import heapq
from typing import List, Optional, Tuple

import numpy as np

from ..type_hints import Model


class SimplePLS:
    """Implementation for SimplePLS, a partial-PLS component regression model."""

    def __init__(
        self, max_vars: Optional[int] = 1, filter_threshold: Optional[float] = None
    ):
        """
        Class SimplePLS initializer.

        Parameters
        ----------
        max_vars: int, optional (default=1)
            The maximum number of variables to use in the regression. The default value
            is 1, which is the special case of simple one-variable least squares
            regression. If None, then max_vars will be set to the number of variables
            in the X model matrix during the fitting process.

        filter_threshold: float, optional (default=None)
            The correlation filter threshold to use. If the ratio of the absolute value
            of the correlation coefficient for a predictor to the absolute value of the
            maximum correlation coefficient of all predictors is less than the filter
            threshold value, then the predictor will be excluded from the regression.
            The default value is None, in which case the filter threshold will be set
            equal to 0.0.

        Attributes
        ----------
        coef_ : numpy.ndarray, shape (n_predictors,)
            The estimated coefficients for the regression problem

        intercept_ : float
            The estimated intercept (bias) for the regression problem
        """
        # initialized attributes
        self._max_vars = max_vars
        self._filter_threshold = 0.0 if filter_threshold is None else filter_threshold

        # public attributes initialized during class usage
        self.coef_: np.ndarray
        self.intercept_: float

        # private attributes initialized during class usage
        self._X_means: np.ndarray
        self._X_std: np.ndarray
        self._y_mean: float
        self._y_std: float
        self._multiplier: float
        self._pls_intercept: float

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Model:
        """
        Fit a linear regression according to the specified initializer arguments.

        The fit process will result in several instance attributes being populated,
        including the public attributes housing the coefficients and intercept.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_predictors)
            The input model matrix. Should be of dtype float.

        y: numpy.ndarray, shape (n_samples, )
            The target vector

        weights: numpy.ndarray, optional (default=None)
            A value of weights to use for the linear regression fit. The default value
            is None, which results in equal weighting for all observations. If a value
            is provided, it should have the same dimensions as the target vector, y.
            The weights are applied to the correlation calculation only. They are not
            used during standardization of the X and y fitted arrays at model
            initialization.

        Returns
        -------
        model: SimplePLS
            Instance of self.
        """
        # initialize model
        Xs, ys = self._initialize_model(X, y)

        # calculate initial model coefficients
        weights = np.ones(y.shape[0]) if weights is None else weights
        coefs = np.sum(Xs * (ys * weights).reshape((-1, 1)), axis=0) / np.sum(weights)

        # mask/filter coefficients
        coefs = self._mask_coefs(coefs)

        # get regression coefficients
        self._multiplier = self._get_coef_multiplier(coefs, Xs, ys)
        coefs *= self._multiplier
        coefs *= self._y_std / self._X_std[0]
        self.coef_ = coefs

        # get intercept
        self.intercept_ = (
            self._y_mean
            - np.sum(coefs * self._X_means[0])
            - self._pls_intercept * self._y_std
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute model predictions for the given model matrix, X.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_predictors)
             The input model matrix. Should be of dtype float. The number of columns
             should be the same number of columns as the X argument that was used to
             fit the model.

        Returns
        -------
        numpy.ndarray, shape (n_samples, )
            A vector of predictions of dtype float
        """
        return self.intercept_ + X.dot(self.coef_)

    def _get_coef_multiplier(
        self, coefs: np.ndarray, Xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        """
        Get the coefficient multiplier that is applied to the coefficients (private).

        In the case of single variable regression, this multiplier will be equal to
        1.0. For multiple variables, A regression is applied to the summation of each
        selected standardized variable multiplied by its correlation coefficient. The
        final regression coefficient for a variable is then equal to its correlation
        coefficient multiplied by the multiplier returned by this method. The
        multiplier that is returned by this method is also stored in the private
        instance attribute, _multiplier.

        Parameters
        ----------
        coefs: numpy.ndarray, shape (n_predictors,)
            A vector of coefficients. For selected variables, the coefficients will be
            equal to the individual correlation coefficients of the variables with the
            target. For variables that are not selected, the coefficient values are
            equal to zero.

        Xs: numpy.ndarray, shape (n_samples, n_predictors)
            The standardized version of model matrix, X, that is being fitted

        ys: numpy.ndarray, shape (n_samples,)
            The standardized version of target vector, y, that is being fitted

        Returns
        -------
        float
            The multiplier that is applied to the selected variable correlation
            coefficients to get the final regression model
        """
        # initialize values
        self._pls_intercept = 0.0
        multiplier = 1.0
        n_coefs = np.sum(coefs != 0.0)

        # regress on weighted (by correlation coefficient) sum if max_vars is greater
        # than 1
        if n_coefs != 1:
            x_pls = Xs.dot(coefs)
            x_pls_mean = x_pls.mean()
            x_pls_std = x_pls.std()
            x_pls_std = np.where(x_pls_std == 0.0, 1.0, x_pls_std)
            x_pls_scaled = (x_pls - x_pls_mean) / x_pls_std
            alpha = np.mean(x_pls_scaled * ys)
            self._pls_intercept = alpha * x_pls_mean / x_pls_std
            multiplier = alpha / x_pls_std

        return multiplier

    def _mask_coefs(self, coefs: np.ndarray) -> np.ndarray:
        """
        Apply initializer arguments max_vars and filter_threshold (private).

        If max_vars is 1 or filter_threshold is 1.0, then only the variable
        that is most correlated with the predictor will be selected. If max_vars is
        greater than or equal to the number of predictors and filter_threshold is
        <= 0.0, then all variables are selected. Otherwise, max_vars will be selected
        and paired down according to the value of filter_threshold in the general case.

        Parameters
        ----------
        coefs: numpy.ndarray, shape (n_predictors,)
            A vector of coefficients calculated during the fitting process. The values
            of the coefficients in this case are equal to the correlation coefficients
            of the predictors with the target vector.

        Returns
        -------
        numpy.ndarray
            The vector of coefficients after max_vars have been selected and paired
            down according to the specified filter_threshold.
        """
        # initialize values
        coefs_abs = np.abs(coefs)
        max_index = np.argmax(coefs_abs)

        # edge case where there will only be one var
        if self._max_vars == 1 or self._filter_threshold >= 1.0:
            coef_mask = np.zeros(coefs.shape[0])
            coef_mask[max_index] = 1.0
            return coefs * coef_mask

        # edge case where all vars used
        num_vars = (
            coefs.shape[0]
            if self._max_vars is None
            else min(self._max_vars, coefs.shape[0])
        )
        if num_vars == coefs.shape[0] and (
            self._filter_threshold <= 0.0 or self._filter_threshold is None
        ):
            return coefs

        # apply correlation filter
        max_value = coefs_abs[max_index]
        rel_coefs = coefs_abs / max_value
        if self._filter_threshold is not None:
            coef_mask = 1.0 * (rel_coefs >= self._filter_threshold)

        # apply max_vars intermediate case if specified
        if coef_mask.sum() > num_vars:
            heap_index = list()  # type: List[Tuple[float, int]]
            for i in np.nonzero(coef_mask == 1)[0]:
                value = rel_coefs[i]
                if len(heap_index) < num_vars:
                    heapq.heappush(heap_index, (value, i))
                else:
                    min_heap_tuple = heapq.heappop(heap_index)
                    if value > min_heap_tuple[0]:
                        heapq.heappush(heap_index, (value, i))
                    else:
                        heapq.heappush(heap_index, min_heap_tuple)
            mask_index = list(map(lambda x: x[1], heap_index))
            coef_mask = np.zeros(coefs.shape[0])
            coef_mask[mask_index] = 1.0

        return coef_mask * coefs

    def _initialize_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Initialize the model by standardizing the X and y fitting arrays (private).

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_predictors)
            The input model matrix. The number of columns should be the same number of
            columns as the X argument that was used to fit the model.

        y: numpy.ndarray, shape (n_samples, )
            The target vector

        Returns
        -------
        Xs: numpy.ndarray, shape (n_samples, n_predictors)
            The standardized version of model matrix, X, that is being fitted

        ys: numpy.ndarray, shape (n_samples, )
            The standardized version of target vector, y, that is being fitted
        """
        self._X_means = X.mean(axis=0, keepdims=True)
        self._X_std = X.std(axis=0, keepdims=True)
        self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
        self._y_mean = y.mean()
        self._y_std = y.std()
        self._y_std = 1.0 if self._y_std == 0.0 else self._y_std
        return (X - self._X_means) / self._X_std, (y - self._y_mean) / self._y_std
