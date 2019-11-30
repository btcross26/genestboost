"""
Implementation for SimplePLS, a one partial-PLS component regression model
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-28


import heapq
import numpy as np


class SimplePLS:
    """
    Implementation for SimplePLS, a one partial-PLS component regression model
    """
    def __init__(self, max_vars=None, filter_threshold=None):
        self._max_vars = max_vars
        self._filter_threshold = filter_threshold
        self._X_means = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None
        self._multiplier = None
        self._pls_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self._is_fit = False

    def fit(self, X, y, weights=None):
        self._is_fit = False
        Xs, ys = self._standardize_Xy(X, y)
        coefs = np.mean(Xs * ys.reshape((-1, 1)), axis=0)
        coefs = self._mask_coefs(coefs)
        self._multiplier = self._get_coef_multiplier(coefs, Xs, ys)
        coefs *= self._multiplier
        coefs *= self._y_std / self._X_std[0]
        self.coef_ = coefs
        self.intercept_ = (
            self._y_mean
            - np.sum(coefs * self._X_means[0])
            - self._pls_intercept * self._y_std
        )
        return self

    def predict(self, X):
        return self.intercept_ + X.dot(self.coef_)

    def _get_coef_multiplier(self, coefs, Xs, ys):
        n_coefs = np.sum(coefs != 0.0)
        if n_coefs == 1:
            self._pls_intercept = 0.0
            return 1.0
        else:
            x_pls = Xs.dot(coefs)
            x_pls_mean = x_pls.mean()
            x_pls_std = x_pls.std()
            x_pls_std = np.where(x_pls_std == 0.0, 1.0, x_pls_std)
            x_pls_scaled = (x_pls - x_pls_mean) / x_pls_std
            alpha = np.mean(x_pls_scaled * ys)
            self._pls_intercept = alpha * x_pls_mean / x_pls_std
            multiplier = alpha / x_pls_std
            return multiplier

    def _mask_coefs(self, coefs):
        coefs_abs = np.abs(coefs)
        max_index = np.argmax(coefs_abs)

        # edge cases
        if (
            self._max_vars == 1
            or (self._filter_threshold is not None and self._filter_threshold > 1.0)
        ):
            coef_mask = np.zeros_like(coefs)
            coef_mask[max_index] = 1.0
            return coefs * coef_mask
        elif (
            (self._max_vars is None or self._max_vars >= coefs.shape[0])
            and self._filter_threshold is None
        ):
            return coefs

        # apply correlation filter
        max_value = coefs_abs[max_index]
        rel_coefs = coefs_abs / max_value
        if self._filter_threshold is not None:
            coef_mask = 1.0 * (rel_coefs >= self._filter_threshold)
        else:
            coef_mask = np.ones_like(coefs)

        # apply max_vars intermediate case if specified
        if self._max_vars is not None and coef_mask.sum() > self._max_vars:
            heap_index = list()
            for i in np.nonzero(coef_mask == 1)[0]:
                value = rel_coefs[i]
                if len(heap_index) < self._max_vars:
                    heapq.heappush(heap_index, (value, i))
                else:
                    min_heap_tuple = heapq.heappop(heap_index)
                    if value > min_heap_tuple[0]:
                        heapq.heappush(heap_index, (value, i))
                    else:
                        heapq.heappush(heap_index, min_heap_tuple)
            heap_index = list(map(lambda x: x[1], heap_index))
            coef_mask = np.zeros_like(coefs)
            coef_mask[heap_index] = 1.0

        return coef_mask * coefs

    def _standardize_Xy(self, X, y):
        if not self._is_fit:
            self._X_means = X.mean(axis=0, keepdims=True)
            self._X_std = X.std(axis=0, keepdims=True)
            self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
            self._y_mean = y.mean()
            self._y_std = y.std()
            self._y_std = 1.0 if self._y_std == 0.0 else self._y_std
            self._is_fit = True
        return (X - self._X_means) / self._X_std, (y - self._y_mean) / self._y_std
