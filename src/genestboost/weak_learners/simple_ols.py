"""
Implementation for simple OLS, i.e. one-variable regression model
"""


import numpy as np


class SimpleOLS:
    """
    Implementation for simple OLS, i.e. one-variable regression model. This class
    returns a single coefficient regression without an intercept. Thus if regression
    through the origin is not desired, then
    """
    def __init__(self):
        self._X_means = None
        self._X_std = None
        self._y_mean = None
        self.coef_ = None
        self.coef_index_ = None
        self._is_fit = False

    def _standardize_Xy(self, X, y):
        if not self._is_fit:
            self._X_means = X.mean(axis=0, keepdims=True)
            self._X_std = X.std(axis=0, keepdims=True)
            self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
            self._y_mean = y.mean()
            self._is_fit = True
        return (X - self._X_means) / self._X_std, y - self._y_mean

    def fit(self, X, y, weights=None):
        self._is_fit = False
        Xs, ys = self._standardize_Xy(X, y)
        coefs = np.sum(Xs * ys.reshape((-1, 1)) / (Xs.shape[0] - 1), axis=0)
        self.coef_index_ = np.argmax(coefs)
        self.coef_ = coefs[self.coef_index_]
        return self

    def predict(self, X):
        X = (X - self._X_means) / self._X_std
        return self.coef_ * X[:, self.coef_index_] + self._y_mean

    @classmethod
    def clone(cls):
        return cls()
