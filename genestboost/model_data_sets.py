"""
ModelDataSets class for splitting data into training and validation sets
and used in BoostedModel class
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-26


from typing import Iterator, List, Optional

import numpy as np


class ModelDataSets:
    def __init__(
        self,
        X: np.ndarray,
        yt: np.ndarray,
        weights: Optional[np.ndarray] = None,
        validation_fraction: float = 0.0,
        validation_stratify: bool = False,
        random_state: Optional[int] = None,
    ):
        # initialize attributes from init args
        self.X = X
        self.yt = yt
        self.weights = np.ones(yt.shape[0]) if weights is None else weights
        self.validation_fraction = validation_fraction
        self.validation_stratify = validation_stratify
        self._rng = np.random.RandomState(random_state)

        # public vars to be created during class usage
        self.X_train: np.ndarray
        self.yt_train: np.ndarray
        self.weights_train: np.ndarray
        self.X_val: np.ndarray  # if validation_fraction > 0.0
        self.yt_val: np.ndarray  # if validation_fraction > 0.0
        self.weights_val: np.ndarray  # if validation_fraction > 0.0

        # private vars to be created during class usage
        self._tindex: List[int]
        self._vindex: List[int]  # if validation_fraction > 0.0

        # split data sets as necessary
        self._create_index(validation_fraction, validation_stratify)
        self._create_data_sets()

    def has_validation_set(self) -> bool:
        return hasattr(self, "_vindex")

    def _create_data_sets(self) -> None:
        self.X_train = self.X[self._tindex]
        self.yt_train = self.yt[self._tindex]
        self.weights_train = self.weights[self._tindex]

        if self.has_validation_set():
            self.X_val = self.X[self._vindex]
            self.yt_val = self.yt[self._vindex]
            self.weights_val = self.weights[self._vindex]

    def _create_index(
        self, validation_fraction: float, validation_stratify: bool
    ) -> None:
        # initialize training index
        self._tindex = np.arange(self.yt.shape[0])

        # create validation index if specified
        if validation_fraction > 0.0:
            if validation_stratify:
                self._vindex = np.hstack(
                    [self._stratify_groups_generator(validation_fraction)]
                )
            else:
                n = self.yt.shape[0]
                self._vindex = np.random.choice(
                    n, int(validation_fraction * n), replace=False
                )
            self._vindex.sort()
            self._tindex = np.setdiff1d(self._tindex, self._vindex, assume_unique=True)

    def _stratify_groups_generator(
        self, validation_fraction: float
    ) -> Iterator[np.array]:
        full_index = np.arange(self.yt.shape[0])
        groups, group_index = np.unique(self.yt, return_inverse=True)
        for group in groups:
            mask = group_index == group
            n = mask.sum()
            g_index = self._rng.choice(
                full_index[mask], int(validation_fraction * n), replace=False
            )
            yield g_index
