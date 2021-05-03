"""Implement model ensemble classes."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2021-03-14


# import logging
# from abc import ABC, abstractmethod
# from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
#
# import numpy as np
#
# from .type_hints import Model
#
# LOGGER = logging.getLogger(__name__)
#
# T = TypeVar("T", bound="BaseEnsemble")
#
#
# class BaseEnsemble(ABC):
#     def __init__(self, base_score: float = 0.0) -> None:
#         self._model_list: List[Tuple[Model, float]] = list()
#         self._base_score = base_score
#         self._transform: Callable[[np.ndarray, ...], np.ndarray] = lambda arr: arr
#
#     def __len__(self):
#         return len(self._model_list)
#
#     def __iter__(self):
#         yield from self._model_list
#
#     def __getitem__(self, i):
#         if isinstance(i, int):
#             return self._model_list[i]
#         elif isinstance(i, slice):
#             return self._model_list[i.start : i.stop : i.step]
#
#     @abstractmethod
#     def predict(self, X: np.ndarray, index: Optional[int] = None) -> np.ndarray:
#         ...
#
#     def predict_transform(self, X: np.ndarray, index: Optional[int]):
#         return self._transform(self.predict(X, index))
#
#     def set_transform(self, transform: Callable[[np.ndarray], np.ndarray]) -> None:
#         self._transform = transform
#
#     def get_base_score(self, X: np.ndarray) -> np.ndarray:
#         return self._base_score
#
#     def add_model(self, model: Model, coefficient: float) -> None:
#         self._model_list.append((model, coefficient))
#
#     def add_ensemble(self: T, ensemble: T) -> None:
#         self._base_score += ensemble.get_base_score()
#         for model, coefficient in ensemble:
#             self.add_model(model, coefficient)
#
#     def get_model_by_index(self, index: int) -> Tuple[Model, float]:
#         return self._model_list[index]
#
#     def trim_ensemble(self, num_models_to_keep: int) -> List[Tuple[Model, float]]:
#         trimmed_models = self._model_list[num_models_to_keep:]
#         self._model_list = self._model_list[:num_models_to_keep]
#         return trimmed_models
#
#
# class ForwardAdditiveEnsemble(BaseEnsemble):
#     def __init__(self, model_init: Optional[float] = 0.0, transform=None) -> None:
#         super().__init__(transform)
#         self._model_init = model_init
#
#     def predict(self, X: np.ndarray, index: Optional[int] = None) -> np.ndarray:
#         eta_p = self.base_predict(X)
#         for model, coef in self[:index]:
#             eta_p += coef * model.predict(X)
#         return eta_p
#
#
# class LinearModelEnsemble(ForwardAdditiveEnsemble):
#     def __init__(self):
#
#         self.coef_ = None
#         self.intercept_ = None
#
#     def add_model(self, model: Model, coefficient: float) -> None:
#         # check coef length
#         self._model_list.append((model, coefficient))
#         # update model coefficients
#
#     def _coef_recalculate(self):
#         # update coef_ and intercept_ based on list of models
#         pass
#
#     def predict(
#         self, X: np.ndarray, y: np.ndarray, index: Optional[int] = None
#     ) -> np.ndarray:
#         # issue user warning if index specified that super is called
#         # use matrix multiplication here instead since a linear model
#         eta_p = 0.0 if self._model_init is None else self._model_init.predict(X)
#         for model, coef in self._model_list[:index]:
#             eta_p += coef * model.predict(X)
#         return eta_p
#
#     def trim_ensemble(self, index: int) -> List[Model]:
#         trimmed_models = super().trim_ensemble(index)
#         # update coefs of trimmed models
#         return trimmed_models
