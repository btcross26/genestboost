"""
BoostSelect weak learners - weaken learners by boosting
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-19


# import numpy as np
#
# from .. import BoostedModel
#
#
# class FixedBoostSelect:
#     def __init__(self, boosted_model: BoostedModel, iterations=3):
#         self.bmodel = boosted_model
#         self.iterations = iterations
#         self._model = None
#
#     def fit(self, X, y, weights=None):
#         self.bmodel.reset_model()
#         self.bmodel.fit(X, y, weights=weights, iterations=self.iterations)
#         self._model = self.bmodel._model_list[-1][0]
#         return self._model
#
#     def predict(self, X):
#         return self._model.predict(X)
#
#
# class RandomBoostSelect:
#     def __init__(self, boosted_model: BoostedModel, iterations=3):
#         self.bmodel = boosted_model
#         self.iterations = iterations
#         self._model = None
#
#     def fit(self, X, y, weights=None):
#         self.bmodel.reset_model()
#         self.bmodel.fit(X, y, weights=weights, iterations=self.iterations)
#         choice = np.random.choice(self.iterations)
#         self._model = self.bmodel._model_list[choice][0]
#         return self._model
#
#     def predict(self, X):
#         return self._model.predict(X)
