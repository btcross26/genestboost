Changelog
=========


0.1.0-beta
----------

The initial release provides boosting functionality via two main classes: BoostedModel and BoostedLinearModel.

The former is the most general class available and is meant to be used to build a boosted ensemble with any regression algorithm. The key is to provide a callable that returns a model object implementing a fit and predict method. Arguments can be passed to the callable via the model_kwargs_dict argument.

The latter class, BoostedLinearModel, is meant to be used with linear models that have :code:`coef_` and :code:`intercept_` attributes. The class provides additional methods relevant to linear models, and it streamlines prediction of the ensemble by taking advantage of the fact that the sum of linear models is simply a linear model.

Between the beta release and 1.0.0, documentation and examples will be updated. Thus, additional functionality will be exposed and usage of the package will become more transparent. The exposed API may change a little with 1.0, and will become more modular under the hood. It is also expected that additional functionality will be provided.
