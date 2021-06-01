.. README.rst

.. image:: https://img.shields.io/badge/python-3.7-green.svg
      :target: https://www.python.org
.. image:: https://img.shields.io/badge/python-3.8-green.svg
      :target: https://www.python.org
.. image:: https://img.shields.io/badge/python-3.9-green.svg
      :target: https://www.python.org
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
      :target: https://github.com/psf/black
.. image:: https://github.com/btcross26/genestboost/workflows/build_tests/badge.svg
      :target: https://github.com/btcross26/genestboost/actions/build_tests
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
      :target: https://opensource.org/licenses/BSD-3-Clause
.. image:: https://badge.fury.io/py/genestboost.svg
      :target: https://pypi.python.org/pypi/genestboost
.. image:: https://img.shields.io/conda/vn/conda-forge/genestboost.svg
      :target: https://anaconda.org/conda-forge/genestboost


|
.. image:: https://user-images.githubusercontent.com/7505706/120132584-968dd780-c198-11eb-8843-55bc23310657.png


`Documentation Home <https://btcross26.github.io/genestboost/build/html/index.html>`__ | `Quick Coding Example`_ | `Additional Examples`_ | `Limitations`_ | `Installation`_ | `Changelog <https://github.com/btcross26/genestboost/blob/main/changelog.rst>`__


:code:`genestboost` is an ML boosting library that separates the modeling algorithm from the boosting algorithm. The result is that you can boost any generic regression model, not just trees. Build a forward-thinking (forward-propagating) neural network if you wish, or build an ensemble of support vector machines if you would so desire. Mix and match link and loss functions at will.


Quick Coding Example
--------------------

Boost simple neural networks to predict a binary target:

.. code-block:: python

    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt

    from genestboost import BoostedModel
    from genestboost.loss_functions import LogLoss
    from genestboost.link_functions import LogitLink

    # generate a dummy dataset - the library expects numpy arrays of dtype float
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        weights=[0.90, 0.10],
        random_state=17,
    )

    # create a boosted model instance
    model = BoostedModel(
        link=LogitLink(),                  # link function to use
        loss=LogLoss(),                    # loss function to use
        model_callback=MLPRegressor,       # callback creates model with fit, predict
        model_callback_kwargs={            # keyword arguments to the callback
            "hidden_layer_sizes": (16,),
            "max_iter": 1000,
            "alpha": 0.2,
        },
        weights="newton",                  # newton = scale gradients with second derivatives
        alpha=1.0,                         # initial learning rate to try
        step_type="decaying",              # learning rate type
        step_decay_factor=0.50,            # learning rate decay factor
        validation_fraction=0.20,          # fraction of training set to use for holdout
        validation_iter_stop=5,            # stopping criteria
        validation_stratify=True,          # stratify the holdout set by the target (classification)
    )

    # fit the model
    model.fit(X, y, min_iterations=10, iterations=100)

    # evaluate the model
    print(model.get_iterations())
    predictions = model.predict(X)        # predicted y's (probabilities in this case)
    scores = model.decision_function(X)   # predicted links (logits in this case)
    plt.plot(model.get_loss_history(), label=["Training", "Holdout"])
    plt.legend(loc="best")


Additional Examples
-------------------
- `Quantile Regression with Different Modeling Algorithms <https://btcross26.github.io/genestboost/build/html/quantile_regression_example.html>`_
- `Binary Target Boosting with Custom Model Callback Wrapper <https://btcross26.github.io/genestboost/build/html/binary_target_with_custom_wrapper_example.html>`_
- `BoostedLinearModel with SimplePLS Algorithm <https://btcross26.github.io/genestboost/build/html/boosted_linear_model_example.html>`_
- `Alternative Fitting Procedure with Surrogate Loss Function <https://btcross26.github.io/genestboost/build/html/alternative_fitting_procedure_example.html>`_
- `Forward Propagating Neural Network <https://btcross26.github.io/genestboost/build/html/forward_neural_network_example.html>`_


Limitations
-----------

Separating the boosting and modeling algorithm may not give the most optimal performance outcomes when it comes to training and prediction speeds. The tool is also programmed in pure Python - for now. Thus, in its current state the library is primarily for research and development. In particular, the library classes can be easily extended to handle custom loss functions and custom link functions. The library can also serve as a foundation for more specialized boosting algorithms when the need to optimize for performance arises.

In the future, the library will be restructured slightly under the hood, and there are plans to parallelize ensemble prediction and move some performance bottlenecks to Nim (i.e., C-extensions). Support for boosting of multivariate targets will be added when time permits.


Installation
------------

Create a virtual environment with Python >=3.7,<=3.9, and install from git:

.. code-block::

    $ pip install git+https://github.com/btcross26/genestboost.git

Alternatively, you can install directly from PyPI:

.. code-block:: bash

    $ pip install genestboost

Or from conda-forge:

.. code-block:: bash

    $ conda install -c conda-forge genestboost


Documentation
-------------

Documentation is a work in progress. The most recent documentation is available on `GitHub Pages <https://btcross26.github.io/genestboost/build/html/index.html>`_.


Bugs / Requests
---------------

Please use the `GitHub issue tracker <https://github.com/btcross26/genestboost/issues>`_ to submit bugs or request features.


Changelog
---------

Consult the `Changelog <https://btcross26.github.io/genestboost/build/html/changelog.html>`_ for the latest release information.


Contributing
------------

If you would like to contribute, please fork this repository, create a branch off of :code:`main` for your contribution, and submit a PR to the :code:`dev_staging` branch. Also, please create an issue describing the nature of the contribution if it has not already been done.
