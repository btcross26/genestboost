.. README.rst

.. image:: https://img.shields.io/badge/python-3.8-green.svg
      :target: https://www.python.org
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
      :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
      :target: https://opensource.org/licenses/BSD-3-Clause


genestboost
===========

:code:`genestboost` is an ML boosting library that separates the modeling algorithm from the boosting algorithm. The result is that you can boost any generic regression model, not just trees. Build a forward-thinking (forward-propagating) neural network if you wish, or build an ensemble of support vector machines if you would so desire. Mix and match link and loss functions at will.

.. image:: docs/source/images/qr_different_alg_plot.png

Separating the two algorithms may not give the most optimal performance outcomes when it comes to training and prediction speeds. The tool is also programmed in pure Python - for now. Thus, in its current state the library is primarily for research and development. In particular, the library classes can be easily extended to handle custom loss functions and custom link functions. The library can also serve as a foundation for more specialized boosting algorithms when the need to optimize for performance arises.

In the future, the library will be restructured slightly under the hood, and there are plans to parallelize ensemble prediction and move some performance bottlenecks to Nim (i.e., C-extensions). Support for boosting of multivariate targets will be added when time permits.


Examples
--------
- `Quantile Regression with Different Modeling Algorithms <https://btcross26.github.io/genestboost/build/html/quantile_regression_example.html>`_
- `Binary Target Boosting with Custom Model Callback Wrapper <https://btcross26.github.io/genestboost/build/html/binary_target_with_custom_wrapper_example.html>`_
- `BoostedLinearModel with SimplePLS Algorithm <https://btcross26.github.io/genestboost/build/html/boosted_linear_model_example.html>`_
- `Alternative Fitting Procedure with Surrogate Loss Function <https://btcross26.github.io/genestboost/build/html/alternative_fitting_procedure_example.html>`_


Installation
------------

Create a virtual environment with Python 3.8 and install from git:

.. code-block::

    $ pip install git+https://github.com/btcross26/genestboost.git


Documentation
-------------

Documentation is a work in progress. The most recent documentation is available on `GitHub Pages <https://btcross26.github.io/genestboost/build/html/index.html>`_


Bugs / Requests
---------------

Please use the `GitHub issue tracker <https://github.com/btcross26/genestboost/issues>`_ to submit bugs or request features.


Changelog
---------

Consult the `Changelog <https://btcross26.github.io/genestboost/build/html/changelog.html>`_ for the latest release information.
