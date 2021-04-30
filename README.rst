.. README.rst

.. image:: https://img.shields.io/badge/python-3.8-green.svg
      :target: https://www.python.org
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
      :target: https://github.com/psf/black

genestboost
===========

:code:`genestboost` is an ML boosting library that separates the modeling algorithm from the boosting algorithm. The result is that you can boost any generic regression model, not just trees. Build a forward-thinking (forward-propagating) neural network if you wish, or build an ensemble of support vector machines if you would so desire.

Note that separating the two algorithms may not give the most optimal performance outcomes when it comes to training and prediction speeds. The tool is also programmed in pure Python - for now. Thus, in its current state the library is primarily for research and development. In particular, the library classes can be easily extended to handle custom loss functions and custom link functions. The library can also serve as a foundation for more specialized boosting algorithms when the need to optimize for performance arises.

In the future, the library will be restructured slightly under the hood, and there are plans to parallelize ensemble prediction and move some performance bottlenecks to Nim (i.e., C-extensions).

Quick Demonstration
-------------------
Lorem ipsum.

Installation
------------

Create a virtual environment with Python 3.8 and install from git:

.. code-block::

    $ pip install git+https://https://github.com/btcross26/genestboost.git

Documentation
-------------

Documentation is a work in progress. Current documentation is available on `GitHub Pages <https://github.com/btcross26/genestboost>`_

Bugs / Requests
---------------

Please use the `GitHub issue tracker <https://github.com/btcross26/genestboost/issues>`_ to submit bugs or request features.

Changelog
---------

Consult the `Changelog <https://github.com/btcross26/genestboost/issues>`_ for the latest release information.
