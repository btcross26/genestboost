Introduction
============

:code:`genestboost` is an ML boosting library that separates the modeling algorithm from the boosting algorithm. The result is that you can boost any generic regression model, not just trees. Build a forward-thinking (forward-propagating) neural network if you wish, or build an ensemble of support vector machines if you would so desire. Mix and match link and loss functions at will.


How it works
------------

Lorem ipsum.


Limitations
-----------

Separating the boosting and modeling algorithm may not give the most optimal performance outcomes when it comes to training and prediction speeds. The tool is also programmed in pure Python - for now. Thus, in its current state the library is primarily for research and development. In particular, the library classes can be easily extended to handle custom loss functions and custom link functions. The library can also serve as a foundation for more specialized boosting algorithms when the need to optimize for performance arises.

In the future, the library will be restructured slightly under the hood, and there are plans to parallelize ensemble prediction and move some performance bottlenecks to Nim (i.e., C-extensions). Support for boosting of multivariate targets will be added when time permits.
