Quick Coding Example
====================

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
