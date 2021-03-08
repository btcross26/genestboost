"""genestboost.loss_functions module."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26


from .absolute_value import AbsoluteLoss
from .base_class import BaseLoss
from .beta_loss import BetaLoss, LeakyBetaLoss
from .least_squares import LeastSquaresLoss
from .log_cosh import LogCoshLoss
from .log_loss import LogLoss
from .poisson import PoissonLoss
from .quantile import QuantileLoss
from .quasilog_loss import QuasiLogLoss
from .students_t import StudentTLoss

__all__ = [
    "AbsoluteLoss",
    "BaseLoss",
    "BetaLoss",
    "LeakyBetaLoss",
    "LeastSquaresLoss",
    "LogCoshLoss",
    "LogLoss",
    "PoissonLoss",
    "QuantileLoss",
    "QuasiLogLoss",
    "StudentTLoss",
]
