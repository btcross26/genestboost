"""
genestboost.loss_functions module
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-08-26

from .absolute_value import AbsoluteLoss
from .least_squares import LeastSquaresLoss
from .log_cosh import LogCoshLoss
from .log_loss import LogLoss
from .poisson import PoissonLoss
from .students_t import StudentTLoss