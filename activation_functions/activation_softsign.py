import context
import numpy as np
from activation_functions.base import ActivationFunction

class Softsign(ActivationFunction):
    """
    Softsign
    f(x) = x / (1 + |x|)
    """
    def run(self, x:float):
        return x / (1 + np.abs(x))