import context
import numpy as np
from activation_functions.base import ActivationFunction

class Softplus(ActivationFunction):
    """
    Softplus
    sp(x) = ln(1 + e^x)
    """
    def run(self, x:float):
        return np.log(1 + np.exp(x))