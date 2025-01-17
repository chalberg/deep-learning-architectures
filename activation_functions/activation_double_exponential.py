import context
import numpy as np
from activation_functions.base import ActivationFunction

class DoubleExponential(ActivationFunction):
    """
    Double Exponential
    alpha: scalar parameter in the exponent argument
    """
    def __init__(self, alpha:float):
        self.alpha = alpha
        if self.alpha <= 0:
            raise("Parameter alpha must be positive.")

    def run(self, x:float):
        return np.exp(-self.alpha * np.abs(x))