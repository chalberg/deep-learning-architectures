import context
import numpy as np
from activation_functions.base import ActivationFunction

class SLU(ActivationFunction):
    """
    Sigmoid Linear Unit
    SLU(x) = x /sigma(x) = x / (1 + e^{-c*x})
    alpha: a positive scalar controling the location and size of dip
    """
    def __init__(self, alpha:float=1):
        self.alpha = alpha
        if self.alpha <= 0:
            raise("Parameter c must be positive")

    def run(self, x:float):
        return x / (1 + np.exp(-self.alpha * x))