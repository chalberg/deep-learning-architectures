import context
import numpy as np
from activation_functions.base import ActivationFunction

class SELU(ActivationFunction):
    """
    Scaled Exponential Linear Unit
    beta: scalar applied to output
    SELU(x) = beta * ELU(x)
    """
    def __init__(self, alpha:float, beta:float):
        self.alpha = alpha
        self.beta = beta

    def run(self, x:float):
        if x <= 0:
            out = self.alpha * (np.exp(x) - 1)
            return self.beta * out
        else:
            return self.beta * x