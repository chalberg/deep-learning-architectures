import context
import numpy as np
from activation_functions.base import ActivationFunction

class ELU(ActivationFunction):
    """
    Exponential Linear Unit
    f(x) = alpha * (e^x - 1) if x<=0, x if x>0
    """
    def __init__(self, alpha:float):
        self.alpha = alpha

    def run(self, x:float):
        if x <= 0:
            return self.alpha * (np.exp(x) - 1)
        else:
            return x
